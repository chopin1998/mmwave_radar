#!/usr/bin/env python3
"""
Dual-UART Serial Manager & Background Streaming Thread for IWR6843.

Supports:
  1. Hardware / PTY Dual-UART mode (CLI @ 115200 bps, Data @ 921600 bps)
  2. Built-in Mock Simulation mode (feeds synthetic binary TLV byte chunks through TLVStreamParser)
"""
from __future__ import annotations

import os
import time
from typing import Optional

import serial
from PySide6.QtCore import QThread, Signal

from host.mock_generator import MockRadarGenerator
from host.tlv_parser import RadarConfig, RadarFrame, TLVStreamParser


class RadarSerialWorker(QThread):
    """Background worker thread for non-blocking CLI configuration and 921600 bps TLV streaming."""

    frame_received = Signal(object)  # emits RadarFrame
    cli_log = Signal(str)  # emits CLI console lines
    status_changed = Signal(str)  # emits status bar text
    stats_updated = Signal(int, int, int, float)  # frames_ok, sync_losses, tlv_errs, fps
    stream_stopped = Signal()  # emitted when thread finishes or aborts

    def __init__(
        self,
        config: Optional[RadarConfig] = None,
        mock_mode: bool = True,
        cli_port: str = "/dev/ttyUSB0",
        data_port: str = "/dev/ttyUSB1",
        cli_baud: int = 115200,
        data_baud: int = 921600,
        fragment_mock_packets: bool = True,
        cfg_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config or RadarConfig()
        self.mock_mode = mock_mode
        self.cli_port = cli_port
        self.data_port = data_port
        self.cli_baud = cli_baud
        self.data_baud = data_baud
        self.fragment_mock_packets = fragment_mock_packets
        self.cfg_path = cfg_path

        if cfg_path and os.path.exists(cfg_path):
            try:
                self.config = RadarConfig.from_cfg_file(cfg_path)
            except Exception:
                pass

        self._running = False
        self._cli_ser: Optional[serial.Serial] = None
        self._data_ser: Optional[serial.Serial] = None
        self.parser = TLVStreamParser(config=self.config)
        self.mock_gen = MockRadarGenerator(config=self.config, enable_azimuth_heatmap=True)

    def _ensure_cli_open(self) -> serial.Serial:
        """Keep CLI serial port open across commands to avoid Linux DTR drop resets."""
        if self._cli_ser is None or not self._cli_ser.is_open:
            self._cli_ser = serial.Serial(
                self.cli_port,
                self.cli_baud,
                timeout=0.15,
                write_timeout=0.5,
                dsrdtr=False,
                rtscts=False,
            )
            # Ensure DTR is asserted so MCU is not held in reset/bootloader
            self._cli_ser.dtr = True
            self._cli_ser.rts = False
        return self._cli_ser

    def _close_cli(self) -> None:
        if self._cli_ser is not None and self._cli_ser.is_open:
            try:
                self._cli_ser.close()
            except Exception:
                pass
            self._cli_ser = None

    def send_cfg_file(self, cfg_path: str) -> bool:
        """Send a .cfg file line-by-line over the 115200 bps User UART (or apply to Mock)."""
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            self.cli_log.emit(f"[ERROR] Failed to read {cfg_path}: {exc}")
            return False

        return self.send_cfg_text(text)

    def send_cfg_text(self, text: str) -> bool:
        """Parse and transmit configuration commands while keeping CLI port alive."""
        self.config = RadarConfig.from_cfg_text(text)
        self.parser.config = self.config
        self.mock_gen.config = self.config

        lines = [
            ln.strip()
            for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("%")
        ]

        if self.mock_mode:
            self.cli_log.emit("[MOCK CLI] Applying radar configuration to simulator:")
            for ln in lines:
                self.cli_log.emit(f"  mmwDemo:/>{ln}\n  Done")
            self.status_changed.emit(
                f"Mock Configured: {self.config.num_range_bins} Range Bins, "
                f"ΔR={self.config.range_resolution_m*100:.2f} cm, "
                f"T_frame={self.config.frame_periodicity_ms:.0f} ms"
            )
            return True

        try:
            cli = self._ensure_cli_open()
            cli.reset_input_buffer()

            # Pre-flight handshake: wake CLI prompt and check if MCU is in 40MHz Deep Sleep (23040 bps -> 0xF0)
            cli.write(b"\n")
            cli.flush()
            time.sleep(0.08)
            probe_bytes = cli.read(256)
            if b"\xf0" in probe_bytes and b"mmwDemo" not in probe_bytes:
                self.cli_log.emit(
                    "[ERROR] 检测到雷达正处于 40MHz Deep Sleep 深度休眠模式 (UART 波特率已降为 23040 bps，DSP 已断电)！\n"
                    "  >>> 解决方法：请按一下雷达开发板上的【NRST 物理复位按键】唤醒雷达，然后再点击 Send .cfg & Start Radar。"
                )
                self.status_changed.emit(
                    "Radar is in Deep Sleep (40MHz) — Please press NRST button on board first!"
                )
                return False

            consecutive_timeouts = 0
            for ln in lines:
                if not self._running and self.isRunning():
                    self.cli_log.emit("[INFO] Configuration upload cancelled by user.")
                    return False

                cli.reset_input_buffer()
                cmd_bytes = (ln + "\n").encode("ascii")
                cli.write(cmd_bytes)
                cli.flush()
                self.cli_log.emit(f">> {ln}")

                is_sensor_start = ln.startswith("sensorStart")
                wait_timeout = 2.0 if is_sensor_start else 0.55
                deadline = time.monotonic() + wait_timeout
                resp_acc = ""
                got_ack = False
                saw_f0_sleep = False

                while time.monotonic() < deadline:
                    if not self._running and self.isRunning():
                        return False
                    raw_bytes = cli.read_until(b"\n")
                    if raw_bytes:
                        if b"\xf0" in raw_bytes and b"mmwDemo" not in raw_bytes and b"Done" not in raw_bytes:
                            saw_f0_sleep = True
                        raw = raw_bytes.decode("utf-8", errors="ignore")
                        resp_acc += raw
                        for rline in raw.splitlines():
                            rstrip = rline.strip()
                            if rstrip:
                                self.cli_log.emit(f"<< {rstrip}")
                        if "Done" in resp_acc or "Ignored" in resp_acc:
                            got_ack = True
                            consecutive_timeouts = 0
                            break
                        if "Error" in resp_acc:
                            self.cli_log.emit(
                                f"[CLI ERROR] Command rejected by radar: {ln}\n"
                                "  >>> 若之前已下发过不同配置，请按一下板载 NRST 复位键后重试。"
                            )
                            self.status_changed.emit(
                                f"CLI Error on '{ln}' (Press NRST on board and retry)"
                            )
                            return False
                    else:
                        time.sleep(0.01)

                if saw_f0_sleep and not got_ack:
                    self.cli_log.emit(
                        "[ERROR] 雷达串口返回 0xF0（处于 40MHz Deep Sleep 休眠状态，DSP 已关闭）！\n"
                        "  >>> 请按一下雷达板上的【NRST 复位键】唤醒后再点击启动。"
                    )
                    self.status_changed.emit(
                        "Radar in Deep Sleep (40MHz) — Press NRST on board to wake!"
                    )
                    return False

                if not got_ack and not is_sensor_start:
                    consecutive_timeouts += 1
                    self.cli_log.emit(f"[WARN] No response for: {ln}")
                    if consecutive_timeouts >= 2:
                        self.cli_log.emit(
                            f"[ERROR] 串口 {self.cli_port} 连续无响应！\n"
                            "  >>> 请按一下雷达板上的【NRST 复位键】后重试。"
                        )
                        self.status_changed.emit(
                            f"No CLI response on {self.cli_port} — Press NRST on board and retry!"
                        )
                        return False
                time.sleep(0.025)

            self.status_changed.emit(f"CLI Config Sent Successfully on {self.cli_port}")
            return True
        except serial.SerialException as exc:
            self.cli_log.emit(f"[UART ERROR] CLI port {self.cli_port}: {exc}")
            self.status_changed.emit(f"CLI Error: {exc}")
            self._close_cli()
            return False

    def send_cli_command(self, cmd: str, wait_s: float = 0.5) -> bool:
        """Send a single CLI command over the 115200 bps User UART (or log in Mock mode)."""
        cmd_clean = cmd.strip()
        if not cmd_clean:
            return True

        if self.mock_mode:
            self.cli_log.emit(f"[MOCK CLI] mmwDemo:/>{cmd_clean}\n  Done")
            return True

        try:
            cli = self._ensure_cli_open()
            cli.reset_input_buffer()
            cli.write((cmd_clean + "\n").encode("ascii"))
            cli.flush()
            self.cli_log.emit(f">> {cmd_clean}")

            deadline = time.monotonic() + wait_s
            resp_acc = ""
            while time.monotonic() < deadline:
                raw = cli.read_until(b"\n").decode("utf-8", errors="ignore")
                if raw:
                    resp_acc += raw
                    for rline in raw.splitlines():
                        rstrip = rline.strip()
                        if rstrip:
                            self.cli_log.emit(f"<< {rstrip}")
                    if "Done" in resp_acc or "Ignored" in resp_acc:
                        break
                    if "Error" in resp_acc:
                        return False
                else:
                    time.sleep(0.01)
            return True
        except serial.SerialException as exc:
            self.cli_log.emit(f"[UART ERROR] CLI port {self.cli_port}: {exc}")
            self._close_cli()
            return False

    def stop(self, deep_power_down: bool = False) -> None:
        self._running = False
        self.wait(1500)
        if deep_power_down:
            self.send_cli_command("idlePowerDown -1 1 0 1 1 1 0 0 0 1000000", wait_s=0.6)
            self.status_changed.emit("Stopped (Hardware Deep Power Down - Press NRST to wake)")
        else:
            self.send_cli_command("sensorStop", wait_s=0.4)
            self.status_changed.emit("Stopped (Sensor Idle / RF Off)")
        self._close_cli()

    def run(self) -> None:
        self._running = True
        self.parser.reset()

        try:
            # Open Data UART first (in HW mode) and drain any stale bytes before sending .cfg
            if not self.mock_mode:
                try:
                    self._data_ser = serial.Serial(
                        self.data_port,
                        self.data_baud,
                        timeout=0.05,
                    )
                    self._data_ser.reset_input_buffer()
                except serial.SerialException as exc:
                    self.cli_log.emit(f"[UART ERROR] Cannot open Data UART {self.data_port}: {exc}")
                    self.status_changed.emit(f"Data UART Error: {exc}")
                    self._running = False
                    return

            # Send .cfg inside background worker thread so GUI never freezes
            if self.cfg_path and os.path.exists(self.cfg_path):
                self.status_changed.emit(f"Sending configuration {os.path.basename(self.cfg_path)}...")
                ok = self.send_cfg_file(self.cfg_path)
                if not ok and not self.mock_mode:
                    self._running = False
                    return

            t_fps_start = time.monotonic()
            frames_since_fps = 0
            current_fps = 0.0

            if self.mock_mode:
                self.status_changed.emit("Streaming: Mock Radar Binary TLV Generator (Active)")
                while self._running:
                    t0 = time.monotonic()
                    period_s = max(0.02, self.config.frame_periodicity_ms * 1e-3)
                    pkt = self.mock_gen.next_frame_bytes(dt_s=period_s)

                    chunks = self.mock_gen.chunk_packet(pkt) if self.fragment_mock_packets else [pkt]
                    for chunk in chunks:
                        decoded_frames = self.parser.feed(chunk)
                        for frame in decoded_frames:
                            self.frame_received.emit(frame)
                            frames_since_fps += 1

                    now = time.monotonic()
                    if now - t_fps_start >= 0.5:
                        current_fps = frames_since_fps / (now - t_fps_start)
                        frames_since_fps = 0
                        t_fps_start = now
                        self.stats_updated.emit(
                            self.parser.frames_parsed,
                            self.parser.sync_losses,
                            self.parser.tlv_errors,
                            current_fps,
                        )

                    elapsed = time.monotonic() - t0
                    sleep_s = max(0.002, period_s - elapsed)
                    time.sleep(sleep_s)
                self.status_changed.emit("Stopped")
                return

            # Real / PTY Data UART Streaming Loop
            if self._data_ser is not None:
                self._data_ser.reset_input_buffer()
                self.status_changed.emit(f"Streaming: Data UART {self.data_port} @ {self.data_baud} bps")
                while self._running:
                    waiting = self._data_ser.in_waiting
                    raw = self._data_ser.read(max(waiting, 256))
                    if raw:
                        decoded_frames = self.parser.feed(raw)
                        for frame in decoded_frames:
                            self.frame_received.emit(frame)
                            frames_since_fps += 1

                    now = time.monotonic()
                    if now - t_fps_start >= 0.5:
                        current_fps = frames_since_fps / (now - t_fps_start)
                        frames_since_fps = 0
                        t_fps_start = now
                        self.stats_updated.emit(
                            self.parser.frames_parsed,
                            self.parser.sync_losses,
                            self.parser.tlv_errors,
                            current_fps,
                        )
        except serial.SerialException as exc:
            self.cli_log.emit(f"[UART ERROR] Data stream interrupted: {exc}")
        finally:
            if self._data_ser and self._data_ser.is_open:
                try:
                    self._data_ser.close()
                except Exception:
                    pass
                self._data_ser = None
            self.stream_stopped.emit()
