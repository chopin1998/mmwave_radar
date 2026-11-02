#!/usr/bin/env python3
"""
Dual-UART Serial Manager & Background Streaming Thread for IWR6843.

Supports:
  1. Hardware / PTY Dual-UART mode (CLI @ 115200 bps, Data @ 921600 bps)
  2. Built-in Mock Simulation mode (feeds synthetic binary TLV byte chunks through TLVStreamParser)
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import serial
from PySide6.QtCore import QThread, Signal

from host.mock_generator import MockRadarGenerator
from host.tlv_parser import RadarConfig, RadarFrame, TLVStreamParser


class RadarSerialWorker(QThread):
    """Background worker thread for CLI configuration and 921600 bps TLV streaming."""

    frame_received = Signal(object)  # emits RadarFrame
    cli_log = Signal(str)  # emits CLI console lines
    status_changed = Signal(str)  # emits status bar text
    stats_updated = Signal(int, int, int, float)  # frames_ok, sync_losses, tlv_errs, fps

    def __init__(
        self,
        config: Optional[RadarConfig] = None,
        mock_mode: bool = True,
        cli_port: str = "/dev/ttyUSB0",
        data_port: str = "/dev/ttyUSB1",
        cli_baud: int = 115200,
        data_baud: int = 921600,
        fragment_mock_packets: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or RadarConfig()
        self.mock_mode = mock_mode
        self.cli_port = cli_port
        self.data_port = data_port
        self.cli_baud = cli_baud
        self.data_baud = data_baud
        self.fragment_mock_packets = fragment_mock_packets

        self._running = False
        self._cli_ser: Optional[serial.Serial] = None
        self._data_ser: Optional[serial.Serial] = None
        self.parser = TLVStreamParser(config=self.config)
        self.mock_gen = MockRadarGenerator(config=self.config, enable_azimuth_heatmap=True)

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
        """Parse and transmit configuration commands."""
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
            with serial.Serial(self.cli_port, self.cli_baud, timeout=1.0) as cli:
                cli.reset_input_buffer()
                for ln in lines:
                    cmd_bytes = (ln + "\n").encode("ascii")
                    cli.write(cmd_bytes)
                    cli.flush()
                    self.cli_log.emit(f">> {ln}")

                    # Wait up to 0.8s for 'Done' or 'Error'
                    deadline = time.monotonic() + 0.8
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
                                self.cli_log.emit(f"[CLI ERROR] Command failed: {ln}")
                                return False
                        else:
                            time.sleep(0.01)
                    time.sleep(0.03)
            self.status_changed.emit(f"CLI Config Sent Successfully on {self.cli_port}")
            return True
        except serial.SerialException as exc:
            self.cli_log.emit(f"[UART ERROR] CLI port {self.cli_port}: {exc}")
            self.status_changed.emit(f"CLI Error: {exc}")
            return False

    def send_cli_command(self, cmd: str, wait_s: float = 0.6) -> bool:
        """Send a single CLI command over the 115200 bps User UART (or log in Mock mode)."""
        cmd_clean = cmd.strip()
        if not cmd_clean:
            return True

        if self.mock_mode:
            self.cli_log.emit(f"[MOCK CLI] mmwDemo:/>{cmd_clean}\n  Done")
            return True

        try:
            with serial.Serial(self.cli_port, self.cli_baud, timeout=0.5) as cli:
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
            return False

    def stop(self, deep_power_down: bool = False) -> None:
        self._running = False
        self.wait(2000)
        if deep_power_down:
            # Stop RF + shut down DSP power, BSS clock, RF analog, APLL, and scale MSS VCLK to 40MHz
            self.send_cli_command("idlePowerDown -1 1 0 1 1 1 0 0 0 1000000", wait_s=0.8)
            self.status_changed.emit("Stopped (Hardware Deep Power Down - Press NRST to wake)")
        else:
            # Stop 60 GHz RF chirps, HWA/DSP frame loop, and Data UART while keeping CLI responsive
            self.send_cli_command("sensorStop", wait_s=0.6)
            self.status_changed.emit("Stopped (Sensor Idle / RF Off)")

    def run(self) -> None:
        self._running = True
        self.parser.reset()
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

        # Real / PTY Data UART Mode
        try:
            self._data_ser = serial.Serial(
                self.data_port,
                self.data_baud,
                timeout=0.05,
            )
            self._data_ser.reset_input_buffer()
            self.status_changed.emit(f"Streaming: Data UART {self.data_port} @ {self.data_baud} bps")
        except serial.SerialException as exc:
            self.cli_log.emit(f"[UART ERROR] Cannot open Data UART {self.data_port}: {exc}")
            self.status_changed.emit(f"Data UART Error: {exc}")
            self._running = False
            return

        try:
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
                self._data_ser.close()
            self.status_changed.emit("Stopped")
