#!/usr/bin/env python3
"""
Linux PTY Virtual Dual-UART IWR6843 Radar Emulator.

Creates two pseudo-terminal symlink nodes:
  - Control UART (CLI): /tmp/ttyMockRadarCLI (responds to .cfg commands with 'Done\\nmmwDemo:/>')
  - Logging UART (DATA): /tmp/ttyMockRadarDATA (streams binary TLV frames after 'sensorStart')

Allows full end-to-end testing of serial port configuration, .cfg handshake, and binary
TLV streaming without physical radar hardware attached.
"""
from __future__ import annotations

import argparse
import os
import pty
import select
import signal
import sys
import time

from host.mock_generator import MockRadarGenerator
from host.tlv_parser import RadarConfig


def run_pty_emulator(
    cli_link: str = "/tmp/ttyMockRadarCLI",
    data_link: str = "/tmp/ttyMockRadarDATA",
    auto_start: bool = False,
) -> None:
    cli_master, cli_slave = pty.openpty()
    data_master, data_slave = pty.openpty()

    cli_slave_name = os.ttyname(cli_slave)
    data_slave_name = os.ttyname(data_slave)

    for link, target in [(cli_link, cli_slave_name), (data_link, data_slave_name)]:
        try:
            if os.path.islink(link) or os.path.exists(link):
                os.unlink(link)
            os.symlink(target, link)
        except OSError as exc:
            print(f"[WARN] Could not create symlink {link}: {exc}", file=sys.stderr)

    print("======================================================================")
    print("  IWR6843 Virtual Dual-UART Emulator Running (PTY Mode)")
    print(f"  User/CLI Port  (115200): {cli_link} -> {cli_slave_name}")
    print(f"  Data/TLV Port  (921600): {data_link} -> {data_slave_name}")
    print("======================================================================")
    sys.stdout.flush()

    cfg = RadarConfig()
    gen = MockRadarGenerator(config=cfg, enable_azimuth_heatmap=True)
    sensor_running = auto_start
    cfg_lines: list[str] = []
    rx_buf = b""
    last_frame_time = time.monotonic()

    def cleanup(*_args: object) -> None:
        for link in (cli_link, data_link):
            if os.path.islink(link):
                try:
                    os.unlink(link)
                except OSError:
                    pass
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        while True:
            rlist, _, _ = select.select([cli_master], [], [], 0.01)
            if cli_master in rlist:
                try:
                    chunk = os.read(cli_master, 1024)
                except OSError:
                    chunk = b""
                if chunk:
                    rx_buf += chunk
                    while b"\n" in rx_buf or b"\r" in rx_buf:
                        sep_idx = min(
                            [i for i in (rx_buf.find(b"\n"), rx_buf.find(b"\r")) if i >= 0]
                        )
                        line = rx_buf[:sep_idx].decode("utf-8", errors="ignore").strip()
                        rx_buf = rx_buf[sep_idx + 1 :]
                        if not line:
                            os.write(cli_master, b"\r\nmmwDemo:/>")
                            continue

                        # Echo command and process
                        reply = f"{line}\r\nDone\r\nmmwDemo:/>"
                        if line.startswith("%"):
                            reply = f"{line}\r\nmmwDemo:/>"
                        elif line.startswith("sensorStop"):
                            sensor_running = False
                            cfg_lines.clear()
                        elif line.startswith("sensorStart"):
                            cfg = RadarConfig.from_cfg_text("\n".join(cfg_lines))
                            gen.config = cfg
                            sensor_running = True
                            print(
                                f"[PTY Radar] sensorStart: rangeBins={cfg.num_range_bins}, "
                                f"period={cfg.frame_periodicity_ms} ms"
                            )
                        elif line.startswith("version"):
                            reply = (
                                "Platform                : xWR68xx\r\n"
                                "mmWave SDK Version      : 03.06.02.00\r\n"
                                "Done\r\nmmwDemo:/>"
                            )
                        else:
                            cfg_lines.append(line)

                        os.write(cli_master, reply.encode("utf-8"))

            now = time.monotonic()
            period_s = max(0.02, cfg.frame_periodicity_ms * 1e-3)
            if sensor_running and (now - last_frame_time) >= period_s:
                last_frame_time = now
                pkt = gen.next_frame_bytes(dt_s=period_s)
                try:
                    os.write(data_master, pkt)
                except OSError:
                    pass
    finally:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IWR6843 PTY Dual-UART Emulator")
    parser.add_argument("--cli-link", default="/tmp/ttyMockRadarCLI")
    parser.add_argument("--data-link", default="/tmp/ttyMockRadarDATA")
    parser.add_argument("--auto-start", action="store_true")
    args = parser.parse_args()
    run_pty_emulator(args.cli_link, args.data_link, args.auto_start)
