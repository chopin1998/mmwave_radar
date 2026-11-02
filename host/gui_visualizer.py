#!/usr/bin/env python3
"""
IWR6843 mmWave Radar Stage-1 Visualizer & Host Controller (PySide6 + PyQtGraph).

Features:
  - Dual-mode operation:
      1. Built-in Mock Radar Simulator (offline test with fragmented binary TLV stream)
      2. Hardware / PTY Dual-UART (115200 CLI .cfg sender + 921600 Data TLV receiver)
  - Real-time visualization:
      - 2D X-Y Bird's-Eye Point Cloud (with radial range rings & FoV bounds)
      - 3D Point Cloud (OpenGL GLViewWidget with automatic 2D Range-Doppler/Elevation fallback)
      - 1D Range Profile & Noise Floor Envelope (dB vs Range in meters)
      - 2D Range-Azimuth Static Heatmap (2D spatial FFT beamforming on TLV Type 4 cmplx16ImRe_t)
  - Real-time DSP & UART telemetry (Inter-frame time, UART TX time, Margin, CPU Load, RF Temp)
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from host.serial_worker import RadarSerialWorker
from host.tlv_parser import RadarConfig, RadarFrame

# Optional OpenGL 3D support
try:
    import pyqtgraph.opengl as gl

    HAS_OPENGL = True
except Exception:
    HAS_OPENGL = False


DEFAULT_CFG_PATH = str(
    Path(__file__).resolve().parent.parent
    / "firmware"
    / "xwr68xx_mmw"
    / "profiles"
    / "profile_3d.cfg"
)


class RadarVisualizerWindow(QMainWindow):
    """Main PySide6 + PyQtGraph GUI window for IWR6843 OOB Demo."""

    def __init__(
        self,
        initial_mock: bool = True,
        cli_port: str = "/dev/ttyUSB0",
        data_port: str = "/dev/ttyUSB1",
        cfg_path: str = DEFAULT_CFG_PATH,
        auto_start: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle("TI IWR6843 mmWave Radar Visualizer (OOB TLV + Mock Simulator)")
        self.resize(1480, 920)

        pg.setConfigOptions(antialias=True, background="#11161d", foreground="#d8e2ec")

        self.config = (
            RadarConfig.from_cfg_file(cfg_path)
            if os.path.exists(cfg_path)
            else RadarConfig()
        )
        self.worker: Optional[RadarSerialWorker] = None

        self._build_ui(initial_mock, cli_port, data_port, cfg_path)

        if auto_start:
            self.start_streaming()

    def _build_ui(
        self,
        initial_mock: bool,
        cli_port: str,
        data_port: str,
        cfg_path: str,
    ) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ---------------- Left Control Panel ----------------
        left_panel = QWidget()
        left_panel.setMinimumWidth(350)
        left_panel.setMaximumWidth(420)
        left_layout = QVBoxLayout(left_panel)

        # Source & Serial Config Group
        conn_group = QGroupBox("1. Data Source & UART Configuration")
        conn_form = QFormLayout(conn_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            [
                "Mock Radar (Simulated Binary TLV)",
                "Hardware / PTY Dual-UART",
            ]
        )
        self.mode_combo.setCurrentIndex(0 if initial_mock else 1)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        conn_form.addRow("Source Mode:", self.mode_combo)

        self.cli_port_edit = QLineEdit(cli_port)
        conn_form.addRow("User UART (115200):", self.cli_port_edit)

        self.data_port_edit = QLineEdit(data_port)
        conn_form.addRow("Data UART (921600):", self.data_port_edit)

        self.pty_preset_btn = QPushButton("Fill PTY Virtual Ports (/tmp/ttyMockRadar*)")
        self.pty_preset_btn.clicked.connect(self._fill_pty_ports)
        conn_form.addRow("", self.pty_preset_btn)

        self.frag_check = QCheckBox("Simulate UART Half-Packet Fragmentation")
        self.frag_check.setChecked(True)
        conn_form.addRow("", self.frag_check)

        left_layout.addWidget(conn_group)

        # Profile .cfg Group
        cfg_group = QGroupBox("2. Radar Profile (.cfg)")
        cfg_vbox = QVBoxLayout(cfg_group)
        cfg_row = QHBoxLayout()
        self.cfg_path_edit = QLineEdit(cfg_path)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_cfg)
        cfg_row.addWidget(self.cfg_path_edit)
        cfg_row.addWidget(browse_btn)
        cfg_vbox.addLayout(cfg_row)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Send .cfg && Start Radar")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #1f7a4c; color: white; font-weight: bold; padding: 6px; }"
        )
        self.start_btn.clicked.connect(self.start_streaming)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #9e2a2b; color: white; font-weight: bold; padding: 6px; }"
        )
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_streaming)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        cfg_vbox.addLayout(btn_row)
        left_layout.addWidget(cfg_group)

        # Telemetry Group
        telem_group = QGroupBox("3. Real-Time Frame & DSP Telemetry")
        telem_form = QFormLayout(telem_group)
        self.lbl_frame = QLabel("0")
        self.lbl_objs = QLabel("0")
        self.lbl_fps = QLabel("0.0 Hz")
        self.lbl_pkt_len = QLabel("0 B")
        self.lbl_dsp_time = QLabel("0 us")
        self.lbl_uart_time = QLabel("0 us")
        self.lbl_margin = QLabel("0 us")
        self.lbl_cpu = QLabel("0% / 0%")
        self.lbl_temp = QLabel("-")
        self.lbl_sync = QLabel("0 / 0")

        telem_form.addRow("Frame Number:", self.lbl_frame)
        telem_form.addRow("Detected Objects:", self.lbl_objs)
        telem_form.addRow("Stream Rate:", self.lbl_fps)
        telem_form.addRow("Packet Size (32B aligned):", self.lbl_pkt_len)
        telem_form.addRow("Inter-frame DSP Time:", self.lbl_dsp_time)
        telem_form.addRow("UART Transmit Time:", self.lbl_uart_time)
        telem_form.addRow("Inter-frame Margin:", self.lbl_margin)
        telem_form.addRow("CPU Load (Act / Inter):", self.lbl_cpu)
        telem_form.addRow("RF Temp (RX0 / TX0):", self.lbl_temp)
        telem_form.addRow("Sync Losses / TLV Err:", self.lbl_sync)
        left_layout.addWidget(telem_group)

        # CLI Console Log
        cli_group = QGroupBox("4. CLI Console & Protocol Log")
        cli_vbox = QVBoxLayout(cli_group)
        self.cli_console = QPlainTextEdit()
        self.cli_console.setReadOnly(True)
        self.cli_console.setFont(QFont("Monospace", 9))
        cli_vbox.addWidget(self.cli_console)
        left_layout.addWidget(cli_group, stretch=1)

        splitter.addWidget(left_panel)

        # ---------------- Right 2x2 Plot Grid ----------------
        right_panel = QWidget()
        grid = QGridLayout(right_panel)

        # Plot 1 (Top-Left): 2D X-Y Top-Down Point Cloud
        self.xy_plot = pg.PlotWidget(title="2D Point Cloud (Top-Down X-Y View)")
        self.xy_plot.setLabel("bottom", "Azimuth X", units="m")
        self.xy_plot.setLabel("left", "Range Y", units="m")
        self.xy_plot.setXRange(-4.0, 4.0)
        self.xy_plot.setYRange(0.0, 8.0)
        self.xy_plot.showGrid(x=True, y=True, alpha=0.25)
        self._draw_radar_polar_overlay(self.xy_plot)
        self.xy_scatter = pg.ScatterPlotItem(size=11, pen=pg.mkPen(None))
        self.xy_plot.addItem(self.xy_scatter)
        grid.addWidget(self.xy_plot, 0, 0)

        # Plot 2 (Top-Right): Range-Velocity / Elevation 2D View (robust on all X11/Wayland drivers)
        self.rv_plot = pg.PlotWidget(
            title="3D Target Height (Z) & Radial Velocity vs Range"
        )
        self.rv_plot.setLabel("bottom", "Range R", units="m")
        self.rv_plot.setLabel("left", "Radial Velocity V_r (m/s) / Height Z (m)")
        self.rv_plot.setXRange(0.0, 8.0)
        self.rv_plot.setYRange(-3.5, 3.5)
        self.rv_plot.showGrid(x=True, y=True, alpha=0.25)
        self.rv_plot.addLegend(offset=(10, 10))
        self.rv_vel_scatter = pg.ScatterPlotItem(
            size=11, brush=pg.mkBrush("#ffbe0b"), name="Velocity V_r (m/s)"
        )
        self.rv_z_scatter = pg.ScatterPlotItem(
            size=9, symbol="t", brush=pg.mkBrush("#00f5d4"), name="Elevation Z (m)"
        )
        self.rv_plot.addItem(self.rv_vel_scatter)
        self.rv_plot.addItem(self.rv_z_scatter)
        grid.addWidget(self.rv_plot, 0, 1)

        # Plot 3 (Bottom-Left): 1D Range Profile + Noise Profile + Target Detections
        self.rp_plot = pg.PlotWidget(title="1D Range Profile & Noise Floor (Zero-Doppler)")
        self.rp_plot.setLabel("bottom", "Range", units="m")
        self.rp_plot.setLabel("left", "Relative Power", units="dB")
        self.rp_plot.setXRange(0.0, 9.0)
        self.rp_plot.setYRange(20.0, 90.0)
        self.rp_plot.showGrid(x=True, y=True, alpha=0.25)
        self.rp_plot.addLegend(offset=(10, 10))
        self.rp_curve = self.rp_plot.plot(
            pen=pg.mkPen("#38bdf8", width=2), name="Range Profile (TLV 2)"
        )
        self.np_curve = self.rp_plot.plot(
            pen=pg.mkPen("#f87171", width=1.5, style=Qt.PenStyle.DashLine),
            name="Noise Floor (TLV 3)",
        )
        self.rp_det_scatter = pg.ScatterPlotItem(
            size=10, symbol="o", brush=pg.mkBrush("#facc15"), name="CFAR Detections"
        )
        self.rp_plot.addItem(self.rp_det_scatter)
        grid.addWidget(self.rp_plot, 1, 0)

        # Plot 4 (Bottom-Right): 2D Range-Azimuth Static Heatmap
        self.hm_plot = pg.PlotWidget(title="2D Range-Azimuth Static Heatmap (TLV 4 FFT)")
        self.hm_plot.setLabel("bottom", "Azimuth Angle", units="deg")
        self.hm_plot.setLabel("left", "Range", units="m")
        self.hm_img = pg.ImageItem()
        self.hm_plot.addItem(self.hm_img)
        cmap = pg.colormap.get("viridis")
        self.hm_img.setColorMap(cmap)
        grid.addWidget(self.hm_plot, 1, 1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)

        self._on_mode_changed()

    def _draw_radar_polar_overlay(self, plot_widget: pg.PlotWidget) -> None:
        """Draw radial range arcs (2m, 4m, 6m, 8m) and +-60 deg FoV lines."""
        theta = np.linspace(-math.pi / 3, math.pi / 3, 90)
        for r in (2.0, 4.0, 6.0, 8.0):
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            arc = pg.PlotCurveItem(
                x, y, pen=pg.mkPen(color=(120, 140, 160, 90), style=Qt.PenStyle.DotLine)
            )
            plot_widget.addItem(arc)
        for ang_deg in (-60, -30, 0, 30, 60):
            rad = math.radians(ang_deg)
            line = pg.PlotCurveItem(
                [0.0, 8.5 * math.sin(rad)],
                [0.0, 8.5 * math.cos(rad)],
                pen=pg.mkPen(color=(120, 140, 160, 70), style=Qt.PenStyle.DotLine),
            )
            plot_widget.addItem(line)

    def _on_mode_changed(self) -> None:
        is_mock = self.mode_combo.currentIndex() == 0
        self.cli_port_edit.setEnabled(not is_mock)
        self.data_port_edit.setEnabled(not is_mock)
        self.pty_preset_btn.setEnabled(not is_mock)
        self.frag_check.setEnabled(is_mock)

    def _fill_pty_ports(self) -> None:
        self.cli_port_edit.setText("/tmp/ttyMockRadarCLI")
        self.data_port_edit.setText("/tmp/ttyMockRadarDATA")

    def _browse_cfg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select TI mmWave .cfg File", str(Path(DEFAULT_CFG_PATH).parent), "CFG Files (*.cfg)"
        )
        if path:
            self.cfg_path_edit.setText(path)

    def start_streaming(self) -> None:
        self.stop_streaming()

        is_mock = self.mode_combo.currentIndex() == 0
        cfg_path = self.cfg_path_edit.text().strip()

        self.worker = RadarSerialWorker(
            config=self.config,
            mock_mode=is_mock,
            cli_port=self.cli_port_edit.text().strip(),
            data_port=self.data_port_edit.text().strip(),
            fragment_mock_packets=self.frag_check.isChecked(),
        )
        self.worker.frame_received.connect(self._on_frame)
        self.worker.cli_log.connect(self.cli_console.appendPlainText)
        self.worker.status_changed.connect(self.statusBar().showMessage)
        self.worker.stats_updated.connect(self._on_worker_stats)

        if os.path.exists(cfg_path):
            ok = self.worker.send_cfg_file(cfg_path)
            self.config = self.worker.config
            if not ok and not is_mock:
                return
        else:
            self.cli_console.appendPlainText(
                f"[WARN] Config file {cfg_path} not found; using default parameters."
            )

        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_streaming(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _on_worker_stats(
        self, frames_ok: int, sync_losses: int, tlv_errs: int, fps: float
    ) -> None:
        self.lbl_fps.setText(f"{fps:.1f} Hz ({frames_ok} pkts)")
        self.lbl_sync.setText(f"{sync_losses} / {tlv_errs}")

    def _on_frame(self, frame: RadarFrame) -> None:
        self.lbl_frame.setText(str(frame.frame_number))
        self.lbl_objs.setText(str(frame.num_detected_obj))
        self.lbl_pkt_len.setText(f"{frame.total_packet_len} B ({frame.num_tlvs} TLVs)")
        self.lbl_dsp_time.setText(f"{frame.inter_frame_proc_time_us} us")
        self.lbl_uart_time.setText(f"{frame.transmit_output_time_us} us")
        self.lbl_margin.setText(f"{frame.inter_frame_margin_us} us")
        self.lbl_cpu.setText(
            f"{frame.active_frame_cpu_load}% / {frame.inter_frame_cpu_load}%"
        )
        if frame.temps_c:
            self.lbl_temp.setText(
                f"{frame.temps_c.get('RX0', 0)}°C / {frame.temps_c.get('TX0', 0)}°C"
            )

        # 1. Update Point Cloud Plots
        pts = frame.points
        if len(pts) > 0:
            x = pts[:, 0]
            y = pts[:, 1]
            z = pts[:, 2]
            v = pts[:, 3]
            r = np.sqrt(x**2 + y**2 + z**2)

            # Color points by radial velocity: blue (approaching), green (static), red (receding)
            brushes = []
            for vel in v:
                if vel < -0.15:
                    brushes.append(pg.mkBrush(56, 189, 248, 220))
                elif vel > 0.15:
                    brushes.append(pg.mkBrush(248, 113, 113, 220))
                else:
                    brushes.append(pg.mkBrush(74, 222, 128, 220))

            self.xy_scatter.setData(x=x, y=y, brush=brushes)
            self.rv_vel_scatter.setData(x=r, y=v)
            self.rv_z_scatter.setData(x=r, y=z)
        else:
            self.xy_scatter.clear()
            self.rv_vel_scatter.clear()
            self.rv_z_scatter.clear()

        # 2. Update 1D Range Profile & Noise Profile
        if frame.range_profile_db is not None:
            num_bins = len(frame.range_profile_db)
            r_axis = np.arange(num_bins, dtype=np.float32) * self.config.range_bin_step_m
            self.rp_curve.setData(r_axis, frame.range_profile_db)

            if frame.noise_profile_db is not None and len(frame.noise_profile_db) == num_bins:
                self.np_curve.setData(r_axis, frame.noise_profile_db)

            # Overlay detected objects on the 1D range profile curve
            if len(pts) > 0:
                obj_r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2 + pts[:, 2] ** 2)
                bin_indices = np.clip(
                    np.round(obj_r / self.config.range_bin_step_m).astype(int),
                    0,
                    num_bins - 1,
                )
                self.rp_det_scatter.setData(x=obj_r, y=frame.range_profile_db[bin_indices])
            else:
                self.rp_det_scatter.clear()

        # 3. Update 2D Range-Azimuth Static Heatmap
        if frame.azimuth_heatmap_db is not None:
            hm = frame.azimuth_heatmap_db  # shape: (num_range_bins, num_angle_bins)
            num_r, num_a = hm.shape
            max_r = num_r * self.config.range_bin_step_m
            # Transpose to (num_angle_bins, num_range_bins) so X=angle (-90..+90), Y=range (0..max_r)
            self.hm_img.setImage(hm.T, autoLevels=True)
            self.hm_img.setRect(pg.QtCore.QRectF(-90.0, 0.0, 180.0, max_r))

    def closeEvent(self, event: object) -> None:
        self.stop_streaming()
        super().closeEvent(event)  # type: ignore[arg-type]


def run_headless_self_test(num_frames: int = 25) -> int:
    """
    Run an automated end-to-end self-test of the Mock Radar Generator + half-packet
    fragmenter + TLVStreamParser without requiring an X11 display.
    """
    from host.mock_generator import MockRadarGenerator
    from host.tlv_parser import TLVStreamParser

    cfg = RadarConfig.from_cfg_file(DEFAULT_CFG_PATH)
    gen = MockRadarGenerator(config=cfg, enable_azimuth_heatmap=True)
    parser = TLVStreamParser(config=cfg)

    decoded = []
    for _ in range(num_frames):
        pkt = gen.next_frame_bytes()
        for chunk in gen.chunk_packet(pkt):
            decoded.extend(parser.feed(chunk))

    print("======================================================================")
    print("  IWR6843 Host TLV Pipeline Self-Test (Fragmented UART Stream)")
    print("======================================================================")
    print(f"  Profile Config      : {DEFAULT_CFG_PATH}")
    print(f"  Range Bins / Res    : {cfg.num_range_bins} bins, ΔR = {cfg.range_resolution_m*100:.2f} cm")
    print(f"  Frames Transmitted  : {num_frames}")
    print(f"  Frames Decoded      : {len(decoded)} (Sync losses: {parser.sync_losses}, TLV errors: {parser.tlv_errors})")
    if decoded:
        last = decoded[-1]
        print(f"  Last Frame #        : {last.frame_number} ({last.total_packet_len} bytes, {last.num_tlvs} TLVs)")
        print(f"  Detected Objects    : {last.num_detected_obj}, Points shape: {last.points.shape}")
        print(f"  Side Info SNR/Noise : shape {last.side_info.shape}, mean SNR = {last.side_info[:,0].mean():.1f} dB")
        if last.range_profile_db is not None:
            print(
                f"  1D Range Profile    : {len(last.range_profile_db)} bins, "
                f"peak = {last.range_profile_db.max():.1f} dB @ "
                f"{last.range_profile_db.argmax() * cfg.range_bin_step_m:.2f} m"
            )
        if last.azimuth_heatmap_db is not None:
            print(f"  2D Azimuth Heatmap  : shape {last.azimuth_heatmap_db.shape} (Range x Angle FFT)")
    print("======================================================================")
    return 0 if (len(decoded) == num_frames and parser.sync_losses == 0 and parser.tlv_errors == 0) else 1


def _default_serial_ports() -> tuple[str, str]:
    if os.path.exists("/dev/ttyACM0"):
        return "/dev/ttyACM0", "/dev/ttyACM1"
    return "/dev/ttyUSB0", "/dev/ttyUSB1"


def main() -> int:
    default_cli, default_data = _default_serial_ports()
    parser = argparse.ArgumentParser(description="IWR6843 OOB Visualizer & Mock Host")
    parser.add_argument("--mock", action="store_true", default=True, help="Start in Mock Radar mode (default)")
    parser.add_argument("--hw", action="store_true", help="Start in Hardware / PTY Dual-UART mode")
    parser.add_argument("--cli-port", default=default_cli, help=f"User UART device path (115200, default: {default_cli})")
    parser.add_argument("--data-port", default=default_data, help=f"Data UART device path (921600, default: {default_data})")
    parser.add_argument("--cfg", default=DEFAULT_CFG_PATH, help="Path to radar .cfg file")
    parser.add_argument("--auto-start", action="store_true", help="Immediately start streaming on launch")
    parser.add_argument(
        "--headless-test",
        type=int,
        default=0,
        metavar="N",
        help="Run N frames of fragmented binary TLV simulation & parser verification in CLI mode",
    )
    args = parser.parse_args()

    if args.headless_test > 0:
        return run_headless_self_test(args.headless_test)

    app = QApplication(sys.argv)
    win = RadarVisualizerWindow(
        initial_mock=not args.hw,
        cli_port=args.cli_port,
        data_port=args.data_port,
        cfg_path=args.cfg,
        auto_start=args.auto_start,
    )
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
