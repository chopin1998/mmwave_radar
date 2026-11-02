#!/usr/bin/env python3
"""
IWR6843 mmWave Radar Stage-1 Visualizer & Host Controller (PySide6 + PyQtGraph + OpenGL 3D).

Features:
  - Dual-mode operation:
      1. Built-in Mock Radar Simulator (offline test with fragmented binary TLV stream)
      2. Hardware / PTY Dual-UART (115200 CLI .cfg sender + 921600 Data TLV receiver)
  - Real-time visualization (2x2 Grid):
      - Top-Left     : 2D X-Y Bird's-Eye Point Cloud (with radial range rings, 1m reference ring, 10-frame trail & centroid)
      - Top-Right    : True 3D Interactive OpenGL Point Cloud (X-Y-Z space with height stems, 3D trail, 3D target bounding box & camera reset)
                       + switchable 2D Range-Velocity / Height view (locked Y-axis)
      - Bottom-Left  : 1D Range Profile & Noise Floor Envelope (dB vs Range in meters)
      - Bottom-Right : 2D Range-Azimuth Heatmap (2D spatial FFT beamforming on TLV Type 4 with Complex MTI Clutter Suppression)
  - Real-time DSP & UART telemetry (Inter-frame time, UART TX time, Margin, CPU Load, RF Temp)
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QVector3D
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
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from host.serial_worker import RadarSerialWorker
from host.tlv_parser import RadarConfig, RadarFrame, compute_azimuth_heatmap_db
from host.tracker import MultiTargetTracker, TrackedTarget

try:
    import pyqtgraph.opengl as gl

    HAS_OPENGL = True
except Exception:
    HAS_OPENGL = False

PROFILES_DIR = Path(__file__).resolve().parent.parent / "firmware" / "xwr68xx_mmw" / "profiles"
DEFAULT_CFG_PATH = str(PROFILES_DIR / "profile_indoor_heatmap.cfg")


class RadarVisualizerWindow(QMainWindow):
    """Main PySide6 + PyQtGraph + OpenGL 3D GUI window for IWR6843 OOB Demo."""

    def __init__(
        self,
        initial_mock: bool = True,
        cli_port: str = "/dev/ttyUSB0",
        data_port: str = "/dev/ttyUSB1",
        cfg_path: str = DEFAULT_CFG_PATH,
        auto_start: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle("TI IWR6843 mmWave Radar Visualizer (OOB TLV + 3D OpenGL + Mock Simulator)")
        self.resize(1540, 950)

        pg.setConfigOptions(antialias=True, background="#11161d", foreground="#d8e2ec")

        self.config = (
            RadarConfig.from_cfg_file(cfg_path)
            if os.path.exists(cfg_path)
            else RadarConfig()
        )
        self.worker: Optional[RadarSerialWorker] = None

        # Multi-frame point cloud trail history: stores (N, 4) arrays [x, y, z, v]
        self._point_history: Deque[np.ndarray] = deque(maxlen=10)
        # Multi-Target DBSCAN + Kalman Tracker (defaults to 1 Person Strict Anti-Multipath, switchable up to 5)
        self.tracker = MultiTargetTracker(max_targets=1)
        self._track_labels_2d: List[pg.TextItem] = []
        # Complex EMA background estimate for TLV Type 4 Azimuth Heatmap MTI subtraction
        self._hm_bg_ema: Optional[np.ndarray] = None
        # Polar overlay items on xy_plot so we can redraw when range scale changes
        self._polar_items: List[pg.GraphicsObject] = []

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
        left_panel.setMinimumWidth(365)
        left_panel.setMaximumWidth(435)
        left_layout = QVBoxLayout(left_panel)

        # Source & Serial Config Group
        conn_group = QGroupBox("1. Data Source && UART Configuration")
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

        # Profile .cfg & Enhancement Group
        cfg_group = QGroupBox("2. Radar Profile (.cfg) && Clutter Filter")
        cfg_vbox = QVBoxLayout(cfg_group)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.cfg_preset_combo = QComboBox()
        self.cfg_preset_combo.addItem(
            "Indoor Human + Heatmap (5 Hz, High-Vel-Res)",
            str(PROFILES_DIR / "profile_indoor_heatmap.cfg"),
        )
        self.cfg_preset_combo.addItem(
            "Indoor Moving Human Only (10 Hz, DSP Clutter-Off)",
            str(PROFILES_DIR / "profile_indoor_human.cfg"),
        )
        self.cfg_preset_combo.addItem(
            "Standard 3D + Heatmap (5 Hz, Full 9m)",
            str(PROFILES_DIR / "profile_3d_heatmap.cfg"),
        )
        self.cfg_preset_combo.addItem(
            "Standard 3D Point Cloud (10 Hz, Full 9m)",
            str(PROFILES_DIR / "profile_3d.cfg"),
        )
        self.cfg_preset_combo.addItem(
            "Standard 2D Point Cloud (10 Hz)",
            str(PROFILES_DIR / "profile_2d.cfg"),
        )
        self.cfg_preset_combo.currentIndexChanged.connect(self._on_cfg_preset_changed)
        preset_row.addWidget(self.cfg_preset_combo, 1)
        cfg_vbox.addLayout(preset_row)

        cfg_row = QHBoxLayout()
        self.cfg_path_edit = QLineEdit(cfg_path)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_cfg)
        cfg_row.addWidget(self.cfg_path_edit)
        cfg_row.addWidget(browse_btn)
        cfg_vbox.addLayout(cfg_row)

        # View Range & Clutter / Trail Controls
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("View Scale:"))
        self.range_scale_combo = QComboBox()
        self.range_scale_combo.addItem("Indoor Near-Field (0 - 3.5 m, 1m Ring)", 3.5)
        self.range_scale_combo.addItem("Medium Room (0 - 5.0 m)", 5.0)
        self.range_scale_combo.addItem("Full Range (0 - 8.5 m)", 8.5)
        self.range_scale_combo.currentIndexChanged.connect(self._on_range_scale_changed)
        view_row.addWidget(self.range_scale_combo, 1)
        cfg_vbox.addLayout(view_row)

        trk_row = QHBoxLayout()
        trk_row.addWidget(QLabel("Track Mode:"))
        self.max_targets_combo = QComboBox()
        self.max_targets_combo.addItem("1 Person (Strict Anti-Multipath Ghost)", 1)
        self.max_targets_combo.addItem("Multi-Target (Up to 2 People)", 2)
        self.max_targets_combo.addItem("Multi-Target (Up to 5 People)", 5)
        self.max_targets_combo.currentIndexChanged.connect(self._on_max_targets_changed)
        trk_row.addWidget(self.max_targets_combo, 1)
        cfg_vbox.addLayout(trk_row)

        self.mti_filter_check = QCheckBox(
            "Suppress Static Desk/Wall Clutter (MTI Velocity Gate + Dynamic Heatmap)"
        )
        self.mti_filter_check.setChecked(True)
        self.mti_filter_check.toggled.connect(self._reset_visual_buffers)
        cfg_vbox.addWidget(self.mti_filter_check)

        self.trail_check = QCheckBox(
            "10-Frame Point Cloud Persistence (Trail + Target Centroid Crosshair)"
        )
        self.trail_check.setChecked(True)
        self.trail_check.toggled.connect(self._reset_visual_buffers)
        cfg_vbox.addWidget(self.trail_check)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Send .cfg && Start Radar")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #1f7a4c; color: white; font-weight: bold; padding: 6px; }"
        )
        self.start_btn.clicked.connect(self.start_streaming)

        self.stop_btn = QPushButton("Stop (RF Off)")
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #9e2a2b; color: white; font-weight: bold; padding: 6px; }"
        )
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_streaming)

        self.deep_sleep_btn = QPushButton("Deep Sleep")
        self.deep_sleep_btn.setToolTip(
            "Send 'idlePowerDown -1 1 0 1 1 1 0 0 0 1000000' to shut down RF, APLL, BSS, DSP & drop MSS to 40MHz (lowest power; requires NRST to wake)"
        )
        self.deep_sleep_btn.setStyleSheet(
            "QPushButton { background-color: #3b4252; color: #e5e9f0; font-weight: bold; padding: 6px; }"
        )
        self.deep_sleep_btn.clicked.connect(self.deep_power_down_hardware)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.deep_sleep_btn)
        cfg_vbox.addLayout(btn_row)

        self.deep_sleep_on_stop_check = QCheckBox(
            "Lowest Power on Stop (idlePowerDown; requires NRST to wake)"
        )
        self.deep_sleep_on_stop_check.setChecked(False)
        cfg_vbox.addWidget(self.deep_sleep_on_stop_check)
        left_layout.addWidget(cfg_group)

        # Telemetry Group
        telem_group = QGroupBox("3. Real-Time Frame && Target Telemetry")
        telem_form = QFormLayout(telem_group)
        self.lbl_frame = QLabel("0")
        self.lbl_objs = QLabel("0")
        self.lbl_target = QLabel("-")
        self.lbl_target.setStyleSheet("color: #facc15; font-weight: bold;")
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
        telem_form.addRow("Primary Target:", self.lbl_target)
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
        cli_group = QGroupBox("4. CLI Console && Protocol Log")
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
        self.xy_plot.showGrid(x=True, y=True, alpha=0.25)
        self.xy_plot.addLegend(offset=(10, 10))
        self.xy_trail_scatter = pg.ScatterPlotItem(
            size=9, pen=pg.mkPen(None), name="10-Frame Trail"
        )
        self.xy_scatter = pg.ScatterPlotItem(
            size=14, pen=pg.mkPen("#ffffff", width=1.0), name="Current Frame"
        )
        self.xy_centroid_scatter = pg.ScatterPlotItem(
            size=22,
            symbol="+",
            pen=pg.mkPen("#facc15", width=2.5),
            brush=pg.mkBrush("#facc15"),
            name="Target Centroid",
        )
        self.xy_plot.addItem(self.xy_trail_scatter)
        self.xy_plot.addItem(self.xy_scatter)
        self.xy_plot.addItem(self.xy_centroid_scatter)
        grid.addWidget(self.xy_plot, 0, 0)

        # Plot 2 (Top-Right): True 3D OpenGL Point Cloud + Switchable 2D R-V/R-Z View
        tr_container = QWidget()
        tr_vbox = QVBoxLayout(tr_container)
        tr_vbox.setContentsMargins(0, 0, 0, 0)
        tr_vbox.setSpacing(4)

        tr_header = QHBoxLayout()
        tr_title = QLabel("<b>3D Spatial Point Cloud (X-Y-Z)</b>")
        tr_title.setStyleSheet("color: #d8e2ec;")
        tr_header.addWidget(tr_title)
        tr_header.addStretch(1)

        self.tr_mode_combo = QComboBox()
        if HAS_OPENGL:
            self.tr_mode_combo.addItem("3D OpenGL Space (X-Y-Z + Stems)")
        self.tr_mode_combo.addItem("2D Range vs Velocity & Height (R-V / R-Z)")
        self.tr_mode_combo.currentIndexChanged.connect(self._on_tr_mode_changed)
        tr_header.addWidget(self.tr_mode_combo)

        self.reset_3d_btn = QPushButton("Reset 3D View")
        self.reset_3d_btn.setFixedWidth(105)
        self.reset_3d_btn.clicked.connect(self._reset_3d_camera)
        tr_header.addWidget(self.reset_3d_btn)
        tr_vbox.addLayout(tr_header)

        self.tr_stack = QStackedWidget()
        tr_vbox.addWidget(self.tr_stack, 1)

        # Page 0: True 3D OpenGL View (GLViewWidget)
        self.gl_view: Optional[gl.GLViewWidget] = None
        if HAS_OPENGL:
            self.gl_view = gl.GLViewWidget()
            self.gl_view.setBackgroundColor("#11161d")
            self._init_3d_scene()
            self.tr_stack.addWidget(self.gl_view)

        # Page 1: 2D Range-Velocity / Elevation View (with locked Y-axis)
        self.rv_plot = pg.PlotWidget(
            title="Target Height (Z) & Radial Velocity vs Range (Locked Y-Axis)"
        )
        self.rv_plot.setLabel("bottom", "Range R", units="m")
        self.rv_plot.setLabel("left", "Radial Velocity V_r (m/s) / Height Z (m)")
        self.rv_plot.setYRange(-3.0, 3.0, padding=0.0)
        self.rv_plot.enableAutoRange(axis="x", enable=False)
        self.rv_plot.enableAutoRange(axis="y", enable=False)
        self.rv_plot.setMouseEnabled(x=True, y=False)
        self.rv_plot.showGrid(x=True, y=True, alpha=0.25)
        self.rv_plot.addLegend(offset=(10, 10))
        self.rv_vel_scatter = pg.ScatterPlotItem(
            size=12, brush=pg.mkBrush("#ffbe0b"), name="Velocity V_r (m/s)"
        )
        self.rv_z_scatter = pg.ScatterPlotItem(
            size=10, symbol="t", brush=pg.mkBrush("#00f5d4"), name="Elevation Z (m)"
        )
        self.rv_plot.addItem(self.rv_vel_scatter)
        self.rv_plot.addItem(self.rv_z_scatter)
        self.tr_stack.addWidget(self.rv_plot)

        grid.addWidget(tr_container, 0, 1)

        # Plot 3 (Bottom-Left): 1D Range Profile + Noise Profile + Target Detections
        self.rp_plot = pg.PlotWidget(title="1D Range Profile & Noise Floor (Zero-Doppler)")
        self.rp_plot.setLabel("bottom", "Range", units="m")
        self.rp_plot.setLabel("left", "Relative Power", units="dB")
        self.rp_plot.setYRange(20.0, 95.0, padding=0.0)
        self.rp_plot.enableAutoRange(axis="x", enable=False)
        self.rp_plot.enableAutoRange(axis="y", enable=False)
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
            size=11, symbol="o", brush=pg.mkBrush("#facc15"), name="Active Detections"
        )
        self.rp_plot.addItem(self.rp_det_scatter)
        grid.addWidget(self.rp_plot, 1, 0)

        # Plot 4 (Bottom-Right): 2D Range-Azimuth Heatmap
        self.hm_plot = pg.PlotWidget(
            title="2D Range-Azimuth Heatmap (TLV 4 FFT / Dynamic MTI)"
        )
        self.hm_plot.setLabel("bottom", "Azimuth Angle", units="deg")
        self.hm_plot.setLabel("left", "Range", units="m")
        self.hm_plot.enableAutoRange(axis="x", enable=False)
        self.hm_plot.enableAutoRange(axis="y", enable=False)
        self.hm_plot.setXRange(-75.0, 75.0, padding=0.0)
        self.hm_img = pg.ImageItem()
        self.hm_plot.addItem(self.hm_img)
        cmap = pg.colormap.get("viridis")
        self.hm_img.setColorMap(cmap)
        grid.addWidget(self.hm_plot, 1, 1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)

        self._on_mode_changed()
        self._on_range_scale_changed()

    def _init_3d_scene(self) -> None:
        """Initialize OpenGL 3D coordinate system, ground grid, FoV cone, 1m arc, and scatter items."""
        if self.gl_view is None:
            return

        # 1. Ground plane grid (X in [-3, 3], Y in [0, 6], Z = 0)
        self.gl_grid = gl.GLGridItem()
        self.gl_grid.setSize(x=6.0, y=6.0, z=1.0)
        self.gl_grid.setSpacing(x=0.5, y=0.5, z=0.5)
        self.gl_grid.translate(0.0, 3.0, 0.0)
        self.gl_grid.setColor((120, 140, 165, 65))
        self.gl_view.addItem(self.gl_grid)

        # 2. Coordinate Axes at Radar Origin (0,0,0): X=Red, Y(Range)=Green, Z(Height)=Blue
        axis_pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.6, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.6],
            ],
            dtype=np.float32,
        )
        axis_colors = np.array(
            [
                [1.0, 0.35, 0.35, 1.0],
                [1.0, 0.35, 0.35, 1.0],
                [0.30, 0.90, 0.50, 1.0],
                [0.30, 0.90, 0.50, 1.0],
                [0.35, 0.75, 1.0, 1.0],
                [0.35, 0.75, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.gl_axes = gl.GLLinePlotItem(
            pos=axis_pts, color=axis_colors, width=2.5, mode="lines", antialias=True
        )
        self.gl_view.addItem(self.gl_axes)

        # 3. 1.0m & 2.0m Reference Range Arcs in 3D (Z = 0 plane)
        theta = np.linspace(-math.pi / 3, math.pi / 3, 60)
        arc1_pts = np.column_stack(
            [1.0 * np.sin(theta), 1.0 * np.cos(theta), np.zeros_like(theta)]
        ).astype(np.float32)
        self.gl_arc_1m = gl.GLLinePlotItem(
            pos=arc1_pts,
            color=(0.98, 0.80, 0.08, 0.70),
            width=2.0,
            mode="line_strip",
            antialias=True,
        )
        self.gl_view.addItem(self.gl_arc_1m)

        arc2_pts = np.column_stack(
            [2.0 * np.sin(theta), 2.0 * np.cos(theta), np.zeros_like(theta)]
        ).astype(np.float32)
        self.gl_arc_2m = gl.GLLinePlotItem(
            pos=arc2_pts,
            color=(0.47, 0.55, 0.63, 0.45),
            width=1.2,
            mode="line_strip",
            antialias=True,
        )
        self.gl_view.addItem(self.gl_arc_2m)

        # 4. 3D Trail Scatter (past 10 frames)
        self.gl_trail_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32), size=7.0, pxMode=True
        )
        self.gl_view.addItem(self.gl_trail_scatter)

        # 5. Current-Frame 3D Scatter (bright spheres)
        self.gl_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3), dtype=np.float32), size=13.0, pxMode=True
        )
        self.gl_view.addItem(self.gl_scatter)

        # 6. Vertical Height Drop-Lines (Stems from (x,y,z) to (x,y,0))
        self.gl_stems = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=(0.85, 0.90, 0.95, 0.45),
            width=1.4,
            mode="lines",
            antialias=True,
        )
        self.gl_view.addItem(self.gl_stems)

        # 7. 3D Target Cluster Bounding Box / Centroid Crosshair
        self.gl_bbox = gl.GLLinePlotItem(
            pos=np.zeros((0, 3), dtype=np.float32),
            color=(0.98, 0.80, 0.08, 0.90),
            width=2.2,
            mode="lines",
            antialias=True,
        )
        self.gl_view.addItem(self.gl_bbox)

        self._reset_3d_camera()

    def _reset_3d_camera(self) -> None:
        """Reset 3D OpenGL camera to look at the indoor test region around Y = 1.2m ~ 1.8m."""
        if self.gl_view is None:
            return
        max_r = float(self.range_scale_combo.currentData() or 3.5)
        center_y = min(max_r * 0.42, 2.5)
        self.gl_view.opts["center"] = QVector3D(0.0, center_y, 0.1)
        self.gl_view.setCameraPosition(
            distance=max_r * 1.15, elevation=26.0, azimuth=-62.0
        )

    def _on_tr_mode_changed(self, index: int) -> None:
        self.tr_stack.setCurrentIndex(index)
        self.reset_3d_btn.setEnabled(index == 0 and HAS_OPENGL)

    @staticmethod
    def _make_3d_bbox_lines(
        cx: float, cy: float, cz: float, hx: float = 0.28, hy: float = 0.28, hz: float = 0.55
    ) -> np.ndarray:
        """Create 12 wireframe edges of a 3D bounding box + center 3D crosshair around (cx, cy, cz)."""
        x0, x1 = cx - hx, cx + hx
        y0, y1 = cy - hy, cy + hy
        z0, z1 = cz - hz, cz + hz
        c = [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        lines = []
        for i, j in edges:
            lines.append(c[i])
            lines.append(c[j])
        # Add 3D crosshair at centroid
        s = 0.15
        lines.extend([[cx - s, cy, cz], [cx + s, cy, cz]])
        lines.extend([[cx, cy - s, cz], [cx, cy + s, cz]])
        lines.extend([[cx, cy, cz - s], [cx, cy, cz + s]])
        return np.array(lines, dtype=np.float32)

    def _draw_radar_polar_overlay(self, max_range_m: float) -> None:
        """Draw radial range arcs and FoV lines scaled to max_range_m."""
        for item in self._polar_items:
            self.xy_plot.removeItem(item)
        self._polar_items.clear()

        theta = np.linspace(-math.pi / 3, math.pi / 3, 90)
        if max_range_m <= 4.0:
            rings = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
        elif max_range_m <= 5.5:
            rings = (1.0, 2.0, 3.0, 4.0, 5.0)
        else:
            rings = (2.0, 4.0, 6.0, 8.0)

        for r in rings:
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            is_one_meter = abs(r - 1.0) < 1e-3
            pen = (
                pg.mkPen(color=(250, 204, 21, 130), width=1.4, style=Qt.PenStyle.DashLine)
                if is_one_meter
                else pg.mkPen(color=(120, 140, 160, 85), style=Qt.PenStyle.DotLine)
            )
            arc = pg.PlotCurveItem(x, y, pen=pen)
            self.xy_plot.addItem(arc)
            self._polar_items.append(arc)

        for ang_deg in (-60, -30, 0, 30, 60):
            rad = math.radians(ang_deg)
            line = pg.PlotCurveItem(
                [0.0, max_range_m * 1.05 * math.sin(rad)],
                [0.0, max_range_m * 1.05 * math.cos(rad)],
                pen=pg.mkPen(color=(120, 140, 160, 70), style=Qt.PenStyle.DotLine),
            )
            self.xy_plot.addItem(line)
            self._polar_items.append(line)

    def _on_range_scale_changed(self) -> None:
        max_r = float(self.range_scale_combo.currentData() or 3.5)
        half_x = round(max_r * 0.6, 1)
        self.xy_plot.setXRange(-half_x, half_x, padding=0.0)
        self.xy_plot.setYRange(0.0, max_r, padding=0.0)
        self.rv_plot.setXRange(0.0, max_r, padding=0.0)
        self.rv_plot.setYRange(-3.0, 3.0, padding=0.0)
        self.rp_plot.setXRange(0.0, max_r, padding=0.0)
        self.hm_plot.setYRange(0.0, max_r, padding=0.0)
        self._draw_radar_polar_overlay(max_r)
        self._reset_3d_camera()

    def _on_max_targets_changed(self) -> None:
        max_t = int(self.max_targets_combo.currentData() or 1)
        self.tracker.max_targets = max_t

    def _reset_visual_buffers(self) -> None:
        self._point_history.clear()
        self.tracker.reset()
        self._hm_bg_ema = None
        self.xy_trail_scatter.clear()
        self.xy_centroid_scatter.clear()
        for lbl in self._track_labels_2d:
            self.xy_plot.removeItem(lbl)
        self._track_labels_2d.clear()
        if self.gl_view is not None:
            empty = np.zeros((0, 3), dtype=np.float32)
            self.gl_trail_scatter.setData(pos=empty)
            self.gl_scatter.setData(pos=empty)
            self.gl_stems.setData(pos=empty)
            self.gl_bbox.setData(pos=empty)

    def _on_mode_changed(self) -> None:
        is_mock = self.mode_combo.currentIndex() == 0
        self.cli_port_edit.setEnabled(not is_mock)
        self.data_port_edit.setEnabled(not is_mock)
        self.pty_preset_btn.setEnabled(not is_mock)
        self.frag_check.setEnabled(is_mock)

    def _fill_pty_ports(self) -> None:
        self.cli_port_edit.setText("/tmp/ttyMockRadarCLI")
        self.data_port_edit.setText("/tmp/ttyMockRadarDATA")

    def _on_cfg_preset_changed(self, index: int) -> None:
        preset_path = self.cfg_preset_combo.itemData(index)
        if preset_path:
            self.cfg_path_edit.setText(str(preset_path))
            if "indoor" in str(preset_path).lower():
                self.range_scale_combo.setCurrentIndex(0)
            else:
                self.range_scale_combo.setCurrentIndex(2)

    def _browse_cfg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select TI mmWave .cfg File",
            str(Path(DEFAULT_CFG_PATH).parent),
            "CFG Files (*.cfg)",
        )
        if path:
            self.cfg_path_edit.setText(path)

    def start_streaming(self) -> None:
        if self.worker is not None:
            self.worker.stop(deep_power_down=False)
            self.worker = None

        self._reset_visual_buffers()
        is_mock = self.mode_combo.currentIndex() == 0
        cfg_path = self.cfg_path_edit.text().strip()

        if os.path.exists(cfg_path):
            try:
                self.config = RadarConfig.from_cfg_file(cfg_path)
            except Exception:
                pass
        else:
            self.cli_console.appendPlainText(
                f"[WARN] Config file {cfg_path} not found; using default parameters."
            )

        self.worker = RadarSerialWorker(
            config=self.config,
            mock_mode=is_mock,
            cli_port=self.cli_port_edit.text().strip(),
            data_port=self.data_port_edit.text().strip(),
            fragment_mock_packets=self.frag_check.isChecked(),
            cfg_path=cfg_path if os.path.exists(cfg_path) else None,
        )
        self.worker.frame_received.connect(self._on_frame)
        self.worker.cli_log.connect(self.cli_console.appendPlainText)
        self.worker.status_changed.connect(self.statusBar().showMessage)
        self.worker.stats_updated.connect(self._on_worker_stats)
        self.worker.stream_stopped.connect(self._on_worker_stopped)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    def _on_worker_stopped(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def stop_streaming(self) -> None:
        deep = self.deep_sleep_on_stop_check.isChecked()
        if self.worker is not None:
            self.worker.stop(deep_power_down=deep)
            self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def deep_power_down_hardware(self) -> None:
        """Immediately stop sensor and enter lowest-power idlePowerDown state."""
        if self.worker is not None:
            self.worker.stop(deep_power_down=True)
            self.worker = None
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            return

        is_mock = self.mode_combo.currentIndex() == 0
        temp_worker = RadarSerialWorker(
            config=self.config,
            mock_mode=is_mock,
            cli_port=self.cli_port_edit.text().strip(),
            data_port=self.data_port_edit.text().strip(),
        )
        temp_worker.cli_log.connect(self.cli_console.appendPlainText)
        temp_worker.send_cli_command(
            "idlePowerDown -1 1 0 1 1 1 0 0 0 1000000", wait_s=0.8
        )
        self.statusBar().showMessage(
            "Hardware in Deep Power Down (Press NRST on board to wake)"
        )

    def _on_worker_stats(
        self, frames_ok: int, sync_losses: int, tlv_errs: int, fps: float
    ) -> None:
        self.lbl_fps.setText(f"{fps:.1f} Hz ({frames_ok} pkts)")
        self.lbl_sync.setText(f"{sync_losses} / {tlv_errs}")

    @staticmethod
    def _vel_to_rgba(vel: float, alpha: float = 0.95) -> tuple[float, float, float, float]:
        if vel < -0.08:
            return (0.22, 0.74, 0.97, alpha)  # Blue-cyan (approaching)
        if vel > 0.08:
            return (0.97, 0.44, 0.44, alpha)  # Coral-red (receding)
        return (0.29, 0.87, 0.50, alpha)      # Green (near-zero radial vel)

    def _on_frame(self, frame: RadarFrame) -> None:
        self.lbl_frame.setText(str(frame.frame_number))
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

        # 1. Filter, Cluster & Multi-Target Kalman Track Point Cloud
        pts = frame.points
        mti_on = self.mti_filter_check.isChecked()
        trail_on = self.trail_check.isChecked()

        if len(pts) > 0:
            # Guard against NaN/Inf or corrupted float values
            valid_mask = np.isfinite(pts).all(axis=1) & (np.abs(pts[:, 2]) <= 6.0)
            pts = pts[valid_mask]

        dt_s = max(0.05, self.config.frame_periodicity_ms * 1e-3)
        confirmed_tracks, active_pts = self.tracker.update(
            raw_points=pts,
            side_info=frame.side_info,
            dt_s=dt_s,
            mti_enabled=mti_on,
        )

        self.lbl_objs.setText(
            f"{len(active_pts)} active / {len(confirmed_tracks)} trk (raw: {frame.num_detected_obj})"
            if mti_on
            else f"{frame.num_detected_obj} pts / {len(confirmed_tracks)} trk"
        )

        # Update multi-frame trail history (2D & 3D)
        if trail_on:
            if len(active_pts) > 0:
                self._point_history.append(active_pts.copy())
            elif len(self._point_history) > 0:
                self._point_history.append(np.zeros((0, 4), dtype=np.float32))

            trail_x: List[float] = []
            trail_y: List[float] = []
            trail_brushes = []
            trail_3d_pos: List[List[float]] = []
            trail_3d_col: List[tuple[float, float, float, float]] = []

            num_hist = len(self._point_history)
            for idx, hist_pts in enumerate(self._point_history):
                if len(hist_pts) == 0:
                    continue
                frac = (idx + 1) / max(num_hist, 1)
                alpha_255 = int(35 + 125 * frac)
                alpha_f = 0.14 + 0.45 * frac
                for p in hist_pts:
                    px, py, pz, pv = float(p[0]), float(p[1]), float(p[2]), float(p[3])
                    trail_x.append(px)
                    trail_y.append(py)
                    if pv < -0.08:
                        trail_brushes.append(pg.mkBrush(56, 189, 248, alpha_255))
                    elif pv > 0.08:
                        trail_brushes.append(pg.mkBrush(248, 113, 113, alpha_255))
                    else:
                        trail_brushes.append(pg.mkBrush(74, 222, 128, alpha_255))
                    trail_3d_pos.append([px, py, pz])
                    trail_3d_col.append(self._vel_to_rgba(pv, alpha_f))

            if trail_x:
                self.xy_trail_scatter.setData(
                    x=trail_x, y=trail_y, brush=trail_brushes
                )
                if self.gl_view is not None:
                    self.gl_trail_scatter.setData(
                        pos=np.array(trail_3d_pos, dtype=np.float32),
                        color=np.array(trail_3d_col, dtype=np.float32),
                    )
            else:
                self.xy_trail_scatter.clear()
                if self.gl_view is not None:
                    self.gl_trail_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
        else:
            self.xy_trail_scatter.clear()
            if self.gl_view is not None:
                self.gl_trail_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))

        # Render current frame points (2D & 3D)
        if len(active_pts) > 0:
            x = active_pts[:, 0]
            y = active_pts[:, 1]
            z = active_pts[:, 2]
            v = active_pts[:, 3]
            r = np.sqrt(x**2 + y**2 + z**2)

            brushes = []
            gl_colors = []
            stem_pts = []
            for px, py, pz, vel in zip(x, y, z, v):
                if vel < -0.08:
                    brushes.append(pg.mkBrush(56, 189, 248, 245))
                elif vel > 0.08:
                    brushes.append(pg.mkBrush(248, 113, 113, 245))
                else:
                    brushes.append(pg.mkBrush(74, 222, 128, 230))
                gl_colors.append(self._vel_to_rgba(float(vel), 0.98))
                stem_pts.append([float(px), float(py), 0.0])
                stem_pts.append([float(px), float(py), float(pz)])

            self.xy_scatter.setData(x=x, y=y, brush=brushes)
            self.rv_vel_scatter.setData(x=r, y=np.clip(v, -3.0, 3.0))
            self.rv_z_scatter.setData(x=r, y=np.clip(z, -3.0, 3.0))

            if self.gl_view is not None:
                self.gl_scatter.setData(
                    pos=active_pts[:, :3].astype(np.float32),
                    color=np.array(gl_colors, dtype=np.float32),
                )
                self.gl_stems.setData(pos=np.array(stem_pts, dtype=np.float32))
        else:
            self.xy_scatter.clear()
            self.rv_vel_scatter.clear()
            self.rv_z_scatter.clear()
            if self.gl_view is not None:
                empty = np.zeros((0, 3), dtype=np.float32)
                self.gl_scatter.setData(pos=empty)
                self.gl_stems.setData(pos=empty)

        # Render Multi-Target Kalman Tracks (up to 5 targets: T1..T5) in 2D & 3D
        for lbl in self._track_labels_2d:
            self.xy_plot.removeItem(lbl)
        self._track_labels_2d.clear()

        if confirmed_tracks:
            tx_list = [t.x for t in confirmed_tracks]
            ty_list = [t.y for t in confirmed_tracks]
            t_pens = [pg.mkPen(t.color_hex, width=2.8) for t in confirmed_tracks]
            t_brushes = [pg.mkBrush(t.color_hex) for t in confirmed_tracks]
            self.xy_centroid_scatter.setData(
                x=tx_list, y=ty_list, pen=t_pens, brush=t_brushes
            )

            all_bbox_pts = []
            all_bbox_cols = []
            summary_strs = []

            for trk in confirmed_tracks:
                state_tag = "HOLD" if trk.misses > 0 else "LOCK"
                lbl_item = pg.TextItem(
                    text=f"T{trk.track_id} ({trk.range_m:.2f}m)",
                    color=trk.color_hex,
                    anchor=(0.5, 1.25),
                )
                lbl_item.setPos(trk.x, trk.y)
                self.xy_plot.addItem(lbl_item)
                self._track_labels_2d.append(lbl_item)

                summary_strs.append(
                    f"T{trk.track_id}[{state_tag}]: R={trk.range_m:.2f}m (X={trk.x:+.2f}, Y={trk.y:.2f}, Vr={trk.mean_vr:+.2f}m/s)"
                )

                hx, hy, hz = float(trk.bbox_half[0]), float(trk.bbox_half[1]), float(trk.bbox_half[2])
                box_pts = self._make_3d_bbox_lines(trk.x, trk.y, trk.z, hx, hy, hz)
                all_bbox_pts.append(box_pts)
                rgba = trk.color_rgba
                all_bbox_cols.append(np.tile(rgba, (len(box_pts), 1)).astype(np.float32))

            self.lbl_target.setText(" | ".join(summary_strs[:2]))
            if self.gl_view is not None:
                self.gl_bbox.setData(
                    pos=np.vstack(all_bbox_pts),
                    color=np.vstack(all_bbox_cols),
                )
        else:
            self.xy_centroid_scatter.clear()
            self.lbl_target.setText("-")
            if self.gl_view is not None:
                self.gl_bbox.setData(pos=np.zeros((0, 3), dtype=np.float32))

        # 2. Update 1D Range Profile & Noise Profile
        if frame.range_profile_db is not None:
            num_bins = len(frame.range_profile_db)
            r_axis = np.arange(num_bins, dtype=np.float32) * self.config.range_bin_step_m
            self.rp_curve.setData(r_axis, frame.range_profile_db)

            if (
                frame.noise_profile_db is not None
                and len(frame.noise_profile_db) == num_bins
            ):
                self.np_curve.setData(r_axis, frame.noise_profile_db)

            if len(active_pts) > 0:
                obj_r = np.sqrt(
                    active_pts[:, 0] ** 2
                    + active_pts[:, 1] ** 2
                    + active_pts[:, 2] ** 2
                )
                bin_indices = np.clip(
                    np.round(obj_r / self.config.range_bin_step_m).astype(int),
                    0,
                    num_bins - 1,
                )
                self.rp_det_scatter.setData(
                    x=obj_r, y=frame.range_profile_db[bin_indices]
                )
            else:
                self.rp_det_scatter.clear()

        # 3. Update 2D Range-Azimuth Heatmap (with Complex MTI Clutter Suppression when enabled)
        if frame.azimuth_Complex is not None:
            az_cplx = frame.azimuth_Complex.astype(np.complex64)
            if mti_on:
                if (
                    self._hm_bg_ema is None
                    or self._hm_bg_ema.shape != az_cplx.shape
                ):
                    self._hm_bg_ema = az_cplx.copy()
                    dyn_cplx = az_cplx
                else:
                    dyn_cplx = az_cplx - self._hm_bg_ema
                    self._hm_bg_ema = 0.92 * self._hm_bg_ema + 0.08 * az_cplx
                hm = compute_azimuth_heatmap_db(dyn_cplx, num_angle_bins=64)
            else:
                hm = (
                    frame.azimuth_heatmap_db
                    if frame.azimuth_heatmap_db is not None
                    else compute_azimuth_heatmap_db(az_cplx, num_angle_bins=64)
                )

            num_r, _ = hm.shape
            max_r = num_r * self.config.range_bin_step_m
            min_bin = max(1, int(0.30 / max(self.config.range_bin_step_m, 1e-3)))
            hm_display = hm.copy()
            if min_bin < num_r:
                hm_display[:min_bin, :] = np.median(hm_display[min_bin:, :])

            self.hm_img.setImage(hm_display.T, autoLevels=True)
            self.hm_img.setRect(pg.QtCore.QRectF(-90.0, 0.0, 180.0, max_r))
        elif frame.azimuth_heatmap_db is not None:
            hm = frame.azimuth_heatmap_db
            num_r, _ = hm.shape
            max_r = num_r * self.config.range_bin_step_m
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
    print(
        f"  Range Bins / Res    : {cfg.num_range_bins} bins, ΔR = {cfg.range_resolution_m*100:.2f} cm"
    )
    print(f"  Frames Transmitted  : {num_frames}")
    print(
        f"  Frames Decoded      : {len(decoded)} (Sync losses: {parser.sync_losses}, TLV errors: {parser.tlv_errors})"
    )
    if decoded:
        last = decoded[-1]
        print(
            f"  Last Frame #        : {last.frame_number} ({last.total_packet_len} bytes, {last.num_tlvs} TLVs)"
        )
        print(
            f"  Detected Objects    : {last.num_detected_obj}, Points shape: {last.points.shape}"
        )
        print(
            f"  Side Info SNR/Noise : shape {last.side_info.shape}, mean SNR = {last.side_info[:,0].mean():.1f} dB"
        )
        if last.range_profile_db is not None:
            print(
                f"  1D Range Profile    : {len(last.range_profile_db)} bins, "
                f"peak = {last.range_profile_db.max():.1f} dB @ "
                f"{last.range_profile_db.argmax() * cfg.range_bin_step_m:.2f} m"
            )
        if last.azimuth_heatmap_db is not None:
            print(
                f"  2D Azimuth Heatmap  : shape {last.azimuth_heatmap_db.shape} (Range x Angle FFT)"
            )
    print("======================================================================")
    return (
        0
        if (
            len(decoded) == num_frames
            and parser.sync_losses == 0
            and parser.tlv_errors == 0
        )
        else 1
    )


def _default_serial_ports() -> tuple[str, str]:
    if os.path.exists("/dev/ttyACM0"):
        return "/dev/ttyACM0", "/dev/ttyACM1"
    return "/dev/ttyUSB0", "/dev/ttyUSB1"


def main() -> int:
    default_cli, default_data = _default_serial_ports()
    parser = argparse.ArgumentParser(description="IWR6843 OOB Visualizer & Mock Host")
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Start in Mock Radar mode (default)",
    )
    parser.add_argument(
        "--hw",
        action="store_true",
        help="Start in Hardware / PTY Dual-UART mode",
    )
    parser.add_argument(
        "--cli-port",
        default=default_cli,
        help=f"User UART device path (115200, default: {default_cli})",
    )
    parser.add_argument(
        "--data-port",
        default=default_data,
        help=f"Data UART device path (921600, default: {default_data})",
    )
    parser.add_argument(
        "--cfg", default=DEFAULT_CFG_PATH, help="Path to radar .cfg file"
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Immediately start streaming on launch",
    )
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
