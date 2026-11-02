#!/usr/bin/env python3
"""
TI xWR68xx Out-of-Box (OOB) Demo Binary TLV Stream Parser & Radar Config Parser.

Strictly matches `MmwDemo_output_message_header` and TLV payload layouts in:
  firmware/xwr68xx_mmw/include/mmw_output.h
  firmware/xwr68xx_mmw/mss/mss_main.c (MmwDemo_transmitProcessedOutput)
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# 8-byte sync word: {0x0102, 0x0304, 0x0506, 0x0708} in little-endian
MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"
HEADER_FMT = "<8s8I"  # 8-byte magic + 8 uint32 fields = 40 bytes
HEADER_SIZE = 40
TL_FMT = "<II"  # type (uint32), length (uint32) = 8 bytes
TL_SIZE = 8
SEGMENT_LEN = 32  # MMWDEMO_OUTPUT_MSG_SEGMENT_LEN

# Standard OOB TLV Types (from MmwDemo_output_message_type_e)
MMWDEMO_OUTPUT_MSG_DETECTED_POINTS = 1
MMWDEMO_OUTPUT_MSG_RANGE_PROFILE = 2
MMWDEMO_OUTPUT_MSG_NOISE_PROFILE = 3
MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP = 4
MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5
MMWDEMO_OUTPUT_MSG_STATS = 6
MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO = 7
MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP = 8
MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS = 9


@dataclass
class RadarConfig:
    """Radar physical and DSP parameters parsed from a TI mmWave .cfg file."""

    start_freq_ghz: float = 60.0
    freq_slope_mhz_us: float = 166.0
    idle_time_us: float = 7.0
    ramp_end_time_us: float = 24.0
    num_adc_samples: int = 256
    dig_out_sample_rate_ksps: float = 12500.0
    rx_channel_mask: int = 0x0F  # 4 RX
    tx_channel_mask: int = 0x07  # 3 TX (or 0x05 for 2 Azimuth TX)
    chirp_start_idx: int = 0
    chirp_end_idx: int = 1
    num_loops: int = 32
    frame_periodicity_ms: float = 100.0
    gui_detected_objects: int = 1
    gui_log_mag_range: int = 1
    gui_noise_profile: int = 1
    gui_range_azimuth_heat_map: int = 0
    gui_range_doppler_heat_map: int = 0
    gui_stats_info: int = 1

    @property
    def num_range_bins(self) -> int:
        n = 1
        while n < self.num_adc_samples:
            n <<= 1
        return n

    @property
    def num_tx_antennas(self) -> int:
        return self.chirp_end_idx - self.chirp_start_idx + 1

    @property
    def num_rx_antennas(self) -> int:
        return bin(self.rx_channel_mask & 0x0F).count("1")

    @property
    def num_virtual_ant_azim(self) -> int:
        # On IWR6843ISK/ODS, TX1 (bit 0) and TX3 (bit 2) are azimuth TX antennas
        azim_tx = bin(self.tx_channel_mask & 0b101).count("1")
        if azim_tx == 0:
            azim_tx = 1
        return azim_tx * self.num_rx_antennas

    @property
    def num_doppler_bins(self) -> int:
        n = 1
        while n < self.num_loops:
            n <<= 1
        return max(16, n)

    @property
    def range_resolution_m(self) -> float:
        c = 299792458.0
        adc_duration_s = self.num_adc_samples / (self.dig_out_sample_rate_ksps * 1e3)
        bandwidth_hz = self.freq_slope_mhz_us * 1e12 * adc_duration_s
        if bandwidth_hz <= 0:
            return 0.044
        return c / (2.0 * bandwidth_hz)

    @property
    def range_bin_step_m(self) -> float:
        return self.range_resolution_m * (self.num_adc_samples / self.num_range_bins)

    @classmethod
    def from_cfg_text(cls, text: str) -> "RadarConfig":
        cfg = cls()
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            cmd = parts[0]
            try:
                if cmd == "channelCfg" and len(parts) >= 3:
                    cfg.rx_channel_mask = int(parts[1])
                    cfg.tx_channel_mask = int(parts[2])
                elif cmd == "profileCfg" and len(parts) >= 12:
                    cfg.start_freq_ghz = float(parts[2])
                    cfg.idle_time_us = float(parts[3])
                    cfg.ramp_end_time_us = float(parts[5])
                    cfg.freq_slope_mhz_us = float(parts[8])
                    cfg.num_adc_samples = int(parts[10])
                    cfg.dig_out_sample_rate_ksps = float(parts[11])
                elif cmd == "frameCfg" and len(parts) >= 6:
                    cfg.chirp_start_idx = int(parts[1])
                    cfg.chirp_end_idx = int(parts[2])
                    cfg.num_loops = int(parts[3])
                    cfg.frame_periodicity_ms = float(parts[5])
                elif cmd == "guiMonitor" and len(parts) >= 8:
                    cfg.gui_detected_objects = int(parts[2])
                    cfg.gui_log_mag_range = int(parts[3])
                    cfg.gui_noise_profile = int(parts[4])
                    cfg.gui_range_azimuth_heat_map = int(parts[5])
                    cfg.gui_range_doppler_heat_map = int(parts[6])
                    cfg.gui_stats_info = int(parts[7])
            except (ValueError, IndexError):
                continue
        return cfg

    @classmethod
    def from_cfg_file(cls, path: str) -> "RadarConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_cfg_text(f.read())


@dataclass
class RadarFrame:
    """Parsed single-frame data from IWR6843 OOB TLV packet."""

    version: int = 0
    total_packet_len: int = 0
    platform: int = 0xA6843
    frame_number: int = 0
    time_cpu_cycles: int = 0
    num_detected_obj: int = 0
    num_tlvs: int = 0
    subframe_number: int = 0

    # Point cloud: shape (N, 4) -> [x, y, z, velocity] in meters and m/s
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=np.float32))
    # Side info: shape (N, 2) -> [snr_db, noise_db] in dB (converted from 0.1 dB int16)
    side_info: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))

    # 1D profiles in dB (converted from Q9 log2 magnitude)
    range_profile_db: Optional[np.ndarray] = None
    noise_profile_db: Optional[np.ndarray] = None

    # Azimuth static heatmap raw complex data: shape (num_range_bins, num_virtual_ant_azim) complex64
    azimuth_Complex: Optional[np.ndarray] = None
    # Computed Range-Azimuth power matrix (num_range_bins, num_angle_bins) in dB
    azimuth_heatmap_db: Optional[np.ndarray] = None

    # Range-Doppler matrix in dB: shape (num_range_bins, num_doppler_bins)
    range_doppler_db: Optional[np.ndarray] = None

    # Stats & Timing (usec & %)
    inter_frame_proc_time_us: int = 0
    transmit_output_time_us: int = 0
    inter_frame_margin_us: int = 0
    inter_chirp_margin_us: int = 0
    active_frame_cpu_load: int = 0
    inter_frame_cpu_load: int = 0

    # Temperature stats (deg C)
    temps_c: Dict[str, int] = field(default_factory=dict)


def q9_log2_to_db(raw_u16: np.ndarray) -> np.ndarray:
    """
    Convert TI mmWave SDK Q9 log2 magnitude uint16 array to dB:
      dB = raw * 20 * log10(2) / 512 = raw * (6.0205999 / 512.0)
    """
    return raw_u16.astype(np.float32) * (20.0 * math.log10(2.0) / 512.0)


def db_to_q9_log2(db_arr: np.ndarray) -> np.ndarray:
    """Inverse of q9_log2_to_db for mock packet generation."""
    scale = 512.0 / (20.0 * math.log10(2.0))
    return np.clip(np.round(db_arr * scale), 0, 65535).astype(np.uint16)


def compute_azimuth_heatmap_db(
    azimuth_complex: np.ndarray, num_angle_bins: int = 64
) -> np.ndarray:
    """
    Compute Range-Azimuth heatmap (in dB) from zero-Doppler virtual antenna complex samples.
    azimuth_complex: shape (num_range_bins, num_virtual_ant_azim)
    Returns: shape (num_range_bins, num_angle_bins) in dB, angles from -90 deg to +90 deg.
    """
    fft_data = np.fft.fftshift(
        np.fft.fft(azimuth_complex, n=num_angle_bins, axis=1), axes=1
    )
    mag = np.abs(fft_data) + 1e-3
    return (20.0 * np.log10(mag)).astype(np.float32)


class TLVStreamParser:
    """
    Stateful byte-stream parser for TI mmWave UART TLV packets.
    Handles arbitrary packet fragmentation (half-packets), concatenated packets,
    corrupted headers, and 32-byte alignment padding.
    """

    MAX_PACKET_LEN = 65536

    def __init__(self, config: Optional[RadarConfig] = None) -> None:
        self.config = config or RadarConfig()
        self._buf = bytearray()
        self.frames_parsed = 0
        self.sync_losses = 0
        self.tlv_errors = 0

    def reset(self) -> None:
        self._buf.clear()

    def feed(self, data: bytes) -> List[RadarFrame]:
        """Feed raw UART bytes and return any complete RadarFrames decoded."""
        if data:
            self._buf.extend(data)

        frames: List[RadarFrame] = []

        while True:
            buf_len = len(self._buf)
            if buf_len < HEADER_SIZE:
                break

            idx = self._buf.find(MAGIC_WORD)
            if idx < 0:
                # Keep last 7 bytes in case magic word is split across chunks
                if buf_len > 7:
                    del self._buf[:-7]
                    self.sync_losses += 1
                break

            if idx > 0:
                # Discard leading garbage before magic word
                del self._buf[:idx]
                self.sync_losses += 1
                if len(self._buf) < HEADER_SIZE:
                    break

            # Unpack 40-byte MmwDemo_output_message_header
            (
                _magic,
                version,
                total_packet_len,
                platform,
                frame_number,
                time_cpu_cycles,
                num_detected_obj,
                num_tlvs,
                subframe_number,
            ) = struct.unpack(HEADER_FMT, self._buf[:HEADER_SIZE])

            # Validate header sanity: totalPacketLen must be >= 40, <= MAX_PACKET_LEN,
            # and a multiple of 32 (MMWDEMO_OUTPUT_MSG_SEGMENT_LEN)
            if (
                total_packet_len < HEADER_SIZE
                or total_packet_len > self.MAX_PACKET_LEN
                or (total_packet_len % SEGMENT_LEN) != 0
                or num_tlvs > 32
            ):
                # False magic word match; advance past first byte and resync
                del self._buf[:1]
                self.sync_losses += 1
                continue

            # Wait for full packet (half-packet buffering)
            if len(self._buf) < total_packet_len:
                break

            packet = bytes(self._buf[:total_packet_len])
            del self._buf[:total_packet_len]

            frame = self._parse_packet(
                packet,
                version,
                total_packet_len,
                platform,
                frame_number,
                time_cpu_cycles,
                num_detected_obj,
                num_tlvs,
                subframe_number,
            )
            if frame is not None:
                self.frames_parsed += 1
                frames.append(frame)

        return frames

    def _parse_packet(
        self,
        packet: bytes,
        version: int,
        total_packet_len: int,
        platform: int,
        frame_number: int,
        time_cpu_cycles: int,
        num_detected_obj: int,
        num_tlvs: int,
        subframe_number: int,
    ) -> Optional[RadarFrame]:
        frame = RadarFrame(
            version=version,
            total_packet_len=total_packet_len,
            platform=platform,
            frame_number=frame_number,
            time_cpu_cycles=time_cpu_cycles,
            num_detected_obj=num_detected_obj,
            num_tlvs=num_tlvs,
            subframe_number=subframe_number,
        )

        offset = HEADER_SIZE
        for _ in range(num_tlvs):
            if offset + TL_SIZE > total_packet_len:
                self.tlv_errors += 1
                return None

            tlv_type, tlv_len = struct.unpack(TL_FMT, packet[offset : offset + TL_SIZE])
            offset += TL_SIZE

            if offset + tlv_len > total_packet_len:
                self.tlv_errors += 1
                return None

            payload = packet[offset : offset + tlv_len]
            offset += tlv_len

            self._decode_tlv(frame, tlv_type, payload)

        return frame

    def _decode_tlv(self, frame: RadarFrame, tlv_type: int, payload: bytes) -> None:
        if tlv_type == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS:
            # DPIF_PointCloudCartesian: 4 floats (x, y, z, velocity) = 16 bytes/point
            if len(payload) % 16 == 0 and len(payload) > 0:
                frame.points = np.frombuffer(payload, dtype="<f4").reshape(-1, 4).copy()

        elif tlv_type == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO:
            # DPIF_PointCloudSideInfo: 2 int16 (snr, noise) in 0.1 dB = 4 bytes/point
            if len(payload) % 4 == 0 and len(payload) > 0:
                raw_side = np.frombuffer(payload, dtype="<i2").reshape(-1, 2)
                frame.side_info = raw_side.astype(np.float32) * 0.1

        elif tlv_type == MMWDEMO_OUTPUT_MSG_RANGE_PROFILE:
            if len(payload) % 2 == 0 and len(payload) > 0:
                raw_u16 = np.frombuffer(payload, dtype="<u2")
                frame.range_profile_db = q9_log2_to_db(raw_u16)

        elif tlv_type == MMWDEMO_OUTPUT_MSG_NOISE_PROFILE:
            if len(payload) % 2 == 0 and len(payload) > 0:
                raw_u16 = np.frombuffer(payload, dtype="<u2")
                frame.noise_profile_db = q9_log2_to_db(raw_u16)

        elif tlv_type == MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP:
            # cmplx16ImRe_t: int16 imag, int16 real (4 bytes per complex sample)
            if len(payload) % 4 == 0 and len(payload) > 0:
                im_re = np.frombuffer(payload, dtype="<i2").reshape(-1, 2)
                cplx = im_re[:, 1].astype(np.float32) + 1j * im_re[:, 0].astype(np.float32)
                num_bins = self.config.num_range_bins
                if frame.range_profile_db is not None:
                    num_bins = len(frame.range_profile_db)
                if num_bins > 0 and len(cplx) % num_bins == 0:
                    num_ant = len(cplx) // num_bins
                    frame.azimuth_Complex = cplx.reshape(num_bins, num_ant)
                    frame.azimuth_heatmap_db = compute_azimuth_heatmap_db(frame.azimuth_Complex)

        elif tlv_type == MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP:
            if len(payload) % 2 == 0 and len(payload) > 0:
                raw_u16 = np.frombuffer(payload, dtype="<u2")
                num_bins = self.config.num_range_bins
                if frame.range_profile_db is not None:
                    num_bins = len(frame.range_profile_db)
                if num_bins > 0 and len(raw_u16) % num_bins == 0:
                    num_dop = len(raw_u16) // num_bins
                    frame.range_doppler_db = q9_log2_to_db(raw_u16).reshape(num_bins, num_dop)

        elif tlv_type == MMWDEMO_OUTPUT_MSG_STATS:
            if len(payload) >= 24:
                (
                    frame.inter_frame_proc_time_us,
                    frame.transmit_output_time_us,
                    frame.inter_frame_margin_us,
                    frame.inter_chirp_margin_us,
                    frame.active_frame_cpu_load,
                    frame.inter_frame_cpu_load,
                ) = struct.unpack("<6I", payload[:24])

        elif tlv_type == MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS:
            if len(payload) >= 28:
                (
                    _valid,
                    _time,
                    rx0,
                    rx1,
                    rx2,
                    rx3,
                    tx0,
                    tx1,
                    tx2,
                    pm,
                    dig1,
                    dig2,
                ) = struct.unpack("<II10h", payload[:28])
                frame.temps_c = {
                    "RX0": rx0,
                    "TX0": tx0,
                    "PM": pm,
                    "DIG": dig1,
                }
