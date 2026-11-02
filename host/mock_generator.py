#!/usr/bin/env python3
"""
Synthetic IWR6843 OOB Binary TLV Packet Generator for Offline Testing.

Generates physically consistent binary TLV frames matching the exact TI xWR68xx OOB
firmware output format:
  - 3D Point Cloud (Type 1) + Side Info SNR/Noise (Type 7)
  - 1D Range Profile (Type 2) + 1D Noise Profile (Type 3) in Q9 log2 format
  - Azimuth Static Heatmap raw virtual-antenna complex samples (Type 4, cmplx16ImRe_t)
  - DSP/UART Timing Stats (Type 6) + RF Temperature Stats (Type 9)
  - 32-byte aligned totalPacketLen + optional random UART chunk fragmentation
"""
from __future__ import annotations

import math
import struct
from typing import List, Optional

import numpy as np

from host.tlv_parser import (
    HEADER_FMT,
    MAGIC_WORD,
    MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP,
    MMWDEMO_OUTPUT_MSG_DETECTED_POINTS,
    MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO,
    MMWDEMO_OUTPUT_MSG_NOISE_PROFILE,
    MMWDEMO_OUTPUT_MSG_RANGE_PROFILE,
    MMWDEMO_OUTPUT_MSG_STATS,
    MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS,
    SEGMENT_LEN,
    TL_FMT,
    RadarConfig,
    db_to_q9_log2,
)


class MockRadarGenerator:
    """Generates coherent synthetic IWR6843 OOB binary TLV frames."""

    def __init__(
        self,
        config: Optional[RadarConfig] = None,
        enable_azimuth_heatmap: bool = True,
        seed: int = 42,
    ) -> None:
        self.config = config or RadarConfig()
        self.enable_azimuth_heatmap = enable_azimuth_heatmap
        self.rng = np.random.default_rng(seed)
        self.frame_number = 0

    def next_frame_bytes(self, dt_s: Optional[float] = None) -> bytes:
        """Build one complete 32-byte-aligned binary TLV packet."""
        self.frame_number += 1
        if dt_s is None:
            dt_s = self.config.frame_periodicity_ms * 1e-3
        t = self.frame_number * dt_s

        num_range_bins = self.config.num_range_bins
        num_ant = self.config.num_virtual_ant_azim
        range_step = self.config.range_bin_step_m
        ranges_m = np.arange(num_range_bins, dtype=np.float32) * range_step

        # Define simulated targets: (x, y, z, vx, vy, vz, snr_db)
        # Target 1: Static calibration corner reflector at (0.0m, 1.80m, 0.0m)
        t1_x, t1_y, t1_z, t1_v, t1_snr = 0.0, 1.80, 0.0, 0.0, 32.0

        # Target 2: Radial oscillating walker between y=2.2m and y=5.8m, x=-1.1m
        omega2 = 0.9
        t2_x = -1.10 + 0.25 * math.sin(0.5 * t)
        t2_y = 4.00 + 1.80 * math.sin(omega2 * t)
        t2_z = 0.15 * math.cos(omega2 * t)
        t2_vy = 1.80 * omega2 * math.cos(omega2 * t)
        t2_r = math.sqrt(t2_x**2 + t2_y**2 + t2_z**2)
        t2_v = (t2_y * t2_vy) / max(t2_r, 0.1)
        t2_snr = 24.5

        # Target 3: Circular trajectory target centered at (1.3m, 3.4m, 0.35m)
        omega3 = 1.3
        t3_x = 1.30 + 0.90 * math.cos(omega3 * t)
        t3_y = 3.40 + 0.90 * math.sin(omega3 * t)
        t3_z = 0.35 + 0.20 * math.sin(0.7 * t)
        t3_vx = -0.90 * omega3 * math.sin(omega3 * t)
        t3_vy = 0.90 * omega3 * math.cos(omega3 * t)
        t3_r = math.sqrt(t3_x**2 + t3_y**2 + t3_z**2)
        t3_v = (t3_x * t3_vx + t3_y * t3_vy) / max(t3_r, 0.1)
        t3_snr = 20.0

        base_targets = [
            (t1_x, t1_y, t1_z, t1_v, t1_snr),
            (t2_x, t2_y, t2_z, t2_v, t2_snr),
            (t3_x, t3_y, t3_z, t3_v, t3_snr),
        ]

        # Add small micro-scatterers around moving targets
        points_list = []
        side_list = []
        noise_floor_base_db = 32.0

        for tx, ty, tz, tv, tsnr in base_targets:
            jx = tx + float(self.rng.normal(0.0, 0.015))
            jy = ty + float(self.rng.normal(0.0, 0.015))
            jz = tz + float(self.rng.normal(0.0, 0.015))
            jv = tv + float(self.rng.normal(0.0, 0.04))
            points_list.append((jx, jy, jz, jv))
            side_list.append((int(round(tsnr * 10.0)), int(round(noise_floor_base_db * 10.0))))

            # Add secondary torso/limb scatterer for moving targets
            if abs(tv) > 0.05:
                sx = tx + float(self.rng.normal(0.0, 0.12))
                sy = ty + float(self.rng.normal(0.0, 0.12))
                sz = tz + float(self.rng.normal(-0.25, 0.15))
                sv = tv + float(self.rng.normal(0.0, 0.25))
                points_list.append((sx, sy, sz, sv))
                side_list.append(
                    (int(round((tsnr - 6.5) * 10.0)), int(round(noise_floor_base_db * 10.0)))
                )

        num_obj = len(points_list)

        # Synthesize 1D Noise Profile & 1D Range Profile (in dB)
        noise_profile_db = noise_floor_base_db + self.rng.normal(0.0, 1.1, size=num_range_bins)
        # DC / near-field leakage bump near bin 0..3
        range_profile_db = noise_profile_db.copy() + self.rng.uniform(0.5, 2.2, size=num_range_bins)
        range_profile_db[:4] += np.array([18.0, 12.0, 6.0, 2.5])

        # Synthesize coherent virtual antenna array for Azimuth Static Heatmap
        # Shape: (num_range_bins, num_ant)
        ant_idx = np.arange(num_ant, dtype=np.float32)
        azim_cplx = (
            self.rng.normal(0.0, 8.0, size=(num_range_bins, num_ant))
            + 1j * self.rng.normal(0.0, 8.0, size=(num_range_bins, num_ant))
        ).astype(np.complex64)

        for tx, ty, tz, _tv, tsnr in base_targets:
            r = math.sqrt(tx**2 + ty**2 + tz**2)
            sin_theta = np.clip(tx / max(r, 0.05), -0.98, 0.98)
            # Gaussian range envelope across range bins
            env = np.exp(-0.5 * ((ranges_m - r) / (1.3 * range_step)) ** 2)
            range_profile_db += tsnr * env

            # Spatial steering vector exp(j * pi * m * sin(theta))
            amp = 15.0 * (10.0 ** (tsnr / 25.0))
            steering = np.exp(1j * math.pi * ant_idx * sin_theta)
            azim_cplx += (amp * env[:, None] * steering[None, :]).astype(np.complex64)

        tlvs: List[bytes] = []

        # TLV 1: Detected Points (DPIF_PointCloudCartesian)
        if self.config.gui_detected_objects in (1, 2) and num_obj > 0:
            pts_arr = np.array(points_list, dtype="<f4")
            payload = pts_arr.tobytes()
            tlvs.append(struct.pack(TL_FMT, MMWDEMO_OUTPUT_MSG_DETECTED_POINTS, len(payload)) + payload)

        # TLV 7: Detected Points Side Info (DPIF_PointCloudSideInfo)
        if self.config.gui_detected_objects == 1 and num_obj > 0:
            side_arr = np.array(side_list, dtype="<i2")
            payload = side_arr.tobytes()
            tlvs.append(
                struct.pack(TL_FMT, MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO, len(payload))
                + payload
            )

        # TLV 2: 1D Range Profile (Q9 log2 uint16)
        if self.config.gui_log_mag_range:
            rp_u16 = db_to_q9_log2(range_profile_db)
            payload = rp_u16.astype("<u2").tobytes()
            tlvs.append(struct.pack(TL_FMT, MMWDEMO_OUTPUT_MSG_RANGE_PROFILE, len(payload)) + payload)

        # TLV 3: 1D Noise Profile (Q9 log2 uint16)
        if self.config.gui_noise_profile:
            np_u16 = db_to_q9_log2(noise_profile_db)
            payload = np_u16.astype("<u2").tobytes()
            tlvs.append(struct.pack(TL_FMT, MMWDEMO_OUTPUT_MSG_NOISE_PROFILE, len(payload)) + payload)

        # TLV 4: Azimuth Static Heat Map (cmplx16ImRe_t: int16 imag, int16 real)
        if self.config.gui_range_azimuth_heat_map or self.enable_azimuth_heatmap:
            im_re = np.empty((num_range_bins * num_ant, 2), dtype="<i2")
            im_re[:, 0] = np.clip(np.round(azim_cplx.imag.ravel()), -32767, 32767).astype("<i2")
            im_re[:, 1] = np.clip(np.round(azim_cplx.real.ravel()), -32767, 32767).astype("<i2")
            payload = im_re.tobytes()
            tlvs.append(
                struct.pack(TL_FMT, MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP, len(payload))
                + payload
            )

        # TLV 6 + TLV 9: Stats & Temperature
        if self.config.gui_stats_info:
            inter_frame_us = 3850 + int(self.rng.integers(-120, 120))
            uart_tx_us = 14200 + int(self.rng.integers(-200, 200))
            margin_us = max(1000, int(self.config.frame_periodicity_ms * 1000) - inter_frame_us - uart_tx_us)
            stats_payload = struct.pack(
                "<6I",
                inter_frame_us,
                uart_tx_us,
                margin_us,
                4,  # interChirpProcessingMargin (us)
                22,  # activeFrameCPULoad (%)
                14,  # interFrameCPULoad (%)
            )
            tlvs.append(
                struct.pack(TL_FMT, MMWDEMO_OUTPUT_MSG_STATS, len(stats_payload)) + stats_payload
            )

            temp_payload = struct.pack(
                "<II10h",
                1,
                int(t * 1000),
                44,
                45,
                44,
                45,
                48,
                47,
                48,
                43,
                42,
                42,
            )
            tlvs.append(
                struct.pack(TL_FMT, MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS, len(temp_payload))
                + temp_payload
            )

        body = b"".join(tlvs)
        raw_len = 40 + len(body)
        total_packet_len = SEGMENT_LEN * ((raw_len + (SEGMENT_LEN - 1)) // SEGMENT_LEN)
        padding_len = total_packet_len - raw_len

        version = (3 << 24) | (6 << 16) | (2 << 8) | 0  # SDK 03.06.02.00
        cpu_cycles = (self.frame_number * 20_000_000) & 0xFFFFFFFF

        header = struct.pack(
            HEADER_FMT,
            MAGIC_WORD,
            version,
            total_packet_len,
            0x000A6843,
            self.frame_number,
            cpu_cycles,
            num_obj,
            len(tlvs),
            0,
        )

        return header + body + (b"\x00" * padding_len)

    def chunk_packet(self, packet: bytes) -> List[bytes]:
        """
        Split a binary packet into random-sized UART chunks to stress-test
        half-packet buffering and Magic Word resynchronization.
        """
        chunks: List[bytes] = []
        offset = 0
        n = len(packet)
        while offset < n:
            step = int(self.rng.integers(17, 256))
            chunks.append(packet[offset : offset + step])
            offset += step
        return chunks
