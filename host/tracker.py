#!/usr/bin/env python3
"""
Multi-Target 3D Point-Cloud Clustering & Kalman Filter Tracker for IWR6843 mmWave Radar.

Includes 3 key indoor radar anti-multipath mechanisms:
  1. Radial Azimuth Shadow / Multipath Ghost Suppression:
     A moving human at range R_1 (~1m) modulates reflections off the back wall (2m-3m) and creates
     double-bounce ghosts along the same azimuth angle θ_1. Any weaker cluster in the radial shadow
     cone behind a dominant foreground cluster (R_2 > R_1 + 0.30m, |Δθ| <= 20 deg) is suppressed.
  2. Human-Scale Cluster Merging & Track NMS (Non-Maximum Suppression):
     Merges torso + swinging arm sub-clusters within 0.78m so a single person at 1m never splits
     into two overlapping boxes.
  3. Adaptive Stationary Hold vs Ghost Fast-Prune:
     - Requires >= 5 points & 3 consecutive hits to confirm a real human track (rejecting transient
       2-3 point multipath bursts).
     - Well-established human tracks (hits >= 6) hold steady when the person stops walking; short-lived
       tracks expire immediately when their points vanish.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

TRACK_PALETTE_HEX = [
    "#facc15",  # T1: Bright Gold
    "#00f5d4",  # T2: Electric Cyan
    "#f472b6",  # T3: Pink/Magenta
    "#a3e635",  # T4: Lime Green
    "#fb923c",  # T5: Warm Orange
]

TRACK_PALETTE_RGBA = [
    (0.98, 0.80, 0.08, 0.92),  # T1: Gold
    (0.00, 0.96, 0.83, 0.92),  # T2: Cyan
    (0.96, 0.45, 0.71, 0.92),  # T3: Pink
    (0.64, 0.90, 0.21, 0.92),  # T4: Lime
    (0.98, 0.57, 0.24, 0.92),  # T5: Orange
]


@dataclass
class TrackedTarget:
    """State of a single tracked target (person)."""

    track_id: int
    # State vector: [x, y, z, vx, vy, vz]
    state: np.ndarray
    # Covariance matrix (6, 6)
    cov: np.ndarray
    hits: int = 1
    age: int = 1
    misses: int = 0
    confirmed: bool = False
    mean_vr: float = 0.0
    num_cluster_pts: int = 0
    score: float = 0.0
    # Half-extents of 3D bounding box [hx, hy, hz]
    bbox_half: np.ndarray = field(
        default_factory=lambda: np.array([0.28, 0.28, 0.58], dtype=np.float32)
    )

    @property
    def x(self) -> float:
        return float(self.state[0])

    @property
    def y(self) -> float:
        return float(self.state[1])

    @property
    def z(self) -> float:
        return float(self.state[2])

    @property
    def vx(self) -> float:
        return float(self.state[3])

    @property
    def vy(self) -> float:
        return float(self.state[4])

    @property
    def speed(self) -> float:
        return float(np.hypot(self.state[3], self.state[4]))

    @property
    def range_m(self) -> float:
        return float(np.hypot(self.state[0], self.state[1]))

    @property
    def azimuth_deg(self) -> float:
        return math.degrees(math.atan2(float(self.state[0]), max(float(self.state[1]), 1e-3)))

    @property
    def color_hex(self) -> str:
        return TRACK_PALETTE_HEX[(self.track_id - 1) % len(TRACK_PALETTE_HEX)]

    @property
    def color_rgba(self) -> Tuple[float, float, float, float]:
        return TRACK_PALETTE_RGBA[(self.track_id - 1) % len(TRACK_PALETTE_RGBA)]


class MultiTargetTracker:
    """
    DBSCAN-style 3D Point-Cloud Clusterer + Anti-Multipath Multi-Target Kalman Tracker.
    """

    def __init__(
        self,
        max_targets: int = 3,
        cluster_eps_xy: float = 0.58,
        cluster_eps_z: float = 0.85,
        min_cluster_pts: int = 5,
        confirm_hits: int = 3,
        max_coast_frames: int = 10,
        min_init_range_m: float = 0.55,
        min_init_vr_mps: float = 0.12,
        nms_merge_radius_m: float = 0.80,
        shadow_cone_deg: float = 22.0,
    ) -> None:
        self.max_targets = max_targets
        self.cluster_eps_xy = cluster_eps_xy
        self.cluster_eps_z = cluster_eps_z
        self.min_cluster_pts = min_cluster_pts
        self.confirm_hits = confirm_hits
        self.max_coast_frames = max_coast_frames
        self.min_init_range_m = min_init_range_m
        self.min_init_vr_mps = min_init_vr_mps
        self.nms_merge_radius_m = nms_merge_radius_m
        self.shadow_cone_deg = shadow_cone_deg

        self._tracks: List[TrackedTarget] = []
        self._next_id: int = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(
        self,
        raw_points: np.ndarray,
        side_info: Optional[np.ndarray] = None,
        dt_s: float = 0.20,
        mti_enabled: bool = True,
    ) -> Tuple[List[TrackedTarget], np.ndarray]:
        """
        Update tracker with current frame's raw points (N, 4) [x, y, z, v_r].
        Returns:
          - confirmed_tracks: list of confirmed TrackedTarget objects (up to max_targets)
          - display_points: filtered points (M, 4) with desk clutter & radial multipath ghosts removed
        """
        dt = max(0.04, min(float(dt_s), 0.5))

        # 1. Predict existing tracks forward
        F = np.eye(6, dtype=np.float32)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        Q = np.diag([0.012, 0.012, 0.012, 0.10, 0.10, 0.05]).astype(np.float32) * (dt / 0.2)

        for trk in self._tracks:
            trk.age += 1
            damping = 0.50 if trk.misses > 0 else 0.85
            trk.state[3:6] *= damping
            trk.state = F @ trk.state
            trk.cov = F @ trk.cov @ F.T + Q
            trk.score *= 0.85

        if len(raw_points) == 0:
            self._handle_unmatched_tracks(set())
            return self.confirmed_tracks, raw_points

        r_all = np.hypot(raw_points[:, 0], raw_points[:, 1])
        snr = (
            side_info[:, 0]
            if (side_info is not None and len(side_info) == len(raw_points))
            else np.full(len(raw_points), 15.0, dtype=np.float32)
        )

        # 2. Point-level MTI + Confirmed-Track Stationary Gate
        if mti_enabled:
            far_enough = r_all >= 0.50
            is_moving = np.abs(raw_points[:, 3]) >= 0.09

            near_confirmed_track = np.zeros(len(raw_points), dtype=bool)
            for trk in self._tracks:
                if trk.confirmed and trk.hits >= 5:
                    d_xy = np.hypot(raw_points[:, 0] - trk.x, raw_points[:, 1] - trk.y)
                    near_confirmed_track |= (d_xy <= 0.45) & (snr >= 9.0)

            keep_mask = far_enough & (is_moving | near_confirmed_track)
            filtered_pts = raw_points[keep_mask]
            filtered_snr = snr[keep_mask]
        else:
            filtered_pts = raw_points
            filtered_snr = snr

        # 3. Cluster points & suppress radial multipath ghosts behind foreground targets
        clusters = self._cluster_and_suppress_multipath(filtered_pts, filtered_snr)

        # 4. Associate surviving real clusters to existing tracks
        matched_trk_indices = set()
        matched_clu_indices = set()

        H = np.zeros((3, 6), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        R_cov = np.diag([0.025, 0.025, 0.05]).astype(np.float32)

        trk_order = sorted(
            range(len(self._tracks)),
            key=lambda i: (not self._tracks[i].confirmed, self._tracks[i].misses, -self._tracks[i].score),
        )

        for t_idx in trk_order:
            trk = self._tracks[t_idx]
            best_c_idx = -1
            best_dist = 0.72

            for c_idx, clu in enumerate(clusters):
                if c_idx in matched_clu_indices:
                    continue
                c_pos = clu["centroid"]
                dist_xy = float(np.hypot(c_pos[0] - trk.x, c_pos[1] - trk.y))
                if dist_xy < best_dist:
                    best_dist = dist_xy
                    best_c_idx = c_idx

            if best_c_idx >= 0:
                matched_trk_indices.add(t_idx)
                matched_clu_indices.add(best_c_idx)
                clu = clusters[best_c_idx]
                z_meas = clu["centroid"]

                is_standing_cluster = abs(clu["mean_vr"]) < 0.10
                R_eff = R_cov * (2.5 if is_standing_cluster else 1.0)

                y_innov = z_meas - (H @ trk.state)
                S = H @ trk.cov @ H.T + R_eff
                K = trk.cov @ H.T @ np.linalg.inv(S)
                trk.state = trk.state + (K @ y_innov)
                trk.cov = (np.eye(6, dtype=np.float32) - K @ H) @ trk.cov

                if is_standing_cluster:
                    trk.state[3:6] *= 0.30

                trk.hits += 1
                trk.misses = 0
                trk.mean_vr = clu["mean_vr"]
                trk.num_cluster_pts = clu["count"]
                trk.score = 0.6 * trk.score + 0.4 * float(clu["power_score"])
                trk.bbox_half = 0.7 * trk.bbox_half + 0.3 * clu["bbox_half"]
                if trk.hits >= self.confirm_hits:
                    trk.confirmed = True

        # 5. Handle unmatched tracks + suppress tracks that fall into the multipath shadow of a stronger track
        self._handle_unmatched_tracks(matched_trk_indices)
        self._prune_shadowed_and_duplicate_tracks()

        # 6. Spawn new tentative tracks from unmatched strong foreground clusters
        for c_idx, clu in enumerate(clusters):
            if c_idx in matched_clu_indices:
                continue
            if len(self._tracks) >= self.max_targets * 2:
                break
            c_pos = clu["centroid"]
            c_range = float(np.hypot(c_pos[0], c_pos[1]))
            c_az = math.degrees(math.atan2(float(c_pos[0]), max(float(c_pos[1]), 1e-3)))

            # Do not spawn a new track if it lies close to or in the radial shadow behind an existing confirmed track
            shadowed_by_track = False
            for trk in self._tracks:
                if not trk.confirmed:
                    continue
                if math.hypot(float(c_pos[0]) - trk.x, float(c_pos[1]) - trk.y) < self.nms_merge_radius_m:
                    shadowed_by_track = True
                    break
                if (
                    c_range > trk.range_m + 0.25
                    and abs(c_az - trk.azimuth_deg) <= self.shadow_cone_deg
                ):
                    shadowed_by_track = True
                    break

            if shadowed_by_track:
                continue

            if (
                c_range >= self.min_init_range_m
                and clu["moving_count"] >= self.min_cluster_pts
                and abs(clu["mean_vr"]) >= self.min_init_vr_mps
            ):
                init_state = np.array(
                    [c_pos[0], c_pos[1], c_pos[2], 0.0, 0.0, 0.0], dtype=np.float32
                )
                init_cov = np.diag([0.05, 0.05, 0.08, 0.25, 0.25, 0.15]).astype(np.float32)
                self._tracks.append(
                    TrackedTarget(
                        track_id=self._next_id,
                        state=init_state,
                        cov=init_cov,
                        hits=1,
                        age=1,
                        misses=0,
                        confirmed=False,
                        mean_vr=clu["mean_vr"],
                        num_cluster_pts=clu["count"],
                        score=float(clu["power_score"]),
                        bbox_half=clu["bbox_half"],
                    )
                )
                self._next_id += 1

        return self.confirmed_tracks, filtered_pts

    def _handle_unmatched_tracks(self, matched_indices: set[int]) -> None:
        surviving: List[TrackedTarget] = []
        for idx, trk in enumerate(self._tracks):
            if idx not in matched_indices:
                trk.misses += 1
                trk.state[3:6] *= 0.20
                trk.mean_vr *= 0.5
            # Only mature human tracks (hits >= 6) get full stationary coasting hold;
            # freshly spawned tracks expire in 2 frames if their points vanish.
            if not trk.confirmed:
                max_miss = 1
            elif trk.hits < 6:
                max_miss = 2
            else:
                max_miss = self.max_coast_frames
            if trk.misses <= max_miss:
                surviving.append(trk)
        self._tracks = surviving

    def _prune_shadowed_and_duplicate_tracks(self) -> None:
        """Merge duplicate tracks within NMS radius and drop weak multipath shadow tracks behind strong foreground tracks."""
        if len(self._tracks) <= 1:
            return

        # Sort by (confirmed, mature hits, score, closer range)
        self._tracks.sort(
            key=lambda t: (t.confirmed, t.hits >= 6, t.score, -t.range_m),
            reverse=True,
        )
        kept: List[TrackedTarget] = []
        for cand in self._tracks:
            drop = False
            for strong in kept:
                dist_xy = math.hypot(cand.x - strong.x, cand.y - strong.y)
                # 1. Duplicate / split-body merge
                if dist_xy < self.nms_merge_radius_m:
                    drop = True
                    break
                # 2. Radial multipath ghost shadow check: if cand is directly behind strong
                #    within shadow_cone_deg and has lower score/fewer points, drop it
                if (
                    cand.range_m > strong.range_m + 0.25
                    and abs(cand.azimuth_deg - strong.azimuth_deg) <= self.shadow_cone_deg
                    and cand.score < strong.score * 0.85
                ):
                    drop = True
                    break
            if not drop:
                kept.append(cand)
        self._tracks = kept

    @property
    def confirmed_tracks(self) -> List[TrackedTarget]:
        confirmed = [t for t in self._tracks if t.confirmed]
        # Sort by highest score so dominant target is always first, up to max_targets
        confirmed.sort(key=lambda t: t.score, reverse=True)
        top = confirmed[: self.max_targets]
        top.sort(key=lambda t: t.track_id)
        return top

    def _cluster_and_suppress_multipath(
        self, pts: np.ndarray, snr: np.ndarray
    ) -> List[dict]:
        """Group 3D points into human-sized clusters and suppress radial multipath ghosts."""
        n = len(pts)
        if n == 0:
            return []

        visited = np.zeros(n, dtype=bool)
        raw_clusters: List[dict] = []

        for i in range(n):
            if visited[i]:
                continue
            queue = [i]
            visited[i] = True
            comp = []
            while queue:
                curr = queue.pop()
                comp.append(curr)
                dx = pts[:, 0] - pts[curr, 0]
                dy = pts[:, 1] - pts[curr, 1]
                dz = pts[:, 2] - pts[curr, 2]
                neighbors = np.where(
                    (~visited)
                    & (np.hypot(dx, dy) <= self.cluster_eps_xy)
                    & (np.abs(dz) <= self.cluster_eps_z)
                )[0]
                for nb in neighbors:
                    visited[nb] = True
                    queue.append(int(nb))

            if len(comp) < 3:
                continue

            c_pts = pts[comp]
            c_snr = np.clip(snr[comp], 1.0, 60.0)
            weights = c_snr / np.sum(c_snr)

            centroid = np.sum(c_pts[:, :3] * weights[:, None], axis=0).astype(np.float32)
            mean_vr = float(np.sum(c_pts[:, 3] * weights))
            moving_count = int(np.sum(np.abs(c_pts[:, 3]) >= 0.09))
            r_m = float(np.hypot(centroid[0], centroid[1]))
            az_deg = math.degrees(math.atan2(float(centroid[0]), max(float(centroid[1]), 1e-3)))
            # Favor closer, denser, higher-SNR clusters
            power_score = float(np.sum(c_snr)) / max(r_m**0.7, 0.6)

            span = np.ptp(c_pts[:, :3], axis=0) * 0.6 + np.array([0.18, 0.18, 0.35])
            bbox_half = np.clip(span, [0.24, 0.24, 0.45], [0.45, 0.45, 0.85]).astype(
                np.float32
            )

            raw_clusters.append(
                {
                    "centroid": centroid,
                    "range_m": r_m,
                    "azimuth_deg": az_deg,
                    "mean_vr": mean_vr,
                    "count": len(comp),
                    "moving_count": moving_count,
                    "power_score": power_score,
                    "bbox_half": bbox_half,
                }
            )

        # Sort by power_score descending so the real foreground human comes first
        raw_clusters.sort(key=lambda c: c["power_score"], reverse=True)

        # Filter out multipath shadow clusters behind stronger foreground clusters
        clean_clusters: List[dict] = []
        for cand in raw_clusters:
            is_ghost = False
            for fg in clean_clusters:
                dist_xy = float(
                    np.hypot(
                        cand["centroid"][0] - fg["centroid"][0],
                        cand["centroid"][1] - fg["centroid"][1],
                    )
                )
                if dist_xy < self.nms_merge_radius_m:
                    is_ghost = True
                    break
                # Radial shadow ghost behind foreground human
                if (
                    cand["range_m"] > fg["range_m"] + 0.25
                    and abs(cand["azimuth_deg"] - fg["azimuth_deg"]) <= self.shadow_cone_deg
                ):
                    is_ghost = True
                    break
            if not is_ghost:
                clean_clusters.append(cand)

        return clean_clusters
