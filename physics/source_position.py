"""
Source position tracks for generated clips.

Sampling interval
-----------------
We store (x, y, t) on a uniform 20 ms grid (50 Hz):

    Δt = 0.020 s

Rationale (simulator limits: v ≤ 100 m/s, |a| ≤ 8 m/s², typical clip 10 s):

- Spatial step at v_max: 100 × 0.02 = 2.0 m — adequate for path plots and CPA geometry.
- Acceleration over one step: 8 × 0.02 = 0.16 m/s — negligible vs speed.
- Storage: ~501 rows × 3 × 4 B ≈ 6 KB per 10 s clip; ~72 KB at 120 s.
- Round 20 ms interval is easy to reason about for analysis and visualization.

Coordinates are observer-relative (same frame as synthesis physics).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np

# Fixed temporal sampling for source_positions.npy (independent of STFT hop).
SOURCE_POSITION_DT_S = 0.020


def source_position_interval_s() -> float:
    """Seconds between stored (x, y, t) samples."""
    return SOURCE_POSITION_DT_S


def _map_xy_at_times(
    points: np.ndarray,
    speed_mps: float,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Polyline (x, y) at arclength speed × t (same law as map_trajectory Doppler)."""
    pts = np.asarray(points, dtype=float)
    t = np.asarray(t, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 1:
        raise ValueError("points must be an (N, 2) array with N >= 1")

    speed_mps = max(0.0, float(speed_mps))
    if len(pts) == 1:
        return np.full(t.shape, pts[0, 0]), np.full(t.shape, pts[0, 1])

    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative_dist = np.insert(np.cumsum(seg_lens), 0, 0.0)
    total_len = float(cumulative_dist[-1])
    if total_len < 1e-9:
        return np.full(t.shape, pts[0, 0]), np.full(t.shape, pts[0, 1])

    s = np.minimum(speed_mps * t, total_len)
    px = np.interp(s, cumulative_dist, pts[:, 0])
    py = np.interp(s, cumulative_dist, pts[:, 1])
    return px, py


def xy_at_times(
    path_type: str,
    params: Dict[str, Any],
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """World (x, y) at each time — reuses recording_labels kinematics where possible."""
    path_type = str(path_type or "straight").lower()
    t = np.asarray(t, dtype=np.float64)

    if path_type in ("map_path", "map_trajectory"):
        points = params.get("points", [])
        speed = float(params.get("speed", 30.0))
        return _map_xy_at_times(np.asarray(points, dtype=float), speed, t)

    from physics.recording_labels import _path_xy_at_times

    return _path_xy_at_times(path_type, params, t)


def compute_source_position_track(
    path_type: str,
    params: Dict[str, Any],
    duration_s: float | None = None,
) -> np.ndarray:
    """
    Build (N, 3) float32 array of [x_m, y_m, time_s] sampled uniformly in time.

    Includes t=0 and t=duration (endpoint=True) for full clip coverage.
    """
    duration_s = float(duration_s if duration_s is not None else params.get("duration", 10.0))
    duration_s = max(1e-3, duration_s)
    dt = source_position_interval_s()
    n = max(2, int(np.ceil(duration_s / dt)) + 1)
    t = np.linspace(0.0, duration_s, n, endpoint=True)
    x, y = xy_at_times(path_type, params, t)
    return np.column_stack([
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        t.astype(np.float32),
    ])


def save_source_positions_npy(
    track: np.ndarray,
    common_dir: str,
    essential_dir: str,
    filename: str = "source_positions.npy",
) -> None:
    """Write identical source_positions.npy under Common/ and Essential/."""
    track = np.asarray(track, dtype=np.float32)
    os.makedirs(common_dir, exist_ok=True)
    os.makedirs(essential_dir, exist_ok=True)
    np.save(os.path.join(common_dir, filename), track)
    np.save(os.path.join(essential_dir, filename), track)
