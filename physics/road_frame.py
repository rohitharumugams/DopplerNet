"""
Roadside coordinate convention for DopplerNet.

Observer at the origin. Vehicle paths use lateral offset y >= MIN_ROAD_Y_M (never
negative y). Motion is primarily along x; travel direction comes from path angle.
"""

from __future__ import annotations

import numpy as np

MIN_ROAD_Y_M = 1.0


def lateral_offset_m(distance_m: float) -> float:
    """Perpendicular road offset (always positive y)."""
    return max(MIN_ROAD_Y_M, abs(float(distance_m)))


def travel_sign_from_angle(angle_deg: float) -> float:
    """+1 = increasing x (left-to-right), -1 = decreasing x (right-to-left)."""
    a = float(angle_deg) % 360.0
    return -1.0 if 90.0 <= a <= 270.0 else 1.0


def straight_passby_kinematics(
    speed_v0_mps: float,
    accel_mps2: float,
    lateral_m: float,
    angle_deg: float,
    duration_s: float,
    cpa_time_s: float | None,
    n_samples: int,
):
    """
    Legacy symmetric pass-through at x = 0 (vertical crossing).

    New pass-by clips use parallel ``track_*`` kinematics instead. Kept for callers
    that still request centered CPA geometry.

    Returns x, y, v_x, v_y (arrays length n_samples).
    """
    lateral_m = lateral_offset_m(lateral_m)
    ux = travel_sign_from_angle(angle_deg)
    t = np.linspace(0.0, float(duration_s), int(n_samples), endpoint=False)
    if cpa_time_s is not None:
        t0 = float(np.clip(cpa_time_s, 0.0, float(duration_s)))
    else:
        t0 = float(duration_s) / 2.0
    dt = t - t0
    v_along = float(speed_v0_mps) + float(accel_mps2) * dt
    s = float(speed_v0_mps) * dt + 0.5 * float(accel_mps2) * dt**2
    x = ux * s
    y = np.full_like(t, lateral_m)
    v_x = ux * v_along
    v_y = np.zeros_like(t)
    return x, y, v_x, v_y, lateral_m, (0.0, lateral_m)


def symmetric_limits_around_observer(
    x,
    y,
    observer_xy=(0.0, 0.0),
    *,
    min_half_span_x: float = 50.0,
    min_half_span_y: float = 25.0,
    pad_frac: float = 0.15,
):
    """Axis limits centered on the observer."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ox, oy = float(observer_xy[0]), float(observer_xy[1])
    hx = max(min_half_span_x, float(np.max(np.abs(x - ox))) if x.size else min_half_span_x)
    hy = max(min_half_span_y, float(np.max(np.abs(y - oy))) if y.size else min_half_span_y)
    pad_x = hx * pad_frac
    pad_y = hy * pad_frac
    return (ox - hx - pad_x, ox + hx + pad_x), (oy - hy - pad_y, oy + hy + pad_y)
