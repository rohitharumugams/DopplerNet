"""
Single source of truth for straight-line vehicle motion (observer at origin).

Pass-by: horizontal road at y = lateral; CPA when along-track position crosses x = 0.
Miss: constant-velocity track that stays on one side of x = 0 (never crosses the observer line).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from physics.road_frame import lateral_offset_m, straight_passby_kinematics


def is_miss_trajectory(params: Dict[str, Any]) -> bool:
    """True when this clip is a non-pass-by (miss / fly-out) straight track."""
    scenario = str(params.get("motion_scenario", "") or "")
    if scenario.startswith("miss_"):
        return True
    return not bool(params.get("pass_by_in_clip", True))


def _has_track_params(params: Dict[str, Any]) -> bool:
    return all(k in params for k in ("track_x0", "track_y0", "track_vx", "track_vy"))


def validate_no_vertical_crossing(
    x0: float, vx: float, duration_s: float, *, margin_m: float = 8.0
) -> None:
    """Tracks must not cross x = 0 (no vertical pass of the observer)."""
    x0, vx, duration_s = float(x0), float(vx), float(duration_s)
    x1 = x0 + vx * duration_s
    if x0 > margin_m and x1 > margin_m:
        return
    if x0 < -margin_m and x1 < -margin_m:
        return
    raise ValueError(
        f"Track crosses observer vertical (x=0): x0={x0:.1f}, vx={vx:.1f}, "
        f"x_end={x1:.1f} over {duration_s:.1f}s"
    )


def validate_miss_along_track(x0: float, vx: float, duration_s: float, *, margin_m: float = 8.0) -> None:
    """Alias for :func:`validate_no_vertical_crossing`."""
    validate_no_vertical_crossing(x0, vx, duration_s, margin_m=margin_m)


def straight_track_positions(
    params: Dict[str, Any],
    n_points: int,
    *,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[float, float]]]:
    """
    Sample (x, y, v_x, v_y) and optional closest-approach marker for plotting.

    Returns
    -------
    x, y, v_x, v_y, closest_xy
        For pass-by, closest is (0, lateral) at x = 0. For miss, argmin range on track.
    """
    duration = float(params.get("duration", 10.0))
    t0 = float(t_start if t_start is not None else 0.0)
    t1 = float(t_end if t_end is not None else duration)
    if t1 <= t0:
        t1 = t0 + 1e-3
    t = np.linspace(t0, t1, int(n_points), endpoint=False)

    if is_miss_trajectory(params) and _has_track_params(params):
        x0 = float(params["track_x0"])
        y0 = lateral_offset_m(float(params["track_y0"]))
        vx = float(params["track_vx"])
        vy = float(params["track_vy"])
        accel = float(params.get("acceleration", 0.0))

        if abs(accel) > 1e-9:
            vmag = max(1e-9, float(np.hypot(vx, vy)))
            ux, uy = vx / vmag, vy / vmag
            s = vmag * t + 0.5 * accel * t**2
            x = x0 + ux * s
            y = y0 + uy * s
            v_t = vmag + accel * t
            v_x = ux * v_t
            v_y = uy * v_t
        else:
            x = x0 + vx * t
            y = y0 + vy * t
            v_x = np.full_like(t, vx)
            v_y = np.full_like(t, vy)

        r = np.sqrt(x * x + y * y)
        idx = int(np.argmin(r))
        closest = (float(x[idx]), float(y[idx]))
        return x, y, v_x, v_y, closest

    # Pass-by through-pass (x = 0 at CPA)
    v = float(params["speed"])
    h = float(params.get("distance", 30.0))
    angle = float(params.get("angle", 0.0))
    accel = float(params.get("acceleration", 0.0))
    from physics.recording_labels import resolve_cpa_time_s

    cpa_time_s = resolve_cpa_time_s(params, duration)

    x, y, v_x, v_y, _lat, (cpa_x, cpa_y) = straight_passby_kinematics(
        v,
        accel,
        h,
        angle,
        duration,
        cpa_time_s,
        len(t),
    )
    if t0 > 0.0 or t1 < duration - 1e-9:
        from physics.road_frame import travel_sign_from_angle

        lat = lateral_offset_m(h)
        ux = travel_sign_from_angle(angle)
        t_cpa = float(cpa_time_s)
        dt_plot = t - t_cpa
        s_plot = v * dt_plot + 0.5 * accel * dt_plot**2
        x = ux * s_plot
        y = np.full_like(t, lat)
        v_along = v + accel * dt_plot
        v_x = ux * v_along
        v_y = np.zeros_like(t)
        cpa_x, cpa_y = 0.0, lat

    return x, y, v_x, v_y, (float(cpa_x), float(cpa_y))
