"""
Pass-by and miss motion (straight, parabola, bezier).

Pass-by: path crosses the observer vertical (x = 0) with CPA time inside the clip.

Miss: path stays on one side of x = 0; closest point at clip edge (t≈0 or t≈T).
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np

from physics.road_frame import lateral_offset_m

# Closest approach below this (m) counts as in-clip pass-by for labeling (B5/B6).
PASS_BY_THRESHOLD_M = 25.0

# Pass-by lateral band: close enough for in-clip CPA labels (B5/B6).
PASS_BY_DISTANCE_MIN_M = 5.0
PASS_BY_DISTANCE_MAX_M = 24.0

# Miss-distance band: audible but not a near pass-by.
MISS_DISTANCE_MIN_M = 28.0
MISS_DISTANCE_MAX_M = 75.0

# Along-track start (m); tracks stay on one side of x = 0.
_ALONG_START_PASS_BY = {
    "pass_by_parallel_right": (35.0, 110.0),
    "pass_by_parallel_left": (-110.0, -35.0),
}

# Along-track start offsets (m from observer). Straight miss tracks stay on one side of x = 0.
_ALONG_START = {
    "miss_recede_right": (45.0, 130.0),
    "miss_recede_left": (-130.0, -45.0),
    "miss_parallel_right": (70.0, 175.0),
    "miss_parallel_left": (-175.0, -70.0),
}


def sample_miss_distance_m() -> float:
    return float(random.uniform(MISS_DISTANCE_MIN_M, MISS_DISTANCE_MAX_M))


def sample_passby_distance_m(params: Dict[str, Any]) -> float:
    """Lateral offset for pass-by (below PASS_BY_THRESHOLD_M)."""
    d = float(params.get("distance", 15.0))
    d = float(np.clip(d, PASS_BY_DISTANCE_MIN_M, PASS_BY_DISTANCE_MAX_M))
    return lateral_offset_m(d)


_MISS_SCENARIO_IDS = (
    "miss_recede_right",
    "miss_recede_left",
    "miss_parallel_right",
    "miss_parallel_left",
)


def sample_benchmark_motion_scenario(
    bench_params: Optional[dict] = None,
    *,
    pass_by_fraction: float = 0.80,
    clip_index: Optional[int] = None,
    total_clips: Optional[int] = None,
    force_pass_by: Optional[bool] = None,
) -> str:
    """Return ``pass_by`` or a miss scenario id."""
    if force_pass_by is True:
        return "pass_by"
    if force_pass_by is False:
        return random.choice(_MISS_SCENARIO_IDS)

    bench_params = bench_params or {}
    frac = float(bench_params.get("pass_by_fraction", pass_by_fraction))
    frac = max(0.0, min(1.0, frac))

    if clip_index is not None and total_clips is not None and int(total_clips) > 0:
        n_miss = int(round(int(total_clips) * (1.0 - frac)))
        n_miss = max(0, min(int(total_clips), n_miss))
        if int(clip_index) <= n_miss:
            return random.choice(_MISS_SCENARIO_IDS)
        return "pass_by"

    if random.random() < frac:
        return "pass_by"
    return random.choice(_MISS_SCENARIO_IDS)


def min_range_straight_track(
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    duration_s: float,
    accel_mps2: float = 0.0,
    n_samples: int = 400,
) -> float:
    """Minimum distance to origin along a straight track (observer at 0,0)."""
    t = np.linspace(0.0, max(1e-6, float(duration_s)), int(n_samples), endpoint=False)
    if abs(accel_mps2) > 1e-9:
        vmag = max(1e-9, float(np.hypot(vx, vy)))
        ux, uy = vx / vmag, vy / vmag
        s = vmag * t + 0.5 * float(accel_mps2) * t**2
        x = x0 + ux * s
        y = y0 + uy * s
    else:
        x = x0 + vx * t
        y = y0 + vy * t
    return float(np.min(np.sqrt(x * x + y * y)))


def min_range_on_path(path_type: str, params: Dict[str, Any], n_samples: int = 400) -> float:
    """Minimum distance to origin along the synthesized path."""
    duration = float(params.get("duration", 10.0))
    n_samples = max(32, int(n_samples))

    if path_type == "straight":
        if all(k in params for k in ("track_x0", "track_y0", "track_vx", "track_vy")):
            return min_range_straight_track(
                float(params["track_x0"]),
                lateral_offset_m(float(params["track_y0"])),
                float(params["track_vx"]),
                float(params["track_vy"]),
                duration,
                float(params.get("acceleration", 0.0)),
                n_samples,
            )
        from physics.recording_labels import path_xy_over_duration

        _t, x, y = path_xy_over_duration("straight", params, n_samples)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return float(np.min(np.sqrt(x * x + y * y)))

    if path_type == "parabola":
        from physics.parabola import sample_parabola_path_xy

        x, y = sample_parabola_path_xy(
            float(params["speed"]),
            float(params["a"]),
            float(params["h"]),
            duration,
            n_samples,
            angle_deg=float(params.get("angle_deg", 0.0)),
            cpa_time_s=None,
            x_offset=float(params.get("parabola_x_offset", 0.0)),
        )
        return float(np.min(np.sqrt(x * x + y * y)))

    if path_type == "bezier":
        from physics.bezier import sample_bezier_path_xy

        x, y = sample_bezier_path_xy(
            float(params["speed"]),
            float(params["x0"]),
            float(params["x1"]),
            float(params["x2"]),
            float(params["x3"]),
            float(params["y0"]),
            float(params["y1"]),
            float(params["y2"]),
            float(params["y3"]),
            duration,
            n_samples,
            angle_deg=float(params.get("angle_deg", 0.0)),
            cpa_time_s=None,
        )
        return float(np.min(np.sqrt(x * x + y * y)))

    raise ValueError(f"Unsupported path type for miss range check: {path_type}")


def _clear_track_params(params: Dict[str, Any]) -> None:
    for key in ("track_x0", "track_y0", "track_vx", "track_vy"):
        params.pop(key, None)
    params.pop("passby_bezier_uses_straight_kinematics", None)


def configure_straight_through_pass_params(params: Dict[str, Any]) -> None:
    """Straight pass-by: crosses x = 0 at ``target_cpa_time`` (lateral offset = distance)."""
    lateral = sample_passby_distance_m(params)
    params["distance"] = lateral
    _clear_track_params(params)
    params.pop("parabola_x_offset", None)


def configure_parabola_through_pass_params(params: Dict[str, Any]) -> None:
    """Parabola pass-by: vertex at x = 0 (crosses observer vertical at CPA)."""
    h = sample_passby_distance_m(params)
    params["h"] = h
    params["distance"] = h
    _clear_track_params(params)
    params.pop("parabola_x_offset", None)
    if "a" not in params:
        params["a"] = random.uniform(5, 20) / 10000.0
    params["angle_deg"] = float(params.get("angle_deg", 0.0))


def configure_bezier_through_pass_params(params: Dict[str, Any]) -> None:
    """Bezier pass-by: curve spans both sides of x = 0 (through-pass)."""
    speed = float(params.get("speed", 25.0))
    duration = float(params.get("duration", 10.0))
    t_cpa = float(params.get("target_cpa_time", params.get("cpa_time", duration * 0.5)))
    t_cpa = float(np.clip(t_cpa, 1e-3, max(1e-3, duration - 1e-3)))
    left_span = max(12.0, abs(speed) * t_cpa)
    right_span = max(12.0, abs(speed) * (duration - t_cpa))
    dist = sample_passby_distance_m(params)
    params["distance"] = dist
    _clear_track_params(params)
    params.pop("parabola_x_offset", None)

    if random.random() > 0.5:
        x_start, x_end = -left_span, right_span
    else:
        x_start, x_end = right_span, -left_span
    params["x0"] = x_start
    params["x3"] = x_end
    lo, hi = min(x_start, x_end), max(x_start, x_end)
    params["x1"] = lo + 0.33 * (hi - lo)
    params["x2"] = lo + 0.66 * (hi - lo)
    # Keep lateral offset near ``dist`` so min range stays in the pass-by band (B5/B6).
    params["y0"] = dist
    params["y3"] = dist
    params["y1"] = dist + random.uniform(-1.0, 2.0)
    params["y2"] = dist + random.uniform(-1.0, 2.0)
    params["angle_deg"] = float(params.get("angle_deg", 0.0))


def path_crosses_observer_vertical(
    path_type: str,
    params: Dict[str, Any],
    *,
    margin_m: float = 1.0,
    n_points: int = 201,
) -> bool:
    """True when the path passes both sides of the observer meridian x = 0."""
    from physics.recording_labels import path_xy_over_duration

    _t, x, _y = path_xy_over_duration(path_type, params, n_points)
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return False
    # Opposite signs ⇒ crosses the vertical through the observer.
    return bool(np.min(x) < -float(margin_m) and np.max(x) > float(margin_m))


def cpa_time_is_interior(t_cpa: float, duration_s: float, *, frac: float = 0.08) -> bool:
    """CPA must not sit on a clip edge (parallel miss → argmin at 0 or T)."""
    duration_s = float(duration_s)
    edge = max(0.45, float(frac) * duration_s)
    t_cpa = float(t_cpa)
    return edge < t_cpa < duration_s - edge


def _bezier_phys_scale(
    speed_mps: float,
    duration_s: float,
    x0: float,
    x1: float,
    x2: float,
    x3: float,
    y0: float,
    y1: float,
    y2: float,
    y3: float,
) -> float:
    """Match ``sample_bezier_path_xy`` spatial scaling (control points → metres)."""
    from physics.bezier import _cubic_bezier_derivative

    T = max(1e-6, float(duration_s))
    tau = np.linspace(0.0, 1.0, 48, endpoint=False)
    dx_dtau = _cubic_bezier_derivative(tau, x0, x1, x2, x3)
    dy_dtau = _cubic_bezier_derivative(tau, y0, y1, y2, y3)
    v_init = np.sqrt((dx_dtau / T) ** 2 + (dy_dtau / T) ** 2)
    mean_v = float(np.mean(v_init)) if v_init.size else 1.0
    return float(speed_mps) / max(mean_v, 1e-6)


def _tighten_passby_geometry(params: Dict[str, Any], path_type: str) -> None:
    """Pull curved pass-by paths closer when range check fails."""
    path_type = str(path_type or "straight").lower()
    if path_type == "straight":
        params["track_y0"] = lateral_offset_m(
            max(
                PASS_BY_DISTANCE_MIN_M,
                float(params.get("track_y0", PASS_BY_DISTANCE_MAX_M)) - 3.0,
            )
        )
        params["distance"] = float(params["track_y0"])
    elif path_type == "parabola":
        params["h"] = lateral_offset_m(
            max(PASS_BY_DISTANCE_MIN_M, float(params.get("h", PASS_BY_DISTANCE_MAX_M)) - 3.0)
        )
        params["distance"] = float(params["h"])
        off = float(params.get("parabola_x_offset", 12.0))
        params["parabola_x_offset"] = off * 0.82 if abs(off) > 1.0 else off
    else:
        for key in ("y0", "y1", "y2", "y3"):
            params[key] = lateral_offset_m(
                max(PASS_BY_DISTANCE_MIN_M, float(params.get(key, PASS_BY_DISTANCE_MAX_M)) - 2.0)
            )
        params["distance"] = float(params["y1"])
        for xk in ("x0", "x3"):
            x = float(params[xk])
            params[xk] = x * 0.88 if abs(x) > 1.0 else x


def configure_passby_for_path(params: Dict[str, Any], path_type: str) -> None:
    """Through-pass geometry: crosses x = 0 with in-clip CPA (``target_cpa_time``)."""
    from physics.cpa_timing import sample_benchmark_cpa_time

    path_type = str(path_type or "straight").lower()
    duration = float(params.get("duration", 10.0))
    params["pass_by_in_clip"] = True
    params["motion_scenario"] = "pass_by"

    t_cpa = params.get("target_cpa_time", params.get("cpa_time"))
    if t_cpa is None:
        t_cpa = sample_benchmark_cpa_time({}, duration)
    else:
        t_cpa = float(t_cpa)
    if not cpa_time_is_interior(t_cpa, duration):
        t_cpa = sample_benchmark_cpa_time({}, duration)
    params["target_cpa_time"] = t_cpa
    params["cpa_time"] = t_cpa

    last_err = None
    for _attempt in range(32):
        try:
            if path_type == "straight":
                configure_straight_through_pass_params(params)
            elif path_type == "parabola":
                configure_parabola_through_pass_params(params)
            elif path_type == "bezier":
                configure_bezier_through_pass_params(params)
            else:
                configure_straight_through_pass_params(params)

            params["target_cpa_time"] = t_cpa
            params["cpa_time"] = t_cpa
            params["min_range_m"] = min_range_on_path(path_type, params)

            if path_crosses_observer_vertical(path_type, params) and cpa_time_is_interior(
                t_cpa, duration
            ):
                return

            last_err = (
                f"{path_type}: crosses={path_crosses_observer_vertical(path_type, params)}, "
                f"min_range={params['min_range_m']:.1f}m"
            )
            t_cpa = sample_benchmark_cpa_time({}, duration)
            params["target_cpa_time"] = t_cpa
            params["cpa_time"] = t_cpa
        except Exception as exc:
            last_err = str(exc)
            t_cpa = sample_benchmark_cpa_time({}, duration)
            params["target_cpa_time"] = t_cpa
            params["cpa_time"] = t_cpa

    raise RuntimeError(
        f"Could not configure {path_type} through-pass pass-by "
        f"(last: {last_err})"
    )


def configure_straight_miss_params(params: Dict[str, Any], scenario: str) -> None:
    """Straight-line miss (observer at origin)."""
    lateral = lateral_offset_m(sample_miss_distance_m())
    speed = float(params.get("speed", 25.0))
    duration = float(params.get("duration", 10.0))

    params["distance"] = lateral
    params.pop("parabola_x_offset", None)

    if scenario == "miss_recede_right":
        x0 = float(random.uniform(*_ALONG_START["miss_recede_right"]))
        params["angle"] = 0
        params["track_x0"] = x0
        params["track_y0"] = lateral
        params["track_vx"] = speed
        params["track_vy"] = 0.0
    elif scenario == "miss_recede_left":
        x0 = float(random.uniform(*_ALONG_START["miss_recede_left"]))
        params["angle"] = 180
        params["track_x0"] = x0
        params["track_y0"] = lateral
        params["track_vx"] = -speed
        params["track_vy"] = 0.0
    elif scenario == "miss_parallel_right":
        x0 = float(random.uniform(*_ALONG_START["miss_parallel_right"]))
        params["angle"] = 0
        params["track_x0"] = x0
        params["track_y0"] = lateral
        params["track_vx"] = speed
        params["track_vy"] = 0.0
    elif scenario == "miss_parallel_left":
        x0 = float(random.uniform(*_ALONG_START["miss_parallel_left"]))
        params["angle"] = 180
        params["track_x0"] = x0
        params["track_y0"] = lateral
        params["track_vx"] = -speed
        params["track_vy"] = 0.0
    else:
        raise ValueError(f"Unknown miss scenario: {scenario}")

    from physics.straight_trajectory import validate_no_vertical_crossing

    validate_no_vertical_crossing(
        params["track_x0"],
        params["track_vx"],
        duration,
    )


def configure_parabola_miss_params(params: Dict[str, Any], scenario: str) -> None:
    """Offset parabola vertex so closest approach is a wide miss, not x=0 CPA."""
    miss_h = lateral_offset_m(sample_miss_distance_m())
    speed = float(params.get("speed", 25.0))
    duration = float(params.get("duration", 10.0))

    params["h"] = miss_h
    params["distance"] = miss_h
    params.pop("track_x0", None)
    params.pop("track_y0", None)
    params.pop("track_vx", None)
    params.pop("track_vy", None)

    if "a" not in params:
        params["a"] = random.uniform(5, 20) / 10000.0

    right = "right" in scenario
    half_span = max(40.0, 0.45 * speed * duration)
    x_off = (1.0 if right else -1.0) * random.uniform(
        half_span + 25.0,
        half_span + 110.0,
    )
    params["parabola_x_offset"] = float(x_off)
    params["angle_deg"] = float(params.get("angle_deg", 0.0))


def configure_bezier_miss_params(params: Dict[str, Any], scenario: str) -> None:
    """Bezier arc on one side of the observer with y ≈ miss distance."""
    miss_y = lateral_offset_m(sample_miss_distance_m())
    speed = float(params.get("speed", 25.0))
    duration = float(params.get("duration", 10.0))
    span = max(35.0, 0.42 * speed * duration)

    params["distance"] = miss_y
    params.pop("parabola_x_offset", None)
    params.pop("track_x0", None)
    params.pop("track_y0", None)
    params.pop("track_vx", None)
    params.pop("track_vy", None)

    right = "right" in scenario
    recede = "recede" in scenario
    y_end = miss_y + random.uniform(10.0, 28.0)

    if right:
        x_start = random.uniform(50.0, 115.0)
        x_delta = random.uniform(25.0, span) if recede else random.uniform(span * 0.55, span)
        params["x0"] = x_start
        params["x3"] = x_start + x_delta
    else:
        x_start = random.uniform(-115.0, -50.0)
        x_delta = random.uniform(25.0, span) if recede else random.uniform(span * 0.55, span)
        params["x0"] = x_start
        params["x3"] = x_start - x_delta

    lo_x = min(params["x0"], params["x3"])
    hi_x = max(params["x0"], params["x3"])
    params["x1"] = lo_x + 0.33 * (hi_x - lo_x)
    params["x2"] = lo_x + 0.66 * (hi_x - lo_x)
    params["y0"] = y_end
    params["y3"] = y_end
    params["y1"] = miss_y + random.uniform(-1.5, 4.0)
    params["y2"] = miss_y + random.uniform(-1.5, 4.0)
    params["angle_deg"] = float(params.get("angle_deg", 0.0))


def configure_miss_for_path(
    params: Dict[str, Any],
    scenario: str,
    path_type: str,
) -> None:
    """Configure non-pass-by motion; keeps ``path_type`` (straight / parabola / bezier)."""
    params["pass_by_in_clip"] = False
    params["motion_scenario"] = scenario
    params.pop("target_cpa_time", None)
    params.pop("cpa_time", None)
    params["acceleration"] = 0.0

    path_type = str(path_type or "straight").lower()
    duration = float(params.get("duration", 10.0))

    for _ in range(40):
        if path_type == "parabola":
            configure_parabola_miss_params(params, scenario)
        elif path_type == "bezier":
            configure_bezier_miss_params(params, scenario)
        else:
            configure_straight_miss_params(params, scenario)

        min_r = min_range_on_path(path_type, params)
        params["min_range_m"] = min_r
        if min_r >= PASS_BY_THRESHOLD_M:
            return

        # Push miss geometry outward and resample.
        if path_type == "straight":
            params["track_y0"] = lateral_offset_m(
                float(params.get("track_y0", MISS_DISTANCE_MIN_M)) + 8.0
            )
            params["distance"] = float(params["track_y0"])
        elif path_type == "parabola":
            params["h"] = lateral_offset_m(float(params.get("h", MISS_DISTANCE_MIN_M)) + 8.0)
            params["distance"] = float(params["h"])
            sign = 1.0 if float(params.get("parabola_x_offset", 0.0)) >= 0 else -1.0
            params["parabola_x_offset"] = sign * (
                abs(float(params["parabola_x_offset"])) + random.uniform(12.0, 28.0)
            )
        else:
            bump = 8.0
            for key in ("y0", "y1", "y2", "y3"):
                params[key] = lateral_offset_m(float(params.get(key, MISS_DISTANCE_MIN_M)) + bump)
            params["distance"] = float(params["y1"])

    raise RuntimeError(
        f"Could not sample {path_type} miss path with min_range >= {PASS_BY_THRESHOLD_M} m"
    )


def configure_benchmark_motion(
    params: Dict[str, Any],
    bench_params: dict,
    path_type: str,
    *,
    clip_index: Optional[int] = None,
    total_clips: Optional[int] = None,
) -> Tuple[str, bool]:
    """
    Configure pass-by vs miss motion for a benchmark clip.

    Returns (scenario_id, pass_by_in_clip). Miss clips keep the batch ``path_type``.
    """
    forced = params.pop("_motion_pass_by", None)
    force_pass_by = None if forced is None else bool(forced)
    scenario = sample_benchmark_motion_scenario(
        bench_params,
        clip_index=clip_index,
        total_clips=total_clips,
        force_pass_by=force_pass_by,
    )
    if scenario == "pass_by":
        configure_passby_for_path(params, path_type)
        return scenario, True

    configure_miss_for_path(params, scenario, path_type)
    return scenario, False
