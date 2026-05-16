"""
Derive B1–B6 ground truth from a finished clip (path + waveform features).

Synthesis produces the recording; labels are read off that recording, not the other way around.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from physics.off_pass import PASS_BY_THRESHOLD_M
from physics.road_frame import lateral_offset_m, travel_sign_from_angle
from physics.straight_trajectory import is_miss_trajectory


def resolve_cpa_time_s(params: Dict[str, Any], duration_s: float) -> float:
    """
    CPA time anchor for path kinematics.

    For pass-by straight motion, CPA time anchors x = 0 (crosses observer vertical).
    """
    duration_s = max(1e-6, float(duration_s))
    for key in ("target_cpa_time", "cpa_time", "cpa_time_sec"):
        val = params.get(key)
        if val is None:
            continue
        try:
            t = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(t):
            return float(np.clip(t, 1e-6, duration_s - 1e-6))
    return duration_s / 2.0


def _path_xy_at_times(
    path_type: str,
    params: Dict[str, Any],
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """World (x, y) at each time sample — same kinematics as synthesis."""
    path_type = str(path_type or "straight").lower()
    duration = float(params.get("duration", 10.0))
    t = np.asarray(t, dtype=np.float64)
    n = max(4, int(t.size))

    from physics.straight_trajectory import _has_track_params

    if path_type == "straight" and is_miss_trajectory(params) and _has_track_params(params):
        x0 = float(params["track_x0"])
        y0 = lateral_offset_m(float(params["track_y0"]))
        vx, vy = float(params["track_vx"]), float(params["track_vy"])
        accel = float(params.get("acceleration", 0.0))
        if abs(accel) > 1e-9:
            vmag = max(1e-9, float(np.hypot(vx, vy)))
            ux, uy = vx / vmag, vy / vmag
            s = vmag * t + 0.5 * accel * t**2
            x = x0 + ux * s
            y = y0 + uy * s
        else:
            x = x0 + vx * t
            y = y0 + vy * t
    elif path_type == "straight":
        v = float(params["speed"])
        h = lateral_offset_m(float(params.get("distance", 30.0)))
        angle = float(params.get("angle", 0.0))
        accel = float(params.get("acceleration", 0.0))
        t_cpa = resolve_cpa_time_s(params, duration)
        ux = travel_sign_from_angle(angle)
        dt = t - t_cpa
        s = v * dt + 0.5 * accel * dt**2
        x = ux * s
        y = np.full_like(t, h)
    elif path_type == "parabola":
        from physics.parabola import sample_parabola_path_xy

        cpa_t = None
        if not is_miss_trajectory(params):
            cpa_t = resolve_cpa_time_s(params, duration)
        x, y = sample_parabola_path_xy(
            float(params["speed"]),
            float(params["a"]),
            float(params["h"]),
            duration,
            n,
            angle_deg=float(params.get("angle_deg", 0.0)),
            cpa_time_s=cpa_t,
            x_offset=float(params.get("parabola_x_offset", 0.0)),
        )
        t_uniform = np.linspace(0.0, duration, n, endpoint=False)
        if t.size != t_uniform.size or np.max(np.abs(t - t_uniform)) > 1e-6:
            from scipy.interpolate import interp1d

            fx = interp1d(t_uniform, x, kind="linear", fill_value="extrapolate")
            fy = interp1d(t_uniform, y, kind="linear", fill_value="extrapolate")
            x = fx(t)
            y = fy(t)
    elif path_type == "bezier":
        from physics.bezier import sample_bezier_path_xy

        cpa_t = None
        if not is_miss_trajectory(params):
            cpa_t = resolve_cpa_time_s(params, duration)
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
            n,
            angle_deg=float(params.get("angle_deg", 0.0)),
            cpa_time_s=cpa_t,
        )
        t_uniform = np.linspace(0.0, duration, n, endpoint=False)
        if t.size != t_uniform.size or np.max(np.abs(t - t_uniform)) > 1e-6:
            from scipy.interpolate import interp1d

            fx = interp1d(t_uniform, x, kind="linear", fill_value="extrapolate")
            fy = interp1d(t_uniform, y, kind="linear", fill_value="extrapolate")
            x = fx(t)
            y = fy(t)
    else:
        x = np.zeros(n)
        y = np.full(n, float(params.get("distance", 30.0)))

    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def path_xy_over_duration(
    path_type: str,
    params: Dict[str, Any],
    n_points: int = 201,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clip path from t=0 to t=duration (inclusive endpoints).

    Returns (time_s, x_m, y_m) with time_s[0]==0 and time_s[-1]==duration.
    """
    duration = float(params.get("duration", 10.0))
    n_points = max(4, int(n_points))
    t = np.linspace(0.0, duration, n_points, endpoint=True)
    x, y = _path_xy_at_times(path_type, params, t)
    return t, x, y


def path_range_over_time(
    path_type: str,
    params: Dict[str, Any],
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (time_s, range_m) along the same path used for synthesis."""
    t, x, y = path_xy_over_duration(path_type, params, n_samples)
    r = np.sqrt(x * x + y * y)
    return t, r


def cpa_index_on_path(t: np.ndarray, cpa_time_sec: float) -> int:
    """Index along the path sample nearest to ``cpa_time_sec``."""
    t = np.asarray(t, dtype=np.float64)
    return int(np.argmin(np.abs(t - float(cpa_time_sec))))


def _acoustic_cpa_time_sec(
    doppler_audio: np.ndarray,
    time_arr: np.ndarray,
    duration_s: float,
    rms_arr: Optional[np.ndarray] = None,
    hop: int = 512,
) -> float:
    """
    CPA time = peak short-time level on the final recording (not clip edges).
    """
    t = np.asarray(time_arr, dtype=np.float64).ravel()
    if t.size < 2:
        return 0.0

    if rms_arr is not None:
        rms = np.asarray(rms_arr, dtype=np.float64).ravel()
        if rms.size > t.size:
            rms = rms[: t.size]
        elif rms.size < t.size:
            rms = np.pad(rms, (0, t.size - rms.size))
    else:
        y = np.asarray(doppler_audio, dtype=np.float64).ravel()
        if y.size < 8:
            return float(t[0])
        import librosa

        rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
        if rms.size > t.size:
            rms = rms[: t.size]
        elif rms.size < t.size:
            rms = np.pad(rms, (0, t.size - rms.size))

    duration_s = float(duration_s)
    margin = max(0.45, 0.06 * duration_s)
    valid = (t >= margin) & (t <= duration_s - margin)
    if not np.any(valid):
        valid = np.ones_like(t, dtype=bool)

  # Smooth so single-frame spikes at t≈0 do not win over the pass-by hump.
    if rms.size >= 5:
        k = min(9, max(3, (rms.size // 40) | 1))
        kernel = np.ones(k, dtype=np.float64) / float(k)
        rms_use = np.convolve(rms, kernel, mode="same")
    else:
        rms_use = rms

    idx_local = int(np.argmax(rms_use[valid]))
    t_peak = float(t[valid][idx_local])
    return t_peak


def derive_recording_labels(
    path_type: str,
    params: Dict[str, Any],
    doppler_audio: np.ndarray,
    features: Dict[str, Any],
    vehicle_name: str,
    direction_label: int,
    direction_text: str,
) -> Dict[str, Any]:
    """
    Build unified clip labels from the synthesized recording + path geometry.
    """
    duration = float(params.get("duration", 10.0))
    n_audio = max(32, len(np.asarray(doppler_audio).ravel()))
    t_path, r_path = path_range_over_time(path_type, params, n_audio)

    min_range_m = float(np.min(r_path))
    from physics.off_pass import cpa_time_is_interior, path_crosses_observer_vertical

    crosses_vertical = path_crosses_observer_vertical(path_type, params)
    t_nom = resolve_cpa_time_s(params, duration)
    configured_pass_by = bool(params.get("pass_by_in_clip", False))
    dist_nom = float(params.get("distance", min_range_m))
    close_enough = min_range_m < PASS_BY_THRESHOLD_M or min_range_m <= dist_nom + 2.5

    pass_by_in_clip = bool(
        configured_pass_by
        and crosses_vertical
        and close_enough
        and cpa_time_is_interior(t_nom, duration)
    )

    cpa_time_sec = None
    if pass_by_in_clip:
        # Same anchor used for synthesis / path warp (B5 ground truth).
        cpa_time_sec = t_nom
        time_arr = np.asarray(features.get("time", t_path), dtype=np.float64)
        if time_arr.size != t_path.size:
            time_arr = t_path
        rms_feat = features.get("rms")
        t_peak = _acoustic_cpa_time_sec(
            doppler_audio,
            time_arr,
            duration,
            rms_arr=rms_feat,
        )
        params["cpa_time_path_sec"] = t_nom
        params["cpa_time_acoustic_sec"] = t_peak
    elif configured_pass_by and not crosses_vertical:
        pass_by_in_clip = False

    return {
        "speed_mps": float(params.get("speed", 0.0)),
        "acceleration_mps2": float(params.get("acceleration", 0.0)),
        "direction_label": int(direction_label),
        "direction_text": str(direction_text),
        "cpa_distance_m": min_range_m,
        "trajectory_type": str(path_type),
        "cpa_time_sec": cpa_time_sec,
        "pass_by_in_clip": pass_by_in_clip,
        "motion_scenario": params.get(
            "motion_scenario", "pass_by" if pass_by_in_clip else "miss"
        ),
        "num_sources": int(params.get("num_sources", 1)),
        "is_crossing": bool(params.get("is_crossing", False)),
        "vehicle_class": vehicle_name,
        "min_range_m": min_range_m,
    }
