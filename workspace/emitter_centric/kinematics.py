"""Straight pass-by kinematics on the emission / observer sample grid."""

from __future__ import annotations

import numpy as np

from physics.road_frame import straight_passby_kinematics

NEAR_FIELD_RADIUS = 6.0
R_REF = 10.0
AMP_BETA = 0.7


def straight_cv_kinematics(
    speed_mps: float,
    lateral_m: float,
    angle_deg: float,
    duration_s: float,
    n_samples: int,
    *,
    accel_mps2: float = 0.0,
    cpa_time_s: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Kinematics sampled uniformly in time t in [0, duration).

    Returns x, y, v_x, v_y, r, v_r, freq_ratio (alpha), amplitude (production heuristic).
    """
    x, y, v_x, v_y, _lat, _cpa = straight_passby_kinematics(
        float(speed_mps),
        float(accel_mps2),
        float(lateral_m),
        float(angle_deg),
        float(duration_s),
        cpa_time_s,
        int(n_samples),
    )
    r = np.sqrt(x**2 + y**2)
    r_safe = np.maximum(r, 1e-9)
    v_r = (v_x * x + v_y * y) / r_safe
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "v_x": v_x.astype(np.float32),
        "v_y": v_y.astype(np.float32),
        "r": r.astype(np.float32),
        "v_r": v_r.astype(np.float32),
    }


def build_doppler_fields(
    r: np.ndarray,
    v_r: np.ndarray,
    c_sound: float,
) -> dict[str, np.ndarray]:
    """Doppler ratio and production-style amplitude from range and radial velocity."""
    r = np.asarray(r, dtype=np.float64)
    v_r = np.asarray(v_r, dtype=np.float64)
    c = float(c_sound)
    alpha = c / (c + v_r)
    spatial = R_REF / np.sqrt(r**2 + NEAR_FIELD_RADIUS**2)
    convective = (c / (c + v_r)) ** 1.0
    amp = (spatial * convective) ** AMP_BETA
    m_r = v_r / c
    return {
        "r": r.astype(np.float32),
        "v_r": v_r.astype(np.float32),
        "freq_ratio": alpha.astype(np.float32),
        "amplitude": amp.astype(np.float32),
        "m_r": m_r.astype(np.float32),
    }


def straight_cv_kinematics_with_c(
    speed_mps: float,
    lateral_m: float,
    angle_deg: float,
    duration_s: float,
    n_samples: int,
    c_sound: float,
    **kwargs,
) -> dict[str, np.ndarray]:
    base = straight_cv_kinematics(
        speed_mps, lateral_m, angle_deg, duration_s, n_samples, **kwargs
    )
    extra = build_doppler_fields(base["r"], base["v_r"], c_sound)
    return {**base, **extra}
