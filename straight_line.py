import numpy as np
from audio_utils import SR

C_SOUND = 343.0  # m/s
NEAR_FIELD_RADIUS = 2.0  # m – effective source size for 1/R damping


def calculate_straight_line_doppler(speed_mps, min_distance_m, angle_deg, duration_s):
    """
    Straight-line pass-by with angle and near-field-safe amplitude.

    Parameters
    ----------
    speed_mps : float
        Constant speed of the source (m/s).
    min_distance_m : float
        Closest distance from path to observer (meters). This is the "height" h.
    angle_deg : float
        Direction of motion angle in degrees (w.r.t. +x axis).
        0° = left-to-right in front of observer.
    duration_s : float
        Total duration (seconds).

    Returns
    -------
    freq_ratios : np.ndarray
        Length N (N = SR * duration_s), instantaneous Doppler frequency ratio f'/f0.
    amplitudes : np.ndarray
        Length N, amplitude envelope ~ 1 / sqrt(r^2 + r0^2) (normalized to max 1).
    """
    # Number of samples and time axis
    num_samples = int(round(SR * duration_s))
    t = np.linspace(0.0, duration_s, num_samples, endpoint=False)

    # Time of closest approach in the middle
    t0 = duration_s / 2.0
    dt = t - t0  # time relative to closest approach

    # Direction unit vector for motion
    theta = np.deg2rad(angle_deg)
    u = np.array([np.cos(theta), np.sin(theta)])  # direction of motion

    # Perpendicular unit vector (from origin to closest point on path)
    n = np.array([-np.sin(theta), np.cos(theta)])  # u · n = 0

    # Point of closest approach p_c such that |p_c| = min_distance_m
    p_c = min_distance_m * n  # shape (2,)

    # Position as function of time: p(t) = p_c + u * v * dt
    # v_vec is constant velocity vector
    v_vec = u * speed_mps  # shape (2,)
    # Broadcast dt to 2D
    p = p_c[:, None] + v_vec[:, None] * dt[None, :]  # shape (2, N)

    # Distance to observer (origin)
    r = np.linalg.norm(p, axis=0)  # (N,)

    # Radial velocity v_r = (v · r_hat) = (v · p) / |p|
    # Avoid division by zero with a tiny epsilon
    eps = 1e-9
    r_safe = np.maximum(r, eps)
    v_dot_r = np.dot(v_vec, p)  # (N,)
    v_r = v_dot_r / r_safe      # (N,)

    # Doppler frequency ratio f'/f0 = c / (c - v_r)
    freq_ratios = C_SOUND / (C_SOUND - v_r)

    # Near-field-safe amplitude: 1 / sqrt(r^2 + r0^2)
    r_eff = np.sqrt(r**2 + NEAR_FIELD_RADIUS**2)
    amplitudes = 1.0 / r_eff

    # Normalize amplitude to max 1 for safety (relative shape preserved)
    max_amp = np.max(np.abs(amplitudes)) if amplitudes.size > 0 else 1.0
    if max_amp > 0:
        amplitudes = amplitudes / max_amp

    return freq_ratios.astype(np.float32), amplitudes.astype(np.float32)
