# bezier.py

import numpy as np
from audio_utils import SR

C_SOUND = 343.0  # m/s
NEAR_FIELD_RADIUS = 2.0  # m – effective source size for 1/R damping


def _cubic_bezier(t, p0, p1, p2, p3):
    """
    Standard cubic Bezier position.
    t: array-like in [0,1] or scalar.
    p*: floats (for x or y component).
    """
    t = np.asarray(t)
    one_minus_t = 1.0 - t
    return (one_minus_t**3) * p0 + 3 * (one_minus_t**2) * t * p1 + \
           3 * one_minus_t * (t**2) * p2 + (t**3) * p3


def _cubic_bezier_derivative(t, p0, p1, p2, p3):
    """
    Derivative of cubic Bezier w.r.t parameter t.
    """
    t = np.asarray(t)
    one_minus_t = 1.0 - t
    return 3 * (one_minus_t**2) * (p1 - p0) + \
           6 * one_minus_t * t * (p2 - p1) + \
           3 * (t**2) * (p3 - p2)


def calculate_bezier_doppler(speed_mps,
                             x0, x1, x2, x3,
                             y0, y1, y2, y3,
                             duration_s):
    """
    Cubic Bezier path with near-field-safe amplitude.

    Observer at origin (0,0).
    Spatial path is B(tau) for tau in [0,1]. We map physical time t in [0, T]
    linearly to tau, and then scale the Bezier derivative so that the *average*
    speed magnitude is approximately speed_mps.

    Parameters
    ----------
    speed_mps : float
        Desired average speed along the Bezier curve (m/s).
    x0..x3, y0..y3 : float
        Control points for the cubic Bezier in meters.
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
    if num_samples < 4:
        num_samples = 4

    t = np.linspace(0.0, duration_s, num_samples, endpoint=False)

    # Parameter tau in [0,1]
    T = float(duration_s)
    if T <= 0:
        T = 1.0
    tau = t / T

    # Bezier position
    x = _cubic_bezier(tau, x0, x1, x2, x3)
    y = _cubic_bezier(tau, y0, y1, y2, y3)

    # Bezier derivative w.r.t. tau
    dx_dtau = _cubic_bezier_derivative(tau, x0, x1, x2, x3)
    dy_dtau = _cubic_bezier_derivative(tau, y0, y1, y2, y3)

    # Convert derivative from param-space to physical time:
    # v_param = dB/dtau, actual v_raw = (dB/dtau) * d(tau)/dt = (1/T) * dB/dtau
    vx_raw = dx_dtau / T
    vy_raw = dy_dtau / T

    # Rescale so that mean |v| ≈ speed_mps
    speed_raw = np.sqrt(vx_raw**2 + vy_raw**2)
    mean_speed_raw = np.mean(speed_raw) if speed_raw.size > 0 else 0.0
    if mean_speed_raw < 1e-6:
        scale = 0.0
    else:
        scale = speed_mps / mean_speed_raw

    vx = vx_raw * scale
    vy = vy_raw * scale

    # Distance to observer
    r = np.sqrt(x**2 + y**2)

    # Use true distance for Doppler geometry, with small epsilon
    eps = 1e-9
    r_safe = np.maximum(r, eps)

    # Radial velocity v_r = (v · r_hat) = (v · p) / |p|
    v_dot_r = vx * x + vy * y
    v_r = v_dot_r / r_safe

    # Clamp radial velocity to keep Doppler ratios realistic,
    # similar behaviour to straight-line (no insane sweeps).
    max_vr = min(0.9 * C_SOUND, 1.2 * abs(speed_mps))
    v_r = np.clip(v_r, -max_vr, max_vr)

    # Doppler ratio
    freq_ratios = C_SOUND / (C_SOUND - v_r)

    # Near-field-safe amplitude
    r_eff = np.sqrt(r**2 + NEAR_FIELD_RADIUS**2)
    amplitudes = 1.0 / r_eff

    # Normalize amplitude to max 1
    max_amp = np.max(np.abs(amplitudes)) if amplitudes.size > 0 else 1.0
    if max_amp > 0:
        amplitudes = amplitudes / max_amp

    return freq_ratios.astype(np.float32), amplitudes.astype(np.float32)
