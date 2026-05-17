# parabola.py

import numpy as np
from audio.audio_utils import SR

# Speed of sound (m/s)
C_SOUND = 343.0

# Minimum effective distance to avoid near-field blowups (meters)
NEAR_FIELD_RADIUS = 6.0


def _parabola_unrotated_geometry(speed_mps, a, h, duration_s, n_samples):
    """
    τ ∈ [-1, 1], half-span from mean-speed refinement, y = a·x² + h, derivatives w.r.t. τ.
    """
    if a <= 0:
        a = abs(a) if a != 0 else 0.01
    if h <= 0:
        h = abs(h) if h != 0 else 5.0
    n_samples = max(4, int(n_samples))
    T = float(duration_s)
    if T <= 0:
        T = 1.0
    tau = np.linspace(-1.0, 1.0, n_samples)
    temp_tau = np.linspace(-1, 1, 100)
    temp_x = (speed_mps * T / 2) * temp_tau
    refinement = np.mean(np.sqrt(1 + (2 * a * temp_x) ** 2))
    half_span_x = (speed_mps * T / 2) / refinement
    x = half_span_x * tau
    y = a * x**2 + h
    dx_dtau = np.full_like(x, half_span_x)
    dy_dtau = 2.0 * a * x * half_span_x
    dtaudt = 2.0 / T
    return a, h, T, x, y, dx_dtau, dy_dtau, dtaudt


def _rotate_point_xy(x, y, angle_deg):
    if angle_deg == 0.0 or angle_deg == 0:
        return x, y
    theta = np.deg2rad(float(angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return x * cos_t - y * sin_t, x * sin_t + y * cos_t


def _rotate_vector_xy(vx, vy, angle_deg):
    if angle_deg == 0.0 or angle_deg == 0:
        return vx, vy
    theta = np.deg2rad(float(angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return vx * cos_t - vy * sin_t, vx * sin_t + vy * cos_t


def sample_parabola_path_xy(
    speed_mps, a, h, duration_s, n_points, angle_deg=0.0, cpa_time_s=None, x_offset=0.0
):
    """
    (x, y) samples matching calculate_parabola_doppler path geometry (for plots / overlays).
    """
    n_points = max(4, int(n_points))
    if cpa_time_s is not None:
        # CPA-tied construction: closest approach occurs at the requested physical time.
        t = np.linspace(0.0, duration_s, n_points, endpoint=False)
        t_cpa = float(np.clip(cpa_time_s, 1e-6, float(duration_s) - 1e-6))
        v_signed = float(speed_mps)
        if abs(v_signed) < 1e-6:
            v_signed = 1e-6
        x_along = v_signed * (t - t_cpa)
        x = x_along + float(x_offset)
        y = a * x_along**2 + h
    else:
        _a, _h, _T, x, y, _dx, _dy, _dtaudt = _parabola_unrotated_geometry(
            speed_mps, a, h, duration_s, n_points
        )
        x = x + float(x_offset)
    if angle_deg:
        return _rotate_point_xy(x, y, angle_deg)
    return x, y


def calculate_parabola_doppler(
    speed_mps,
    a,
    h,
    duration_s,
    n_steps=None,
    c_sound=343.0,
    angle_deg=0.0,
    accel_mps2=0.0,
    cpa_time_s=None,
    x_offset=0.0,
):
    """
    Compute Doppler frequency ratios and amplitudes for a parabolic path.

    Path model (observer at origin):
        x(τ) = L * τ,   τ ∈ [-1, 1]
        y(τ) = a * x(τ)^2 + h
    
    If angle_deg is provided, the entire path is rotated by this angle (in degrees)
    around the origin.

    We then map physical time t ∈ [0, T] linearly to τ ∈ [-1, 1] and
    rescale the velocity so that the *mean* speed magnitude is approximately
    speed_mps (similar to the Bezier implementation).

    This fixes:
      - Unrealistic speed explosions away from the vertex.
      - Over-aggressive Doppler ratios.
      - Sign convention differences vs. straight-line.

    Parameters
    ----------
    speed_mps : float
        Desired average speed along the parabolic path (m/s).
    a : float
        Curvature (> 0 for a "U" shape opening upwards).
    h : float
        Vertex height above observer (m, should be > 0).
    duration_s : float
        Total clip duration (s).
    n_steps : int
        Number of Doppler samples (interpolated later to audio length).

    Returns
    -------
    freq_ratios : np.ndarray
        Length n_steps, instantaneous Doppler frequency ratio f'(t)/f0.
    amplitudes : np.ndarray
        Length n_steps, amplitude envelope (normalized to max 1).
    """

    if n_steps is None:
        n_steps = int(round(SR * duration_s))
    n_steps = int(n_steps)
    if n_steps < 4:
        n_steps = 4

    a, h, T, _x0, _y0, dx_dtau, dy_dtau, dtaudt = _parabola_unrotated_geometry(
        speed_mps, a, h, duration_s, n_steps
    )
    half_span_x = float(dx_dtau[0]) if dx_dtau.size else 0.0
    x_off = float(x_offset)
    t = np.linspace(0.0, duration_s, n_steps, endpoint=False)
    dt = max(1e-9, float(duration_s) / max(1, n_steps))

    if cpa_time_s is not None:
        # Physics uses the same CPA-tied path model as plotting/labels.
        t_cpa = float(np.clip(cpa_time_s, 1e-6, float(duration_s) - 1e-6))
        v_signed = float(speed_mps)
        if abs(v_signed) < 1e-6:
            v_signed = 1e-6
        x_along = v_signed * (t - t_cpa)
        x = x_along + x_off
        y = a * x_along**2 + h
        vx_raw = np.full_like(t, v_signed)
        vy_raw = 2.0 * a * x_along * v_signed
    elif abs(float(accel_mps2)) > 1e-9:
        # B7: acceleration-aware progress along the path.
        v_t = np.maximum(1e-3, float(speed_mps) + float(accel_mps2) * t)
        s_t = np.cumsum(v_t) * dt
        total_s = max(1e-9, float(s_t[-1]))
        tau = -1.0 + 2.0 * (s_t / total_s)
        x_along = half_span_x * tau
        x = x_along + x_off
        y = a * x_along**2 + h
        dx_dtau_path = np.full_like(t, half_span_x)
        dy_dtau_path = 2.0 * a * x_along * half_span_x
        dtaudt_path = 2.0 * v_t / total_s
    else:
        tau = -1.0 + 2.0 * t / max(1e-9, float(duration_s))
        x_along = half_span_x * tau
        x = x_along + x_off
        y = a * x_along**2 + h
        dx_dtau_path = np.full_like(t, half_span_x)
        dy_dtau_path = 2.0 * a * x_along * half_span_x
        dtaudt_path = np.full_like(t, 2.0 / max(1e-9, float(duration_s)))

    if cpa_time_s is None:
        vx_raw = dx_dtau_path * dtaudt_path
        vy_raw = dy_dtau_path * dtaudt_path

    if angle_deg:
        x, y = _rotate_point_xy(x, y, angle_deg)
        vx_raw, vy_raw = _rotate_vector_xy(vx_raw, vy_raw, angle_deg)

    # Distance to observer
    r = np.sqrt(x**2 + y**2)

    # Use true distance for Doppler geometry, but avoid division by zero
    eps = 1e-9
    r_safe = np.maximum(r, eps)

    # Radial velocity: v_r = (v · r_hat) = (v · p) / |p|
    v_dot_r = vx_raw * x + vy_raw * y
    v_r = v_dot_r / r_safe

    # Clamp radial velocity to avoid unrealistic/supersonic Doppler
    max_vr = min(0.9 * c_sound, 1.2 * abs(speed_mps))
    v_r = np.clip(v_r, -max_vr, max_vr)

    # Doppler frequency ratio f'/f0 = c / (c + v_r)
    freq_ratios = c_sound / (c_sound + v_r)

    # Near-field-safe amplitude
    r_eff = np.sqrt(r**2 + NEAR_FIELD_RADIUS**2)

    # Combined amplitude with master gain and gamma compression for audibility
    amplitudes = (10.0 * (1.0 / r_eff) * (c_sound / (c_sound + v_r))**1.1)**0.7
    
    return freq_ratios.astype(np.float32), amplitudes.astype(np.float32)
