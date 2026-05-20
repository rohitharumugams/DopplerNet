# bezier.py

import numpy as np
from audio.audio_utils import SR

C_SOUND = 343.0  # m/s
NEAR_FIELD_RADIUS = 6.0  # m – broader near-field for smoother pass-by envelope


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


def _scaled_bezier_geometry(speed_mps, x0, x1, x2, x3, y0, y1, y2, y3, duration_s, n_samples):
    """Build scaled Bezier geometry exactly as used by Doppler physics."""
    n_samples = max(4, int(n_samples))
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False)
    T = float(duration_s)
    if T <= 0:
        T = 1.0
    tau = t / T

    dx_dtau_init = _cubic_bezier_derivative(tau, x0, x1, x2, x3)
    dy_dtau_init = _cubic_bezier_derivative(tau, y0, y1, y2, y3)
    vx_init = dx_dtau_init / T
    vy_init = dy_dtau_init / T
    speed_init = np.sqrt(vx_init**2 + vy_init**2)
    mean_speed_init = np.mean(speed_init) if speed_init.size > 0 else 1.0
    phys_scale = speed_mps / mean_speed_init

    xs0, xs1, xs2, xs3 = x0 * phys_scale, x1 * phys_scale, x2 * phys_scale, x3 * phys_scale
    ys0, ys1, ys2, ys3 = y0 * phys_scale, y1 * phys_scale, y2 * phys_scale, y3 * phys_scale

    x = _cubic_bezier(tau, xs0, xs1, xs2, xs3)
    y = _cubic_bezier(tau, ys0, ys1, ys2, ys3)
    dx_dtau = _cubic_bezier_derivative(tau, xs0, xs1, xs2, xs3)
    dy_dtau = _cubic_bezier_derivative(tau, ys0, ys1, ys2, ys3)
    vx = dx_dtau / T
    vy = dy_dtau / T
    return x, y, vx, vy


def _bezier_tau_at_closest_approach(xs0, xs1, xs2, xs3, ys0, ys1, ys2, ys3, n_scan=400):
    tau = np.linspace(0.0, 1.0, max(32, int(n_scan)))
    x = _cubic_bezier(tau, xs0, xs1, xs2, xs3)
    y = _cubic_bezier(tau, ys0, ys1, ys2, ys3)
    return float(tau[int(np.argmin(np.sqrt(x * x + y * y)))])


def sample_bezier_path_xy(
    speed_mps,
    x0,
    x1,
    x2,
    x3,
    y0,
    y1,
    y2,
    y3,
    duration_s,
    n_points,
    angle_deg=0.0,
    cpa_time_s=None,
):
    """(x, y) samples that match calculate_bezier_doppler geometry."""
    n = max(4, int(n_points))
    T = float(duration_s) if float(duration_s) > 0 else 1.0
    t = np.linspace(0.0, T, n, endpoint=False)

    dx_dtau_init = _cubic_bezier_derivative(t / T, x0, x1, x2, x3)
    dy_dtau_init = _cubic_bezier_derivative(t / T, y0, y1, y2, y3)
    speed_init = np.sqrt(dx_dtau_init**2 + dy_dtau_init**2) / T
    mean_speed_init = np.mean(speed_init) if speed_init.size > 0 else 1.0
    phys_scale = speed_mps / max(1e-6, mean_speed_init)
    xs0, xs1, xs2, xs3 = x0 * phys_scale, x1 * phys_scale, x2 * phys_scale, x3 * phys_scale
    ys0, ys1, ys2, ys3 = y0 * phys_scale, y1 * phys_scale, y2 * phys_scale, y3 * phys_scale

    if cpa_time_s is not None:
        from physics.cpa_timing import warp_tau_for_cpa

        tau_cpa = _bezier_tau_at_closest_approach(xs0, xs1, xs2, xs3, ys0, ys1, ys2, ys3)
        tau, _ = warp_tau_for_cpa(t, T, float(cpa_time_s), tau_cpa)
        x = _cubic_bezier(tau, xs0, xs1, xs2, xs3)
        y = _cubic_bezier(tau, ys0, ys1, ys2, ys3)
    else:
        x, y, _vx, _vy = _scaled_bezier_geometry(
            speed_mps, x0, x1, x2, x3, y0, y1, y2, y3, duration_s, n
        )
    if angle_deg:
        x, y = _rotate_point_xy(x, y, angle_deg)
    return x, y


def calculate_bezier_doppler(
    speed_mps,
    x0,
    x1,
    x2,
    x3,
    y0,
    y1,
    y2,
    y3,
    duration_s,
    c_sound=343.0,
    angle_deg=0.0,
    accel_mps2=0.0,
    cpa_time_s=None,
):
    """
    Cubic Bezier path with near-field-safe amplitude.

    Observer at origin (0,0).
    Spatial path is B(tau) for tau in [0,1]. We map physical time t in [0, T]
    linearly to tau, and then scale the Bezier derivative so that the *average*
    speed magnitude is approximately speed_mps.

    If angle_deg is provided, the entire path is rotated by this angle (in degrees)
    around the origin.

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

    # Use scaled Bezier control points, then apply acceleration-aware timing.
    n = num_samples
    T = float(duration_s) if float(duration_s) > 0 else 1.0
    t = np.linspace(0.0, T, n, endpoint=False)
    dt = max(1e-9, T / max(1, n))

    dx_dtau_init = _cubic_bezier_derivative(t / T, x0, x1, x2, x3)
    dy_dtau_init = _cubic_bezier_derivative(t / T, y0, y1, y2, y3)
    vx_init = dx_dtau_init / T
    vy_init = dy_dtau_init / T
    speed_init = np.sqrt(vx_init**2 + vy_init**2)
    mean_speed_init = np.mean(speed_init) if speed_init.size > 0 else 1.0
    phys_scale = speed_mps / max(1e-6, mean_speed_init)

    xs0, xs1, xs2, xs3 = x0 * phys_scale, x1 * phys_scale, x2 * phys_scale, x3 * phys_scale
    ys0, ys1, ys2, ys3 = y0 * phys_scale, y1 * phys_scale, y2 * phys_scale, y3 * phys_scale

    # Build dense arc-length table for exact mapping
    N_dense = 10000
    tau_dense = np.linspace(0.0, 1.0, N_dense)
    dx_dtau_dense = _cubic_bezier_derivative(tau_dense, xs0, xs1, xs2, xs3)
    dy_dtau_dense = _cubic_bezier_derivative(tau_dense, ys0, ys1, ys2, ys3)
    ds_dtau_dense = np.sqrt(dx_dtau_dense**2 + dy_dtau_dense**2)
    s_dense = np.concatenate(([0.0], np.cumsum(0.5 * (ds_dtau_dense[1:] + ds_dtau_dense[:-1])) * (1.0 / (N_dense - 1))))
    total_L = s_dense[-1]

    # Time-zero alignment with CPA
    tau_cpa = _bezier_tau_at_closest_approach(xs0, xs1, xs2, xs3, ys0, ys1, ys2, ys3)
    s_cpa = np.interp(tau_cpa, tau_dense, s_dense)
    
    if cpa_time_s is not None:
        t_cpa = float(np.clip(cpa_time_s, 0.0, T))
    else:
        t_cpa = T / 2.0
    
    dt_arr = t - t_cpa
    v_t = np.maximum(1e-3, float(speed_mps) + float(accel_mps2) * dt_arr)
    s_t = s_cpa + float(speed_mps) * dt_arr + 0.5 * float(accel_mps2) * dt_arr**2
    
    # Map physical distance s(t) back to curve parameter tau
    tau = np.interp(s_t, s_dense, tau_dense)
    valid_mask = (s_t >= 0.0) & (s_t <= total_L)
    
    x = _cubic_bezier(tau, xs0, xs1, xs2, xs3)
    y = _cubic_bezier(tau, ys0, ys1, ys2, ys3)
    
    dx_dtau_actual = _cubic_bezier_derivative(tau, xs0, xs1, xs2, xs3)
    dy_dtau_actual = _cubic_bezier_derivative(tau, ys0, ys1, ys2, ys3)
    ds_dtau_actual = np.maximum(np.sqrt(dx_dtau_actual**2 + dy_dtau_actual**2), 1e-9)
    
    # Extrapolate out-of-bounds positions along tangent
    # At tau = 0
    dx_0 = _cubic_bezier_derivative(0.0, xs0, xs1, xs2, xs3)
    dy_0 = _cubic_bezier_derivative(0.0, ys0, ys1, ys2, ys3)
    ds_0 = max(np.hypot(dx_0, dy_0), 1e-9)
    ux_0, uy_0 = dx_0 / ds_0, dy_0 / ds_0
    x_0, y_0 = _cubic_bezier(0.0, xs0, xs1, xs2, xs3), _cubic_bezier(0.0, ys0, ys1, ys2, ys3)
    
    mask_before = (s_t < 0.0)
    x[mask_before] = x_0 + ux_0 * s_t[mask_before]
    y[mask_before] = y_0 + uy_0 * s_t[mask_before]
    
    # At tau = 1
    dx_1 = _cubic_bezier_derivative(1.0, xs0, xs1, xs2, xs3)
    dy_1 = _cubic_bezier_derivative(1.0, ys0, ys1, ys2, ys3)
    ds_1 = max(np.hypot(dx_1, dy_1), 1e-9)
    ux_1, uy_1 = dx_1 / ds_1, dy_1 / ds_1
    x_1, y_1 = _cubic_bezier(1.0, xs0, xs1, xs2, xs3), _cubic_bezier(1.0, ys0, ys1, ys2, ys3)
    
    mask_after = (s_t > total_L)
    s_excess = s_t[mask_after] - total_L
    x[mask_after] = x_1 + ux_1 * s_excess
    y[mask_after] = y_1 + uy_1 * s_excess
    
    # Velocity raw is tangent * v_t
    vx_raw = np.zeros_like(t)
    vy_raw = np.zeros_like(t)
    
    vx_raw[valid_mask] = (dx_dtau_actual[valid_mask] / ds_dtau_actual[valid_mask]) * v_t[valid_mask]
    vy_raw[valid_mask] = (dy_dtau_actual[valid_mask] / ds_dtau_actual[valid_mask]) * v_t[valid_mask]
    
    vx_raw[mask_before] = ux_0 * v_t[mask_before]
    vy_raw[mask_before] = uy_0 * v_t[mask_before]
    
    vx_raw[mask_after] = ux_1 * v_t[mask_after]
    vy_raw[mask_after] = uy_1 * v_t[mask_after]

    # Rotate path if angle is non-zero
    if angle_deg:
        x, y = _rotate_point_xy(x, y, angle_deg)
        vx_raw, vy_raw = _rotate_vector_xy(vx_raw, vy_raw, angle_deg)

    # Distance to observer
    r = np.sqrt(x**2 + y**2)

    # Use true distance for Doppler geometry, with small epsilon
    eps = 1e-9
    r_safe = np.maximum(r, eps)

    # Radial velocity v_r = (v · r_hat) = (v · p) / |p|
    v_dot_r = vx_raw * x + vy_raw * y
    v_r = v_dot_r / r_safe

    # Clamp radial velocity to keep Doppler ratios realistic,
    # similar behaviour to straight-line (no insane sweeps).
    max_vr = min(0.9 * c_sound, 1.2 * abs(speed_mps))
    v_r = np.clip(v_r, -max_vr, max_vr)

    # Doppler ratio
    freq_ratios = c_sound / (c_sound + v_r)

    # Combined amplitude with master gain and gamma compression for audibility
    r_ref = 10.0
    spatial_amp = r_ref / np.sqrt(r**2 + NEAR_FIELD_RADIUS**2)
    convective_amp = (c_sound / (c_sound + v_r))**1.0
    amplitudes = (spatial_amp * convective_amp)**0.7
    
    return freq_ratios.astype(np.float32), amplitudes.astype(np.float32)
