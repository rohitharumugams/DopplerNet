# bezier.py

import numpy as np
from audio.audio_utils import SR, apply_distance_fade

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
                             duration_s, c_sound=343.0, angle_deg=0.0):
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

    # Solve for spatial scale to match speed_mps
    # v_actual = (1/T) * dB/dtau
    vx_init = dx_dtau / T
    vy_init = dy_dtau / T
    speed_init = np.sqrt(vx_init**2 + vy_init**2)
    mean_speed_init = np.mean(speed_init) if speed_init.size > 0 else 1.0
    phys_scale = speed_mps / mean_speed_init

    # Re-calculate position and velocity with scaled control points
    # (This ensures x(t) and v(t) are kinematically consistent)
    x = _cubic_bezier(tau, x0*phys_scale, x1*phys_scale, x2*phys_scale, x3*phys_scale)
    y = _cubic_bezier(tau, y0*phys_scale, y1*phys_scale, y2*phys_scale, y3*phys_scale)
    dx_dtau = _cubic_bezier_derivative(tau, x0*phys_scale, x1*phys_scale, x2*phys_scale, x3*phys_scale)
    dy_dtau = _cubic_bezier_derivative(tau, y0*phys_scale, y1*phys_scale, y2*phys_scale, y3*phys_scale)

    vx_raw = dx_dtau / T
    vy_raw = dy_dtau / T

    # Rotate path if angle is non-zero
    if angle_deg != 0:
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = x * cos_t - y * sin_t
        y_rot = x * sin_t + y * cos_t
        vx_rot = vx_raw * cos_t - vy_raw * sin_t
        vy_rot = vx_raw * sin_t + vy_raw * cos_t
        x, y, vx_raw, vy_raw = x_rot, y_rot, vx_rot, vy_rot

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
    spatial_amp = 1.0 / np.sqrt(r**2 + NEAR_FIELD_RADIUS**2)
    convective_amp = (c_sound / (c_sound + v_r))**2
    amplitudes = (10.0 * spatial_amp * convective_amp)**0.7
    
    # Smooth fade-in/out to prevent abrupt spawning
    amplitudes = apply_distance_fade(amplitudes, fade_duration_s=1.0)
    
    return freq_ratios.astype(np.float32), amplitudes.astype(np.float32)


def calculate_bezier_doppler_dual(speed_mps, x0, x1, x2, x3, y0, y1, y2, y3, 
                                   duration_s, mic_separation_m, c_sound=343.0, angle_deg=0.0):
    """
    Cubic Bezier path with dual microphones placed symmetrically along X-axis.
    
    Microphones are placed at:
        M1 at (-mic_separation_m/2, 0)
        M2 at (+mic_separation_m/2, 0)
    
    Parameters
    ----------
    speed_mps : float
        Desired average speed along the Bezier curve (m/s).
    x0..x3, y0..y3 : float
        Control points for the cubic Bezier in meters.
    duration_s : float
        Total duration (seconds).
    mic_separation_m : float
        Distance between the two microphones (meters).
    
    Returns
    -------
    dict with keys:
        'm1': dict with 'freq_ratios', 'amplitudes', 'distances', 'velocities'
        'm2': dict with 'freq_ratios', 'amplitudes', 'distances', 'velocities'
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
    
    # Convert derivative from param-space to physical time
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
    
    # Rotate path if angle is non-zero
    if angle_deg != 0:
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = x * cos_t - y * sin_t
        y_rot = x * sin_t + y * cos_t
        vx_rot = vx * cos_t - vy * sin_t
        vy_rot = vx * sin_t + vy * cos_t
        x, y, vx, vy = x_rot, y_rot, vx_rot, vy_rot

    # Microphone positions
    half_sep = mic_separation_m / 2.0
    mic1_pos = np.array([-half_sep, 0.0])  # M1
    mic2_pos = np.array([half_sep, 0.0])   # M2
    
    eps = 1e-9
    
    # Calculate for M1
    r1 = np.sqrt((x - mic1_pos[0])**2 + (y - mic1_pos[1])**2)
    max_vr = min(0.9 * c_sound, 1.2 * abs(speed_mps))
    v_r1 = np.clip((vx * (x - mic1_pos[0]) + vy * (y - mic1_pos[1])) / np.maximum(r1, 1e-9), -max_vr, max_vr)
    freq_ratios_m1 = c_sound / (c_sound + v_r1)
    amplitudes_m1 = (10.0 * (1.0 / np.sqrt(r1**2 + NEAR_FIELD_RADIUS**2)) * (c_sound / (c_sound + v_r1))**2)**0.7
    
    # Calculate for M2
    r2 = np.sqrt((x - mic2_pos[0])**2 + (y - mic2_pos[1])**2)
    v_r2 = np.clip((vx * (x - mic2_pos[0]) + vy * (y - mic2_pos[1])) / np.maximum(r2, 1e-9), -max_vr, max_vr)
    freq_ratios_m2 = c_sound / (c_sound + v_r2)
    amplitudes_m2 = (10.0 * (1.0 / np.sqrt(r2**2 + NEAR_FIELD_RADIUS**2)) * (c_sound / (c_sound + v_r2))**2)**0.7
    
    # Smooth fade-in/out to prevent abrupt spawning
    amplitudes_m1 = apply_distance_fade(amplitudes_m1, fade_duration_s=1.0)
    amplitudes_m2 = apply_distance_fade(amplitudes_m2, fade_duration_s=1.0)
    
    return {
        'm1': {
            'freq_ratios': freq_ratios_m1.astype(np.float32),
            'amplitudes': amplitudes_m1.astype(np.float32),
            'distances': r1.astype(np.float32),
            'velocities': v_r1.astype(np.float32)
        },
        'm2': {
            'freq_ratios': freq_ratios_m2.astype(np.float32),
            'amplitudes': amplitudes_m2.astype(np.float32),
            'distances': r2.astype(np.float32),
            'velocities': v_r2.astype(np.float32)
        }
    }

