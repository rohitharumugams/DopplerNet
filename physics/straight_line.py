import numpy as np
from audio.audio_utils import SR, apply_distance_fade

C_SOUND_STANDARD = 343.0  # m/s
NEAR_FIELD_RADIUS = 2.0  # m – effective source size for 1/R damping


def calculate_straight_line_doppler(speed_mps, min_distance_m, angle_deg, duration_s, c_sound=343.0):
    """
    Straight-line pass-by with angle and near-field-safe amplitude.
    (Constant velocity variant)
    """
    return calculate_straight_line_accelerated_doppler(speed_mps, 0.0, min_distance_m, angle_deg, duration_s, c_sound)


def calculate_straight_line_accelerated_doppler(speed_v0_mps, accel_mps2, min_distance_m, angle_deg, duration_s, c_sound=343.0):
    """
    Straight-line pass-by with constant acceleration (B7).
    
    Parameters
    ----------
    speed_v0_mps : float
        Initial speed at t=0 (m/s).
    accel_mps2 : float
        Constant acceleration (m/s^2).
    min_distance_m : float
        Closest distance from path to observer (meters).
    angle_deg : float
        Direction of motion angle.
    duration_s : float
        Total duration (seconds).
    """
    num_samples = int(round(SR * duration_s))
    t = np.linspace(0.0, duration_s, num_samples, endpoint=False)
    t0 = duration_s / 2.0
    dt = t - t0

    # Instantaneous speed: v(t) = v0 + a * t
    # However, let's define v0 as the speed AT t=t0 (CPA) for consistency with min_distance
    v_t = speed_v0_mps + accel_mps2 * dt
    
    # Position: p(t) = p_c + integral of v(t) * u
    # p(t) = p_c + u * (v0 * dt + 0.5 * a * dt^2)
    theta = np.deg2rad(angle_deg)
    u = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-np.sin(theta), np.cos(theta)])
    
    p_c = min_distance_m * n
    # Displacement along path relative to p_c (t=t0)
    s_t = speed_v0_mps * dt + 0.5 * accel_mps2 * dt**2
    
    p = p_c[:, None] + u[:, None] * s_t[None, :]
    r = np.linalg.norm(p, axis=0)
    r_safe = np.maximum(r, 1e-9)
    
    # Velocity vector at each time t
    v_vec = u[:, None] * v_t[None, :]
    # Radial velocity v_r = (v_vec · p) / |p|
    v_r = np.sum(v_vec * p, axis=0) / r_safe
    
    freq_ratios = c_sound / (c_sound + v_r)
    spatial_amp = 1.0 / np.sqrt(r**2 + NEAR_FIELD_RADIUS**2)
    convective_amp = (c_sound / (c_sound + v_r))**2
    amplitudes = (10.0 * spatial_amp * convective_amp)**0.7
    
    # Smooth fade-in/out to prevent abrupt spawning
    amplitudes = apply_distance_fade(amplitudes, fade_duration_s=1.0)
    
    return freq_ratios.astype(np.float32), amplitudes.astype(np.float32)


def calculate_straight_line_doppler_dual(speed_mps, min_distance_m, angle_deg, duration_s, mic_separation_m, c_sound=343.0):
    """
    Straight-line pass-by with dual microphones placed symmetrically along X-axis.
    
    Parameters
    ----------
    speed_mps : float
        Constant speed of the source (m/s).
    min_distance_m : float
        Closest distance from path to the origin (meters).
    angle_deg : float
        Direction of motion angle in degrees (w.r.t. +x axis).
    duration_s : float
        Total duration (seconds).
    mic_separation_m : float
        Distance between the two microphones (meters).
        M1 will be at (-mic_separation_m/2, 0)
        M2 will be at (+mic_separation_m/2, 0)
    
    Returns
    -------
    dict with keys:
        'm1': dict with 'freq_ratios', 'amplitudes', 'distances', 'velocities'
        'm2': dict with 'freq_ratios', 'amplitudes', 'distances', 'velocities'
    """
    # Number of samples and time axis
    num_samples = int(round(SR * duration_s))
    t = np.linspace(0.0, duration_s, num_samples, endpoint=False)
    
    # Time of closest approach to origin
    t0 = duration_s / 2.0
    dt = t - t0
    
    # Direction unit vector for motion
    theta = np.deg2rad(angle_deg)
    u = np.array([np.cos(theta), np.sin(theta)])
    
    # Perpendicular unit vector
    n = np.array([-np.sin(theta), np.cos(theta)])
    
    # Point of closest approach to origin
    p_c = min_distance_m * n
    
    # Velocity vector
    v_vec = u * speed_mps
    
    # Position as function of time: p(t) = p_c + u * v * dt
    p = p_c[:, None] + v_vec[:, None] * dt[None, :]  # shape (2, N)
    
    # Microphone positions (fixed)
    half_sep = mic_separation_m / 2.0
    mic1_pos = np.array([-half_sep, 0.0])  # M1 at (-D/2, 0)
    mic2_pos = np.array([half_sep, 0.0])   # M2 at (+D/2, 0)
    
    # Calculate for M1
    r1_vec = p - mic1_pos[:, None]
    r1 = np.linalg.norm(r1_vec, axis=0)
    eps = 1e-9
    r1_safe = np.maximum(r1, eps)
    v_r1 = np.dot(v_vec, r1_vec) / r1_safe
    
    freq_ratios_m1 = c_sound / (c_sound + v_r1)
    spatial_amp_m1 = 1.0 / np.sqrt(r1**2 + NEAR_FIELD_RADIUS**2)
    convective_amp_m1 = (c_sound / (c_sound + v_r1))**2
    amplitudes_m1 = (10.0 * spatial_amp_m1 * convective_amp_m1)**0.7
    
    # Calculate for M2
    r2_vec = p - mic2_pos[:, None]
    r2 = np.linalg.norm(r2_vec, axis=0)
    r2_safe = np.maximum(r2, eps)
    v_r2 = np.dot(v_vec, r2_vec) / r2_safe
    
    freq_ratios_m2 = c_sound / (c_sound + v_r2)
    spatial_amp_m2 = 1.0 / np.sqrt(r2**2 + NEAR_FIELD_RADIUS**2)
    convective_amp_m2 = (c_sound / (c_sound + v_r2))**2
    amplitudes_m2 = (10.0 * spatial_amp_m2 * convective_amp_m2)**0.7
    
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

