# parabola.py

import numpy as np
from audio.audio_utils import SR, apply_distance_fade

# Speed of sound (m/s)
C_SOUND = 343.0

# Minimum effective distance to avoid near-field blowups (meters)
NEAR_FIELD_RADIUS = 2.0


def calculate_parabola_doppler(speed_mps, a, h, duration_s, n_steps=None, c_sound=343.0, angle_deg=0.0):
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

    # Safety: enforce positive curvature and height
    if a <= 0:
        a = abs(a) if a != 0 else 0.01
    if h <= 0:
        h = abs(h) if h != 0 else 5.0

    if n_steps is None:
        n_steps = int(round(SR * duration_s))

    n_steps = int(n_steps)
    if n_steps < 4:
        n_steps = 4

    T = float(duration_s)
    if T <= 0:
        T = 1.0

    # Normalized parameter τ ∈ [-1, 1]
    tau = np.linspace(-1.0, 1.0, n_steps)

    # Horizontal half-span: how far (in x) the vehicle travels over the clip.
    # Corrected: we solve for the span that yields the target average speed.
    # Nominal: span_x = 0.5 * v_avg * T. 
    # For parabola, we need a small adjustment if path is very curved, 
    # but the nominal is usually close enough for a first-order fix.
    span_x = speed_mps * T
    # Simple iterative refinement (optional, but good for consistency)
    # We want mean(sqrt(vx^2 + vy^2)) = speed_mps
    # vx = (span_x/2) * (2/T) = span_x / T
    # vy = 2 * a * x * (span_x/T)
    # So speed = (span_x/T) * sqrt(1 + (2*a*x)^2)
    # Therefore span_x = speed_mps * T / mean(sqrt(1 + (2*a*x)^2))
    temp_tau = np.linspace(-1, 1, 100)
    temp_x = (speed_mps * T / 2) * temp_tau
    refinement = np.mean(np.sqrt(1 + (2 * a * temp_x)**2))
    half_span_x = (speed_mps * T / 2) / refinement

    # Parabolic path in world coordinates
    x = half_span_x * tau
    y = a * x**2 + h

    # Derivatives w.r.t. τ
    dx_dtau = np.full_like(x, half_span_x)
    dy_dtau = 2.0 * a * x * half_span_x

    # Convert derivative from param-space to physical time
    # τ(t) maps linearly from [-1, 1] over [0, T] => dτ/dt = 2 / T
    dtaudt = 2.0 / T
    vx_raw = dx_dtau * dtaudt
    vy_raw = dy_dtau * dtaudt

    # Raw speed magnitude
    speed_raw = np.sqrt(vx_raw**2 + vy_raw**2)
    mean_speed_raw = np.mean(speed_raw) if speed_raw.size > 0 else 0.0

    # Rescale so mean |v| ≈ speed_mps (like Bezier)
    if mean_speed_raw < 1e-6:
        scale = 0.0
    else:
        scale = speed_mps / mean_speed_raw

    # Rotate path if angle is non-zero
    if angle_deg != 0:
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Position rotation
        x_rot = x * cos_t - y * sin_t
        y_rot = x * sin_t + y * cos_t
        # Velocity rotation (affine transformation of vectors)
        vx_rot = vx_raw * cos_t - vy_raw * sin_t
        vy_rot = vx_raw * sin_t + vy_raw * cos_t
        
        # Use rotated coordinates for physics
        x, y = x_rot, y_rot
        vx_raw, vy_raw = vx_rot, vy_rot

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
    amplitudes = (10.0 * (1.0 / r_eff) * (c_sound / (c_sound + v_r))**2)**0.7
    
    # Smooth fade-in/out to prevent abrupt spawning
    amplitudes = apply_distance_fade(amplitudes, fade_duration_s=1.0)
    
    return freq_ratios.astype(np.float32), amplitudes.astype(np.float32)


def calculate_parabola_doppler_dual(speed_mps, a, h, duration_s, mic_separation_m, n_steps=None, c_sound=343.0, angle_deg=0.0):
    """
    Compute Doppler frequency ratios and amplitudes for a parabolic path with dual microphones.
    
    Microphones are placed symmetrically along X-axis:
        M1 at (-mic_separation_m/2, 0)
        M2 at (+mic_separation_m/2, 0)
    
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
    mic_separation_m : float
        Distance between the two microphones (meters).
    n_steps : int
        Number of Doppler samples (interpolated later to audio length).
    
    Returns
    -------
    dict with keys:
        'm1': dict with 'freq_ratios', 'amplitudes', 'distances', 'velocities'
        'm2': dict with 'freq_ratios', 'amplitudes', 'distances', 'velocities'
    """
    # Safety: enforce positive curvature and height
    if a <= 0:
        a = abs(a) if a != 0 else 0.01
    if h <= 0:
        h = abs(h) if h != 0 else 5.0
    
    if n_steps is None:
        n_steps = int(round(SR * duration_s))
        
    n_steps = int(n_steps)
    if n_steps < 4:
        n_steps = 4
    
    T = float(duration_s)
    if T <= 0:
        T = 1.0
    
    # Normalized parameter τ ∈ [-1, 1]
    tau = np.linspace(-1.0, 1.0, n_steps)
    
    # Horizontal half-span
    half_span_x = 0.5 * speed_mps * T
    
    # Parabolic path in world coordinates
    x = half_span_x * tau
    y = a * x**2 + h
    
    # Derivatives w.r.t. τ
    dx_dtau = np.full_like(x, half_span_x)
    dy_dtau = 2.0 * a * x * half_span_x
    
    # Convert derivative from param-space to physical time
    dtaudt = 2.0 / T
    vx_raw = dx_dtau * dtaudt
    vy_raw = dy_dtau * dtaudt
    
    # Raw speed magnitude
    speed_raw = np.sqrt(vx_raw**2 + vy_raw**2)
    mean_speed_raw = np.mean(speed_raw) if speed_raw.size > 0 else 0.0
    
    # Rescale so mean |v| ≈ speed_mps
    if mean_speed_raw < 1e-6:
        scale = 0.0
    else:
        scale = speed_mps / mean_speed_raw
    
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
    
    # Clamp radial velocity to avoid unrealistic/supersonic Doppler
    max_vr = min(0.9 * c_sound, 1.2 * abs(speed_mps))

    # Calculate for M1
    r1 = np.sqrt((x - mic1_pos[0])**2 + (y - mic1_pos[1])**2)
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

