# parabola.py

import numpy as np
from audio_utils import SR

# Speed of sound (m/s)
C_SOUND = 343.0

# Minimum effective distance to avoid near-field blowups (meters)
NEAR_FIELD_RADIUS = 2.0


def calculate_parabola_doppler(speed_mps, a, h, duration_s, n_steps=4096):
    """
    Compute Doppler frequency ratios and amplitudes for a parabolic path.

    Path model (observer at origin):
        x(τ) = L * τ,   τ ∈ [-1, 1]
        y(τ) = a * x(τ)^2 + h

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

    n_steps = int(n_steps)
    if n_steps < 4:
        n_steps = 4

    T = float(duration_s)
    if T <= 0:
        T = 1.0

    # Normalized parameter τ ∈ [-1, 1]
    tau = np.linspace(-1.0, 1.0, n_steps)

    # Horizontal half-span: how far (in x) the vehicle travels over the clip.
    # Use nominal straight-line distance: D ≈ speed * duration, half on each side.
    half_span_x = 0.5 * speed_mps * T

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

    vx = vx_raw * scale
    vy = vy_raw * scale

    # Distance to observer
    r = np.sqrt(x**2 + y**2)

    # Use true distance for Doppler geometry, but avoid division by zero
    eps = 1e-9
    r_safe = np.maximum(r, eps)

    # Radial velocity: v_r = (v · r_hat) = (v · p) / |p|
    v_dot_r = vx * x + vy * y
    v_r = v_dot_r / r_safe

    # Clamp radial velocity to avoid unrealistic/supersonic Doppler
    max_vr = min(0.9 * C_SOUND, 1.2 * abs(speed_mps))
    v_r = np.clip(v_r, -max_vr, max_vr)

    # Doppler frequency ratio f'/f0 = c / (c - v_r)
    freq_ratios = C_SOUND / (C_SOUND - v_r)

    # Near-field-safe amplitude, similar to straight-line / Bezier
    r_eff = np.sqrt(r**2 + NEAR_FIELD_RADIUS**2)
    amplitudes = 1.0 / r_eff

    # Normalize amplitude to max 1, optional mild compression
    max_amp = np.max(np.abs(amplitudes)) if amplitudes.size > 0 else 1.0
    if max_amp > 0:
        amplitudes = amplitudes / max_amp

    # Mild gamma compression to avoid extreme loud/quiet differences
    amplitudes = amplitudes**0.7

    return freq_ratios.astype(np.float32), amplitudes.astype(np.float32)
