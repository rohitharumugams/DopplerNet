"""Co-moving source-frame audio (no geometric Doppler)."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from audio.audio_utils import SR


def speed_profile(
    n_samples: int,
    duration_s: float,
    speed_mps: float,
    acceleration: float,
    sr: int = SR,
) -> tuple[np.ndarray, np.ndarray]:
    """Along-track speed v(t) and time axis (source frame clock)."""
    t = np.linspace(0.0, float(duration_s), int(n_samples), endpoint=False, dtype=np.float64)
    v = float(speed_mps) + float(acceleration) * t
    return t.astype(np.float32), np.maximum(v, 0.1).astype(np.float32)


def pitch_factor_linear(v: np.ndarray, v_ref: float) -> np.ndarray:
    """Simple RPM proxy: f_emit ∝ v/v_ref."""
    v_ref = max(float(v_ref), 0.1)
    return (np.asarray(v, dtype=np.float64) / v_ref).astype(np.float32)


def synthesize_source_frame_audio(
    source: np.ndarray,
    *,
    duration_s: float,
    speed_mps: float,
    acceleration: float = 0.0,
    v_ref: float | None = None,
    enable_rpm_coupling: bool = True,
    level_from_speed: bool = True,
    sr: int = SR,
) -> tuple[np.ndarray, dict]:
    """
    Co-moving microphone: resample source by intrinsic pitch vs speed; no geometric Doppler.
    """
    n = int(round(duration_s * sr))
    src = np.asarray(source, dtype=np.float64)
    if len(src) < n:
        src = np.pad(src, (0, n - len(src)))
    src = src[:n]

    t, v = speed_profile(n, duration_s, speed_mps, acceleration, sr=sr)
    if v_ref is None:
        v_ref = max(float(speed_mps), 1.0)

    if enable_rpm_coupling:
        alpha = pitch_factor_linear(v, v_ref)
    else:
        alpha = np.ones(n, dtype=np.float32)

    input_positions = np.concatenate(([0.0], np.cumsum(alpha[1:], dtype=np.float64)))
    if input_positions[-1] > 0 and input_positions[-1] > n - 1:
        input_positions *= (n - 1) / input_positions[-1]
    input_positions = np.clip(input_positions, 0, n - 1)
    resampler = interp1d(
        np.arange(n, dtype=np.float64),
        src,
        kind="cubic",
        bounds_error=False,
        fill_value=(src[0], src[-1]),
    )
    y = resampler(input_positions).astype(np.float32)

    if level_from_speed:
        gain = (v / max(v_ref, 0.1)) ** 0.35
        y = (y * gain.astype(np.float32)).astype(np.float32)

    peak = float(np.max(np.abs(y)) + 1e-9)
    if peak > 1.0:
        y = (y / peak * 0.95).astype(np.float32)

    meta = {
        "frame": "co_moving_source",
        "geometric_doppler": False,
        "rpm_coupling": bool(enable_rpm_coupling),
        "v_ref_mps": float(v_ref),
        "speed_mps": float(speed_mps),
        "acceleration_mps2": float(acceleration),
    }
    return y, meta
