"""
Workspace-only pass-by synthesis for analysis-by-synthesis grids.

Uses DopplerSim straight-line physics + time-domain warp (no changes to main batch pipeline).
"""

from __future__ import annotations

import numpy as np
import librosa

from audio.audio_utils import (
    apply_doppler_to_audio_fixed,
    extend_audio_with_overlap,
    get_speed_of_sound,
    SR as SIM_SR,
)
from physics.straight_line import calculate_straight_line_doppler


def resample_by_inverse_doppler(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """
    Apply constant inverse-Doppler resampling: f_emit = f_obs * ratio.

    ratio = (c + v_r) / c  (inverse of emission factor c/(c+v_r)).
    """
    ratio = float(max(0.25, min(4.0, ratio)))
    n = len(audio)
    if n < 4:
        return audio.astype(np.float32)
    src_pos = np.arange(n, dtype=np.float64) * ratio
    src_pos -= src_pos[0]
    if src_pos[-1] > (n - 1):
        src_pos *= (n - 1) / max(src_pos[-1], 1e-9)
    x_idx = np.arange(n, dtype=np.float64)
    return np.interp(src_pos, x_idx, np.asarray(audio, dtype=np.float64)).astype(np.float32)


def dedopplerize_far_field_segment(
    segment: np.ndarray,
    sr: int,
    speed_mps: float,
    *,
    temp_c: float = 20.0,
    humidity: float = 50.0,
) -> np.ndarray:
    """
    Far-field CV inverse Doppler for candidate speed v.

    Assumes the 1 s segment was recorded with negligible radial velocity (Doppler ≈ 0),
    then applies the inverse scale for a candidate pass-by speed v as in the prof grid procedure.
    """
    c = get_speed_of_sound(temp_c, humidity)
    v_r = float(speed_mps)  # simplified far-field radial component
    ratio = (c + v_r) / max(c, 1e-6)
    return resample_by_inverse_doppler(segment, sr, ratio)


def stitch_repeat_segment(segment: np.ndarray, target_len: int) -> np.ndarray:
    """Repeat segment until target_len (prof stitch; may introduce periodic copies)."""
    seg = np.asarray(segment, dtype=np.float32)
    if len(seg) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(seg) >= target_len:
        return seg[:target_len].copy()
    reps = int(np.ceil(target_len / len(seg)))
    out = np.tile(seg, reps)[:target_len]
    return out.astype(np.float32)


def frame_rms_envelope(y: np.ndarray, sr: int, hop_length: int) -> tuple[np.ndarray, np.ndarray]:
    rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=hop_length, center=True)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return times.astype(np.float32), rms.astype(np.float32)


def find_peak_time(y: np.ndarray, sr: int, hop_length: int = 64) -> float:
    times, rms = frame_rms_envelope(y, sr, hop_length)
    if len(rms) == 0:
        return 0.5 * len(y) / sr
    return float(times[int(np.argmax(rms))])


def align_to_peak(y: np.ndarray, sr: int, target_peak_s: float, hop_length: int = 64) -> np.ndarray:
    """Circular shift so RMS peak aligns with target_peak_s."""
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    if n < 2:
        return y
    peak_s = find_peak_time(y, sr, hop_length)
    shift_s = target_peak_s - peak_s
    shift_samples = int(round(shift_s * sr))
    if shift_samples == 0:
        return y
    return np.roll(y, shift_samples).astype(np.float32)


def synthesize_passby_straight(
    source_audio: np.ndarray,
    sr: int,
    speed_mps: float,
    distance_m: float,
    duration_s: float,
    *,
    cpa_time_s: float | None = None,
    angle_deg: float = 0.0,
    temp_c: float = 20.0,
    humidity: float = 50.0,
    target_peak_s: float | None = None,
) -> np.ndarray:
    """
    Synthesize straight-line pass-by using stitched source + DopplerSim physics.
    """
    c = get_speed_of_sound(temp_c, humidity)
    dur = float(duration_s)
    if cpa_time_s is None:
        cpa_time_s = dur / 2.0

    # Buffer for upward Doppler consumption
    audio_buf = extend_audio_with_overlap(source_audio, dur * 2.0, sr)
    freq_ratios, amplitudes = calculate_straight_line_doppler(
        float(speed_mps),
        float(distance_m),
        float(angle_deg),
        dur,
        c_sound=c,
        cpa_time_s=float(cpa_time_s),
    )
    target_n = int(round(sr * dur))
    # Physics modules use SIM_SR (22.05 kHz) for sample count; resample curves to analysis sr.
    if len(freq_ratios) != target_n:
        x_src = np.linspace(0.0, 1.0, len(freq_ratios))
        x_dst = np.linspace(0.0, 1.0, target_n)
        freq_ratios = np.interp(x_dst, x_src, np.asarray(freq_ratios, dtype=np.float64)).astype(np.float32)
        amplitudes = np.interp(x_dst, x_src, np.asarray(amplitudes, dtype=np.float64)).astype(np.float32)
    out = apply_doppler_to_audio_fixed(audio_buf, freq_ratios, amplitudes)
    if len(out) > target_n:
        out = out[:target_n]
    elif len(out) < target_n:
        padded = np.zeros(target_n, dtype=np.float32)
        padded[: len(out)] = out
        out = padded

    if target_peak_s is not None:
        out = align_to_peak(out, sr, float(target_peak_s))

    peak = np.max(np.abs(out))
    if peak > 1e-8:
        out = (out / peak).astype(np.float32)
    return out
