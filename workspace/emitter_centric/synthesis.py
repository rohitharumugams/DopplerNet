"""Emitter-centric synthesis (workspace sandbox only)."""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np

from audio.audio_utils import (
    apply_true_doppler_shift,
    extend_audio_with_overlap,
    get_speed_of_sound,
    load_audio,
    write_soundfile,
)
from physics.straight_line import calculate_straight_line_doppler
from workspace.emitter_centric import config
from workspace.emitter_centric.kinematics import straight_cv_kinematics_with_c
from scipy.interpolate import interp1d

from workspace.emitter_centric.observer_deposit import forward_point_deposit
from workspace.emitter_centric.retarded_time import (
    check_monotonicity,
    deposit_jacobian_amplitude,
    forward_retarded_time,
)


def load_source_for_emitter(
    *,
    audio_path: str | None,
    vehicle_name: str | None,
    duration_s: float,
    sr: int = config.SR,
) -> np.ndarray:
    """Load vehicle library clip or user WAV (mirrors workspace distance panel)."""
    if audio_path and os.path.isfile(audio_path):
        y, _ = load_audio(audio_path, sr=sr, mono=True)
    elif vehicle_name:
        from core.config import DRONE_SOUNDS_FOLDER, UPLOAD_FOLDER

        vehicle_file = None
        for folder in (UPLOAD_FOLDER, DRONE_SOUNDS_FOLDER):
            for ext in (".wav", ".mp3", ".ogg", ".flac"):
                test = os.path.join(folder, f"{vehicle_name}{ext}")
                if os.path.isfile(test):
                    vehicle_file = test
                    break
            if vehicle_file:
                break
        if not vehicle_file:
            raise FileNotFoundError(f"Vehicle not found: {vehicle_name}")
        y, _ = load_audio(vehicle_file, sr=sr, mono=True)
    else:
        raise ValueError("Provide audio_path or vehicle_name")

    y = extend_audio_with_overlap(y.astype(np.float32), duration_s * 2.0, sr)
    n = int(round(duration_s * sr))
    if len(y) >= n:
        return y[:n].astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    out[: len(y)] = y
    return out


def synthesize_observer_centric(
    source: np.ndarray,
    speed_mps: float,
    distance_m: float,
    duration_s: float,
    *,
    angle_deg: float = 0.0,
    temp_c: float = config.DEFAULT_TEMP_C,
    humidity: float = config.DEFAULT_HUMIDITY,
    cpa_time_s: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Production formulation 2 (read-only reuse of audio_utils)."""
    c = get_speed_of_sound(temp_c, humidity)
    if cpa_time_s is None:
        cpa_time_s = duration_s / 2.0
    n = int(round(config.SR * duration_s))
    freq, amp = calculate_straight_line_doppler(
        speed_mps,
        distance_m,
        angle_deg,
        duration_s,
        c_sound=c,
        cpa_time_s=cpa_time_s,
    )
    if len(source) < n:
        source = np.pad(source, (0, n - len(source)))
    y = apply_true_doppler_shift(source[: max(len(source), n)], freq, amp)
    meta = {
        "formulation": "observer_centric",
        "c_sound": c,
        "n_samples": n,
    }
    return y.astype(np.float32), meta


def synthesize_emitter_uniform_observer(
    source: np.ndarray,
    freq_ratio: np.ndarray,
    amplitude: np.ndarray,
) -> np.ndarray:
    """
    Emitter-centric dual on a uniform observer grid: integrated emission phase
    maps to observer sample k (equivalent to apply_true_doppler_shift).
    """
    n = len(freq_ratio)
    src = np.asarray(source, dtype=np.float64)
    if len(src) < n:
        src = np.pad(src, (0, n - len(src)))
    alpha = np.asarray(freq_ratio, dtype=np.float64)
    amp = np.asarray(amplitude, dtype=np.float64)
    input_positions = np.concatenate(([0.0], np.cumsum(alpha[1:], dtype=np.float64)))
    if input_positions[-1] > 0 and input_positions[-1] > len(src) - 1:
        input_positions = input_positions * ((len(src) - 1) / input_positions[-1])
    input_positions = np.clip(input_positions, 0, len(src) - 1)
    resampler = interp1d(
        np.arange(len(src)),
        src,
        kind="cubic",
        bounds_error=False,
        fill_value=(src[0], src[-1]),
    )
    return (resampler(input_positions) * amp).astype(np.float32)


def synthesize_emitter_forward(
    source: np.ndarray,
    speed_mps: float,
    distance_m: float,
    duration_s: float,
    *,
    angle_deg: float = 0.0,
    temp_c: float = config.DEFAULT_TEMP_C,
    humidity: float = config.DEFAULT_HUMIDITY,
    cpa_time_s: float | None = None,
    use_propagation_delay: bool = False,
    apply_jacobian: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Emitter-centric forward model: uniform emission grid, deposit to observer timeline.

    Default (parity mode): kinematic cumulative t_o without R/c; Jacobian on amplitude.
    """
    c = get_speed_of_sound(temp_c, humidity)
    if cpa_time_s is None:
        cpa_time_s = duration_s / 2.0
    n = int(round(config.SR * duration_s))
    kin = straight_cv_kinematics_with_c(
        speed_mps,
        distance_m,
        angle_deg,
        duration_s,
        n,
        c_sound=c,
        cpa_time_s=cpa_time_s,
    )
    check_monotonicity(kin["freq_ratio"], c)

    s_e = source[:n].astype(np.float32)
    if len(s_e) < n:
        s_e = np.pad(s_e, (0, n - len(s_e)))

    if use_propagation_delay:
        t_e = np.arange(n, dtype=np.float64) / config.SR
        t_o = forward_retarded_time(t_e, kin["r"], c, use_propagation_delay=True)
        weights = (
            deposit_jacobian_amplitude(kin["amplitude"], kin["freq_ratio"])
            if apply_jacobian
            else kin["amplitude"].astype(np.float32)
        )
        w = np.asarray(weights, dtype=np.float32)
        y_buf, _ = forward_point_deposit(
            s_e,
            t_o,
            w,
            config.SR,
            buffer_duration_s=duration_s + float(np.max(t_o)) + config.R_MAX_M / config.C_MIN_MPS,
        )
        t_uni = np.arange(n, dtype=np.float64) / config.SR
        t_axis = np.arange(len(y_buf), dtype=np.float64) / config.SR
        y = np.interp(t_uni, t_axis, y_buf.astype(np.float64)).astype(np.float32)
    else:
        # Parity mode: uniform observer clock, emission-phase integral (dual to formulation 2).
        y = synthesize_emitter_uniform_observer(
            s_e, kin["freq_ratio"], kin["amplitude"]
        )

    meta = {
        "formulation": "emitter_forward",
        "c_sound": c,
        "use_propagation_delay": use_propagation_delay,
        "apply_jacobian": apply_jacobian,
        "n_samples": n,
    }
    return y.astype(np.float32), meta


def run_straight_cv_job(
    *,
    speed_mps: float,
    distance_m: float,
    duration_s: float,
    angle_deg: float = 0.0,
    temp_c: float = config.DEFAULT_TEMP_C,
    humidity: float = config.DEFAULT_HUMIDITY,
    vehicle_name: str | None = "KiaSportage",
    audio_path: str | None = None,
    use_propagation_delay: bool = False,
    out_dir: str | None = None,
    job_name: str | None = None,
) -> dict:
    """
    Single-clip shortcut — uses the same batch layout as the UI (no flat timestamp folders).
    """
    from workspace.emitter_centric.batch_paths import normalize_batch_folder_name
    from workspace.emitter_centric.batch_runner import run_batch

    batch_name = normalize_batch_folder_name(job_name)
    cfg = {
        "total_clips": 1,
        "batch_name": batch_name,
        "save_path": out_dir or config.OUTPUT_ROOT,
        "vehicles": [vehicle_name or "KiaSportage"],
        "path_types": ["straight"],
        "duration_s": duration_s,
        "speed": {"min": speed_mps, "max": speed_mps},
        "distance": {"min": distance_m, "max": distance_m},
        "angle": {"min": angle_deg, "max": angle_deg},
        "temperature": {"min": temp_c, "max": temp_c},
        "humidity": {"min": humidity, "max": humidity},
        "simulation_mode": "cv",
        "compare_observer": True,
        "enable_rpm_coupling": True,
        "output": {"format": "wav", "spectrogram_type": "cqt", "generate_diagnostics": True},
    }
    if use_propagation_delay:
        cfg["use_propagation_delay"] = True
    result = run_batch(cfg)
    result["legacy_note"] = (
        "Flat folders like 20260603_004724 are no longer created; "
        "outputs use batch_*/emitter_centric|observer_centric|comparison_outputs/."
    )
    return result
