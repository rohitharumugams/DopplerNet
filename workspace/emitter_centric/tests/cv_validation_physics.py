"""Physics and artifact helpers for CV validation."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from workspace.emitter_centric.config import SR
from workspace.emitter_centric.source_frame import pitch_factor_linear, speed_profile

# Keys that must match between emitter / observer sample_metadata parameters.
SHARED_PARAM_KEYS = (
    "speed",
    "distance",
    "angle",
    "duration",
    "acceleration",
    "temperature",
    "humidity",
    "path_type",
    "h",
)


def spectral_centroid_hz(y: np.ndarray, sr: int = SR) -> float:
    """Simple FFT centroid (Hz) for source-vs-emitter comparison."""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = min(len(y), sr * 10)
    if n < 256:
        n = min(len(y), 4096)
    if n < 64:
        return 0.0
    seg = y[:n]
    window = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
    denom = float(np.sum(spec)) + 1e-12
    return float(np.sum(freqs * spec) / denom)


def load_vehicle_source_buffer(vehicle: str, duration_s: float) -> np.ndarray:
    from workspace.emitter_centric.synthesis import load_source_for_emitter

    return load_source_for_emitter(
        audio_path=None,
        vehicle_name=vehicle,
        duration_s=float(duration_s),
        sr=SR,
    )


def recompute_src_pitch_curve(
    *,
    n_samples: int,
    duration_s: float,
    speed_mps: float,
    acceleration: float,
    v_ref: float | None = None,
) -> np.ndarray:
    """Match batch synthesis: v_ref defaults to speed when not provided."""
    _, v = speed_profile(n_samples, duration_s, speed_mps, acceleration, sr=SR)
    if v_ref is None:
        v_ref = max(float(speed_mps), 1.0)
    return pitch_factor_linear(v, v_ref)


def expected_straight_freq_ratios(
    params: dict[str, Any],
    n_samples: int,
) -> np.ndarray:
    """Recompute observer geometric Doppler ratio on uniform sample grid."""
    from audio.audio_utils import get_speed_of_sound
    from physics.recording_labels import resolve_cpa_time_s
    from workspace.emitter_centric.kinematics import straight_cv_kinematics_with_c

    duration = float(params["duration"])
    temp = float(params.get("temperature", 20.0))
    hum = float(params.get("humidity", 50.0))
    c = float(get_speed_of_sound(temp, hum))
    cpa_t = resolve_cpa_time_s(params, duration)
    kin = straight_cv_kinematics_with_c(
        float(params["speed"]),
        float(params.get("distance", params.get("h", 15.0))),
        float(params.get("angle", 0.0)),
        duration,
        n_samples,
        c_sound=c,
        accel_mps2=float(params.get("acceleration", 0.0)),
        cpa_time_s=cpa_t,
    )
    return np.asarray(kin["freq_ratio"], dtype=np.float64)


def analytic_doppler_bounds(speed_mps: float, temp_c: float, humidity: float) -> tuple[float, float]:
    """Loose extrema c/(c+v) and c/(c-v) for |v_r| <= v pass-by."""
    from audio.audio_utils import get_speed_of_sound

    c = float(get_speed_of_sound(temp_c, humidity))
    v = float(speed_mps)
    if v >= c * 0.95:
        return c / (c + v), c / (c + v)
    return c / (c + v), c / (c - v)


def align_series(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return np.asarray(a[:n], dtype=np.float64), np.asarray(b[:n], dtype=np.float64)


def freq_ratio_time_axis(duration_s: float, n_samples: int) -> np.ndarray:
    """Uniform time axis for observer freq_ratios (one ratio per audio sample)."""
    n = int(n_samples)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(0.0, float(duration_s), n, dtype=np.float64)


def metadata_parameters_match(emit_meta: dict, obs_meta: dict) -> list[str]:
    """Return list of mismatch descriptions."""
    ep = emit_meta.get("parameters") or {}
    op = obs_meta.get("parameters") or {}
    issues: list[str] = []
    for key in SHARED_PARAM_KEYS:
        if key in ep or key in op:
            ev, ov = ep.get(key), op.get(key)
            if ev is None or ov is None:
                if ev != ov:
                    issues.append(f"{key}: emit={ev!r} obs={ov!r}")
                continue
            if isinstance(ev, (int, float)) and isinstance(ov, (int, float)):
                if abs(float(ev) - float(ov)) > 1e-5:
                    issues.append(f"{key}: {ev} vs {ov}")
            elif ev != ov:
                issues.append(f"{key}: {ev!r} vs {ov!r}")
    if ep.get("vehicle") != op.get("vehicle") and ("vehicle" in ep or "vehicle" in op):
        if ep.get("vehicle") != op.get("vehicle"):
            issues.append(f"vehicle: {ep.get('vehicle')} vs {op.get('vehicle')}")
    return issues


def scan_sidecars_finite(common_dir: str) -> list[str]:
    """List .npy paths under common_dir containing non-finite values."""
    bad: list[str] = []
    if not os.path.isdir(common_dir):
        return [f"missing dir {common_dir}"]
    for name in os.listdir(common_dir):
        if not name.endswith(".npy"):
            continue
        path = os.path.join(common_dir, name)
        arr = np.load(path)
        if not np.all(np.isfinite(arr)):
            bad.append(name)
    return bad


def estimate_cpa_time_from_freq_ratio(
    freq_ratios: np.ndarray,
    time_s: np.ndarray,
) -> float:
    """Time where freq_ratio is closest to 1 (pass-by nominal)."""
    fr, t = align_series(freq_ratios, time_s)
    if len(fr) < 1:
        return 0.0
    return float(t[int(np.argmin(np.abs(fr - 1.0)))])


def synthesizer_deterministic_probe(spec: dict, *, enable_rpm: bool = True) -> bool:
    """Run co-moving synthesis twice; True if outputs match bit-for-bit."""
    from workspace.emitter_centric.source_frame import synthesize_source_frame_audio

    src = load_vehicle_source_buffer(spec["vehicle"], float(spec["duration"]))
    kwargs = dict(
        duration_s=float(spec["duration"]),
        speed_mps=float(spec["speed"]),
        acceleration=float(spec.get("acceleration", 0.0)),
        enable_rpm_coupling=enable_rpm,
    )
    y1, _ = synthesize_source_frame_audio(src, **kwargs)
    y2, _ = synthesize_source_frame_audio(src, **kwargs)
    return bool(np.array_equal(y1, y2))
