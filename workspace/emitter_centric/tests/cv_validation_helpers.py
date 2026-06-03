"""Shared types and helpers for emitter-centric CV validation scripts."""

from __future__ import annotations

import json
import os
import struct
import wave
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

ResultStatus = Literal["PASS", "FAIL", "INCONCLUSIVE", "SKIP"]

# Thresholds (tune here).
V_SPEED_TOL = 0.02  # m/s
V_SPEED_STD_MAX = 1e-4
PITCH_STD_MAX = 1e-5
PITCH_MEAN_TOL = 1e-4
RAW_CORR_MAX = 0.15
OBS_FREQ_SPAN_MIN = 0.05
STRAIGHT_DOPPLER_PEAK_CORR_MAX = 0.35  # |corr(freq_ratios, emit CQT peak)| ceiling
EMIT_RIDGE_VEHICLE_TOL = 1e-4  # same vehicle @ CV: emit ridge slope stable across scenes
STRAIGHT_RIDGE_SLOPE_RATIO_MIN = 4.0
STRAIGHT_EMIT_RIDGE_SLOPE_MAX = 0.005
STRAIGHT_ARC_LEN_RATIO_MIN = 0.995
STRAIGHT_ARC_LEN_RATIO_MAX = 1.005
CURVED_ARC_LEN_RATIO_MAX = 1.08
EMIT_RMS_CV_MAX_OBS_RATIO = 0.85
DOPPLER_BOUNDS_REL_TOL = 0.06
KINEMATICS_FR_MAX_ABS_ERR = 0.035
SOURCE_EMIT_CENTROID_REL_TOL = 0.18
PITCH_RECOMPUTE_TOL = 1e-4
SAMPLE_COUNT_TOL = 1
CPA_TIME_TOL_S = 0.35
SPEED_RANGE_MIN_SPAN_FRAC = 0.12
SAME_VEHICLE_CENTROID_SPREAD_TOL = 0.15


@dataclass
class CheckResult:
    check_id: str
    section: str
    title: str
    rationale: str
    status: ResultStatus
    justification: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_wav_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        fmt = {1: "b", 2: "h", 4: "i"}[w.getsampwidth()]
        data = np.array(struct.unpack("<" + fmt * n, raw), dtype=np.float32)
        if w.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(axis=1)
        scale = float(2 ** (8 * w.getsampwidth() - 1))
        return data / scale


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clip_paths(batch_root: str, sample_dir: str) -> dict[str, str]:
    emit = os.path.join(batch_root, "emitter_centric", "audio_clips", sample_dir)
    obs = os.path.join(batch_root, "observer_centric", "audio_clips", sample_dir)
    return {
        "emit_common": os.path.join(emit, "Common"),
        "obs_common": os.path.join(obs, "Common"),
        "emit_sample": emit,
        "obs_sample": obs,
        "comparison": os.path.join(batch_root, "comparison_outputs", sample_dir),
    }


def cqt_ridge_metrics(cqt: np.ndarray) -> dict[str, float]:
    peaks = np.argmax(cqt, axis=0)
    if len(peaks) < 2:
        return {"slope": 0.0, "peak_std": 0.0}
    t = np.arange(len(peaks), dtype=np.float64)
    slope = float(np.polyfit(t, peaks.astype(np.float64), 1)[0])
    return {"slope": slope, "peak_std": float(np.std(peaks))}


def path_arc_length(positions: np.ndarray) -> float:
    pos = np.asarray(positions, dtype=np.float64)
    if len(pos) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
