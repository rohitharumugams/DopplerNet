"""Compare observer-centric vs emitter-centric waveforms."""

from __future__ import annotations

import numpy as np


def _level_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale b to match RMS of a for fair residual."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    rms_a = np.sqrt(np.mean(a**2) + 1e-20)
    rms_b = np.sqrt(np.mean(b**2) + 1e-20)
    if rms_b > 0:
        b = b * (rms_a / rms_b)
    return a, b


def compare_waveforms(
    observer_centric: np.ndarray,
    emitter_centric: np.ndarray,
    *,
    align_samples: int = 0,
) -> dict:
    """
    Level-matched max abs residual and dBFS metrics (plan §8.10 target < -60 dBFS).
    """
    a = np.asarray(observer_centric, dtype=np.float64)
    b = np.asarray(emitter_centric, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if align_samples:
        shift = int(align_samples)
        if shift > 0:
            b = np.roll(b, shift)
        elif shift < 0:
            a = np.roll(a, -shift)
    a_m, b_m = _level_match(a, b)
    residual = a_m - b_m
    peak = float(np.max(np.abs(a_m)) + 1e-20)
    max_abs = float(np.max(np.abs(residual)))
    rms_res = float(np.sqrt(np.mean(residual**2)))
    dbfs_peak_residual = float(20.0 * np.log10(max_abs / peak + 1e-20))
    dbfs_rms_residual = float(20.0 * np.log10(rms_res / peak + 1e-20))
    corr = float(np.corrcoef(a_m, b_m)[0, 1]) if n > 1 else 1.0
    return {
        "n_samples": n,
        "max_abs_residual": max_abs,
        "rms_residual": rms_res,
        "dbfs_peak_residual_vs_peak": dbfs_peak_residual,
        "dbfs_rms_residual_vs_peak": dbfs_rms_residual,
        "correlation": corr,
        "parity_target_dbfs": -60.0,
        "meets_parity_target": dbfs_peak_residual < -60.0,
    }
