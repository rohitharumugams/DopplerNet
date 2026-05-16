"""
CPA time sampling and asymmetric time→path-parameter maps for benchmark clips.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np

BENCH_SINGLE_IDS = frozenset({"B1", "B2", "B3", "B4", "B5", "B6"})
DEFAULT_BENCH_DURATION_S = 10.0
DEFAULT_CPA_MARGIN_S = 0.5


def is_single_vehicle_benchmark_active(bench_cfg: dict) -> bool:
    if not bench_cfg.get("enabled", False):
        return False
    selected = set(bench_cfg.get("selected", []) or [])
    return bool(selected & BENCH_SINGLE_IDS)


def sample_benchmark_cpa_time(
    bench_params: dict,
    duration_s: float = DEFAULT_BENCH_DURATION_S,
    margin_s: float = DEFAULT_CPA_MARGIN_S,
) -> float:
    """Uniform CPA time over the clip (margins only), for pass-by benchmark clips."""
    del bench_params  # reserved for future options; no min/max band
    lo = float(margin_s)
    hi = float(duration_s) - float(margin_s)
    if hi <= lo:
        return float(duration_s) / 2.0
    return float(random.uniform(lo, hi))


def parabola_tau_at_time(
    t: np.ndarray,
    duration_s: float,
    cpa_time_s: float,
) -> np.ndarray:
    """
    Map physical time t ∈ [0, T] to τ ∈ [-1, 1] with vertex (CPA) at ``cpa_time_s``.
    """
    t = np.asarray(t, dtype=float)
    T = max(1e-9, float(duration_s))
    t_cpa = float(np.clip(cpa_time_s, 1e-6, T - 1e-6))
    tau = np.empty_like(t)
    left = t <= t_cpa
    tau[left] = -1.0 + (t[left] / t_cpa)
    denom = max(1e-9, T - t_cpa)
    tau[~left] = (t[~left] - t_cpa) / denom
    return np.clip(tau, -1.0, 1.0)


def warp_tau_for_cpa(
    t: np.ndarray,
    duration_s: float,
    cpa_time_s: float,
    tau_at_cpa: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Piecewise-linear τ(t) with τ(t_cpa) = tau_at_cpa, τ(0)=0, τ(T)=1.
    Returns (tau, dtau_dt).
    """
    t = np.asarray(t, dtype=float)
    T = max(1e-9, float(duration_s))
    t_cpa = float(np.clip(cpa_time_s, 1e-6, T - 1e-6))
    tau_cpa = float(np.clip(tau_at_cpa, 1e-6, 1.0 - 1e-6))
    tau = np.empty_like(t)
    dtaudt = np.empty_like(t)
    left = t <= t_cpa
    tau[left] = (tau_cpa / t_cpa) * t[left]
    dtaudt[left] = tau_cpa / t_cpa
    denom = max(1e-9, T - t_cpa)
    tau[~left] = tau_cpa + ((1.0 - tau_cpa) / denom) * (t[~left] - t_cpa)
    dtaudt[~left] = (1.0 - tau_cpa) / denom
    return tau, dtaudt
