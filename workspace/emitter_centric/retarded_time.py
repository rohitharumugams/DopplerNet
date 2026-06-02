"""Retarded-time maps and monotonicity guards (emitter-centric only)."""

from __future__ import annotations

import numpy as np

from workspace.emitter_centric.config import MR_MAX


def forward_retarded_time(
    t_e: np.ndarray,
    r: np.ndarray,
    c_sound: float,
    *,
    use_propagation_delay: bool,
) -> np.ndarray:
    """Observer arrival time for each emission instant."""
    t_e = np.asarray(t_e, dtype=np.float64)
    if not use_propagation_delay:
        return t_e.copy()
    return t_e + np.asarray(r, dtype=np.float64) / float(c_sound)


def cumulative_kinematic_observer_time(
    t_e: np.ndarray,
    freq_ratio: np.ndarray,
) -> np.ndarray:
    """
    Observer clock from emission clock without explicit R/c delay.

    dt_o/dt_e = 1/alpha with alpha = c/(c+v_r).
    """
    t_e = np.asarray(t_e, dtype=np.float64)
    alpha = np.asarray(freq_ratio, dtype=np.float64)
    n = len(t_e)
    if n == 0:
        return t_e
    dt_e = np.diff(t_e, prepend=t_e[0] - (t_e[1] - t_e[0]) if n > 1 else 0.0)
    if n == 1:
        dt_e = np.array([1.0 / max(alpha[0], 1e-9)])
    else:
        dt_e = np.diff(t_e)
        dt_e = np.concatenate(([dt_e[0]], dt_e))
    inv_alpha = 1.0 / np.maximum(alpha, 1e-9)
    t_o = np.cumsum(dt_e * inv_alpha)
    t_o -= t_o[0]
    return t_o


def check_monotonicity(
    freq_ratio: np.ndarray,
    c_sound: float,
    *,
    mr_max: float = MR_MAX,
) -> None:
    """
    Raise ValueError if radial Mach exceeds guard or 1 + v_r/c <= 0.

    alpha = c/(c+v_r) => 1 + v_r/c = 1/alpha.
    """
    alpha = np.asarray(freq_ratio, dtype=np.float64)
    one_plus_mr = 1.0 / np.maximum(alpha, 1e-12)
    m_r = 1.0 - one_plus_mr
    if np.any(one_plus_mr <= 0):
        raise ValueError("Monotonicity violated: 1 + v_r/c <= 0 for some samples")
    if np.any(np.abs(m_r) >= mr_max):
        raise ValueError(
            f"Radial Mach |v_r/c| >= {mr_max} for some samples (max |M_r|={np.max(np.abs(m_r)):.4f})"
        )


def deposit_jacobian_amplitude(
    amplitude: np.ndarray,
    freq_ratio: np.ndarray,
) -> np.ndarray:
    """A_deposit = A_physics / (1 + M_r) = A_physics * alpha."""
    amp = np.asarray(amplitude, dtype=np.float64)
    alpha = np.asarray(freq_ratio, dtype=np.float64)
    return (amp * alpha).astype(np.float32)
