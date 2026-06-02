"""Forward overlap-add deposit onto the observer buffer."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def splat_linear(
    buffer: np.ndarray,
    index_float: float,
    value: float,
) -> None:
    """Add value to buffer with linear interpolation between adjacent samples."""
    if not np.isfinite(index_float):
        return
    i0 = int(np.floor(index_float))
    frac = index_float - i0
    if 0 <= i0 < len(buffer) - 1:
        buffer[i0] += value * (1.0 - frac)
        buffer[i0 + 1] += value * frac
    elif i0 == len(buffer) - 1 and 0 <= i0 < len(buffer):
        buffer[i0] += value


def forward_point_deposit(
    source_emission: np.ndarray,
    t_o: np.ndarray,
    weights: np.ndarray,
    sr: int,
    *,
    buffer_duration_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deposit emission samples at arrival times t_o (seconds).

    Returns (observer_buffer, t_o_used).
    """
    n = len(source_emission)
    if n == 0:
        return np.zeros(0, dtype=np.float32), t_o
    t_o = np.asarray(t_o, dtype=np.float64)
    if buffer_duration_s is None:
        buffer_duration_s = float(t_o.max()) + 2.0 / sr
    buf_len = int(np.ceil(buffer_duration_s * sr)) + 1
    y = np.zeros(buf_len, dtype=np.float64)
    for m in range(n):
        splat_linear(y, t_o[m] * sr, float(weights[m] * source_emission[m]))
    return y.astype(np.float32), t_o


def resample_to_uniform_observer(
    t_o: np.ndarray,
    weights: np.ndarray,
    source_emission: np.ndarray,
    sr: int,
) -> np.ndarray:
    """
    Uniform observer grid by inverting emission→observer time map (kinematic warp, no R/c).

    Equivalent to observer-centric cumulative-alpha read mapping when t_o integrates dt/alpha.
    """
    n = len(source_emission)
    t_uni = np.arange(n, dtype=np.float64) / sr
    t_e = np.arange(n, dtype=np.float64) / sr
    src = np.asarray(source_emission, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if n < 2:
        return (src * w).astype(np.float32)
    t_o = np.asarray(t_o, dtype=np.float64)
    eps = 1e-9
    t_o_mono = t_o + eps * np.arange(n)
    te_at_tu = np.interp(t_uni, t_o_mono, t_e, left=t_e[0], right=t_e[-1])
    read_idx = te_at_tu * sr
    x_idx = np.arange(n, dtype=np.float64)
    if n >= 4:
        f_src = interp1d(x_idx, src, kind="cubic", bounds_error=False, fill_value=(src[0], src[-1]))
        f_w = interp1d(x_idx, w, kind="cubic", bounds_error=False, fill_value=(w[0], w[-1]))
        raw = f_src(read_idx)
        w_interp = f_w(read_idx)
    else:
        raw = np.interp(read_idx, x_idx, src, left=src[0], right=src[-1])
        w_interp = np.interp(read_idx, x_idx, w, left=w[0], right=w[-1])
    return (raw * w_interp).astype(np.float32)


def cubic_resample_uniform(
    y_irregular_times: np.ndarray,
    y_irregular_values: np.ndarray,
    t_uniform: np.ndarray,
) -> np.ndarray:
    """Cubic interpolation of a sparse/deposited trace onto uniform times."""
    if len(y_irregular_times) < 4:
        return np.interp(t_uniform, y_irregular_times, y_irregular_values).astype(np.float32)
    f = interp1d(
        y_irregular_times,
        y_irregular_values,
        kind="cubic",
        bounds_error=False,
        fill_value=(y_irregular_values[0], y_irregular_values[-1]),
    )
    return f(t_uniform).astype(np.float32)
