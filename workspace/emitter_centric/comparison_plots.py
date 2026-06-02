"""Side-by-side and diagnostic comparison figures (emitter-centric workspace)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from audio.audio_utils import SR


def _caption_footer(params: dict, vehicle: str, source_file: str) -> str:
    return (
        f"vehicle={vehicle}  |  speed={params.get('speed', 0):.2f} m/s  |  "
        f"accel={params.get('acceleration', 0):.2f} m/s²  |  path={params.get('path_type')}  |  "
        f"duration={params.get('duration', 10):.1f} s  |  source={source_file}"
    )


def _spec_mag(y: np.ndarray, *, wideband: bool, sr: int = SR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import librosa

    hop = 512
    if wideband:
        n_fft = 2048
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        fmax = 8000.0
    else:
        n_fft = 4096
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        fmax = 1200.0
    mask = freqs <= fmax
    S = S[mask, :]
    freqs = freqs[mask]
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    return S, freqs, times


def _plot_side_by_side_spec(
    y_obs: np.ndarray,
    y_emit: np.ndarray,
    out_path: str,
    *,
    title: str,
    wideband: bool,
    vehicle: str,
    speed: float,
    footer: str,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa

    S0, freqs, times = _spec_mag(y_obs, wideband=wideband)
    S1, _, _ = _spec_mag(y_emit, wideband=wideband)
    extent = [float(times[0]), float(times[-1]), float(freqs[0]), float(freqs[-1])]
    ref = max(np.max(S0), np.max(S1), 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, S, label in zip(
        axes,
        (S0, S1),
        ("Observer-centric (roadside)", "Emitter-centric (co-moving source)"),
    ):
        ax.imshow(
            librosa.amplitude_to_db(S, ref=ref),
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="magma",
        )
        ax.set_title(f"{label}\n{vehicle} @ {speed:.1f} m/s")
        ax.set_ylabel("Hz")
        ax.set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.02, footer, ha="center", fontsize=8, color="#444")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor="white")
    plt.close(fig)
    return out_path


def _plot_cqt_pair(
    y_obs: np.ndarray,
    y_emit: np.ndarray,
    out_path: str,
    *,
    vehicle: str,
    speed: float,
    footer: str,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display

    hop = 512
    C0 = np.abs(librosa.cqt(y_obs, sr=SR, hop_length=hop, n_bins=84, bins_per_octave=12))
    C1 = np.abs(librosa.cqt(y_emit, sr=SR, hop_length=hop, n_bins=84, bins_per_octave=12))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, C, label in zip(
        axes,
        (C0, C1),
        ("Observer-centric", "Emitter-centric (source frame)"),
    ):
        librosa.display.specshow(
            librosa.amplitude_to_db(C, ref=np.max),
            sr=SR,
            hop_length=hop,
            x_axis="time",
            y_axis="cqt_hz",
            ax=ax,
        )
        ax.set_title(f"{label}\n{vehicle} @ {speed:.1f} m/s")
    fig.suptitle("CQT comparison", fontsize=12)
    fig.text(0.5, 0.02, footer, ha="center", fontsize=8, color="#444")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor="white")
    plt.close(fig)
    return out_path


def _plot_diagnostics_grid(
    out_path: str,
    *,
    params: dict,
    kin_obs: dict[str, np.ndarray] | None,
    kin_emit: dict[str, np.ndarray] | None,
    freq_ratio: np.ndarray | None,
    y_obs: np.ndarray,
    y_emit: np.ndarray,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa

    n = min(len(y_obs), len(y_emit))
    t = np.arange(n) / SR
    hop = 512
    rms_obs = librosa.feature.rms(y=y_obs[:n], hop_length=hop)[0]
    rms_emit = librosa.feature.rms(y=y_emit[:n], hop_length=hop)[0]
    t_rms = librosa.frames_to_time(np.arange(len(rms_obs)), sr=SR, hop_length=hop)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    ax = axes.ravel()

    if kin_obs and "x" in kin_obs:
        ax[0].plot(t[: len(kin_obs["x"])], kin_obs["x"], label="x")
        ax[0].plot(t[: len(kin_obs["y"])], kin_obs["y"], label="y")
        ax[0].scatter([0], [0], c="gold", s=40, label="observer")
        ax[0].set_title("Trajectory (scene)")
        ax[0].legend(fontsize=7)
    else:
        ax[0].text(0.5, 0.5, "No kinematics", ha="center")
    ax[0].set_xlabel("t (s)")

    if kin_obs and "v_r" in kin_obs:
        tt = t[: len(kin_obs["v_r"])]
        ax[1].plot(tt, kin_obs["v_r"], label="v_r observer frame")
        if freq_ratio is not None:
            ax[1].twinx().plot(tt, freq_ratio[: len(tt)], "r--", alpha=0.7, label="Doppler ratio")
        ax[1].set_title("Radial velocity / Doppler ratio")
    ax[1].set_xlabel("t (s)")

    if kin_emit and "v" in kin_emit:
        ax[2].plot(kin_emit.get("t", t[: len(kin_emit["v"])]), kin_emit["v"])
        ax[2].set_title("Along-track speed (source frame)")
    ax[2].set_xlabel("t (s)")

    ax[3].plot(t_rms, rms_obs, label="observer RMS")
    ax[3].plot(t_rms[: len(rms_emit)], rms_emit[: len(t_rms)], label="emitter RMS")
    ax[3].set_title("RMS envelope")
    ax[3].legend(fontsize=7)

    if freq_ratio is not None:
        ax[4].plot(t[: len(freq_ratio)], freq_ratio)
        ax[4].set_title("Doppler ratio α(t) — observer only")
    else:
        ax[4].axis("off")

    diff = y_obs[:n] - y_emit[:n]
    ax[5].plot(t, diff)
    ax[5].set_title("Waveform residual (obs − emit)")
    ax[5].set_xlabel("t (s)")

    fig.suptitle(
        f"Diagnostics — {params.get('path_type')}  v={params.get('speed', 0):.1f} m/s  "
        f"a={params.get('acceleration', 0):.2f} m/s²",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)
    return out_path


def generate_all_comparisons(
    comp_dir: str,
    *,
    y_obs: np.ndarray,
    y_emit: np.ndarray,
    params: dict,
    vehicle: str,
    source_file: str,
    kin_obs: dict | None = None,
    kin_emit: dict | None = None,
    freq_ratio: np.ndarray | None = None,
    metrics: dict | None = None,
) -> dict[str, str]:
    """Write comparison_outputs artifacts for one sample."""
    os.makedirs(comp_dir, exist_ok=True)
    footer = _caption_footer(params, vehicle, source_file)
    speed = float(params.get("speed", 0))
    paths = {
        "narrowband_comparison": _plot_side_by_side_spec(
            y_obs,
            y_emit,
            os.path.join(comp_dir, "narrowband_comparison.png"),
            title="Narrowband spectrogram comparison (0–1.2 kHz)",
            wideband=False,
            vehicle=vehicle,
            speed=speed,
            footer=footer,
        ),
        "wideband_comparison": _plot_side_by_side_spec(
            y_obs,
            y_emit,
            os.path.join(comp_dir, "wideband_comparison.png"),
            title="Wideband spectrogram comparison (0–8 kHz)",
            wideband=True,
            vehicle=vehicle,
            speed=speed,
            footer=footer,
        ),
        "cqt_comparison": _plot_cqt_pair(
            y_obs,
            y_emit,
            os.path.join(comp_dir, "cqt_comparison.png"),
            vehicle=vehicle,
            speed=speed,
            footer=footer,
        ),
        "diagnostics_panel": _plot_diagnostics_grid(
            os.path.join(comp_dir, "diagnostics_panel.png"),
            params=params,
            kin_obs=kin_obs,
            kin_emit=kin_emit,
            freq_ratio=freq_ratio,
            y_obs=y_obs,
            y_emit=y_emit,
        ),
    }
    if metrics:
        from workspace.emitter_centric.comparison_report import write_comparison_report

        notes = [
            "Emitter branch has no geometric Doppler; observer branch shows pass-by wings.",
            "Large narrowband divergence at CPA wings is expected when comparison is enabled.",
        ]
        write_comparison_report(
            os.path.join(comp_dir, "comparison_report.txt"),
            params=params,
            vehicle=vehicle,
            comparison_metrics=metrics,
            notes=notes,
        )
        paths["comparison_report"] = os.path.join(comp_dir, "comparison_report.txt")
    return paths
