"""Side-by-side and diagnostic comparison figures (emitter-centric workspace)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from audio.audio_utils import SR
from visualization.plot_utils import format_image_number

# Upper frequency limit for all comparison spectrograms (pass-by ridge band).
FMAX_HZ = 1250.0


def _caption_footer(params: dict, vehicle: str, source_file: str) -> str:
    return (
        f"vehicle={vehicle} | speed={format_image_number(params.get('speed', 0))} m/s | "
        f"accel={format_image_number(params.get('acceleration', 0))} m/s² | path={params.get('path_type')} | "
        f"duration={format_image_number(params.get('duration', 10))} s | source={source_file}"
    )


def _spec_mag(
    y: np.ndarray,
    *,
    sr: int = SR,
    fmax: float = FMAX_HZ,
    n_fft: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import librosa

    hop = 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = freqs <= fmax
    S = S[mask, :]
    freqs = freqs[mask]
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    return S, freqs, times


def _cqt_db(y: np.ndarray, *, hop: int, ref: float) -> tuple[np.ndarray, np.ndarray]:
    """CQT magnitude (dB) trimmed to FMAX_HZ; returns (data, y_extent_lo)."""
    import librosa

    n_bins = 84
    fmin = float(librosa.note_to_hz("C1"))
    C = np.abs(librosa.cqt(y, sr=SR, hop_length=hop, n_bins=n_bins, bins_per_octave=12))
    freqs = librosa.cqt_frequencies(n_bins, fmin=fmin, bins_per_octave=12)
    mask = freqs <= FMAX_HZ
    C = C[mask, :]
    y_lo = float(freqs[mask][0]) if np.any(mask) else 0.0
    return librosa.amplitude_to_db(C, ref=ref), y_lo


def _stft_db(
    y: np.ndarray,
    *,
    hop: int,
    n_fft: int,
    ref: float,
) -> np.ndarray:
    import librosa

    S, freqs, _times = _spec_mag(y, n_fft=n_fft)
    return librosa.amplitude_to_db(S, ref=ref)


def _plot_stacked_spectrogram_comparison(
    y_obs: np.ndarray,
    y_emit: np.ndarray,
    out_path: str,
    *,
    vehicle: str,
    speed: float,
    footer: str,
) -> str:
    """Publication-style 3×2 spectrogram grid with row labels and shared colorbars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa
    from matplotlib.colors import Normalize

    import librosa

    hop = 512
    t_end = min(len(y_obs), len(y_emit)) / float(SR)
    row_names = ("CQT", "Narrowband", "Wideband")
    col_titles = ("Observer-centric (roadside)", "Emitter-centric (co-moving source)")

    C0 = np.abs(librosa.cqt(y_obs, sr=SR, hop_length=hop, n_bins=84, bins_per_octave=12))
    C1 = np.abs(librosa.cqt(y_emit, sr=SR, hop_length=hop, n_bins=84, bins_per_octave=12))
    cqt_ref = max(float(np.max(C0)), float(np.max(C1)), 1e-9)
    d0_cqt, y_lo_cqt = _cqt_db(y_obs, hop=hop, ref=cqt_ref)
    d1_cqt, _ = _cqt_db(y_emit, hop=hop, ref=cqt_ref)

    s_ref_n = max(
        float(np.max(np.abs(librosa.stft(y_obs, n_fft=4096, hop_length=hop)))),
        float(np.max(np.abs(librosa.stft(y_emit, n_fft=4096, hop_length=hop)))),
        1e-9,
    )
    d0_narrow = _stft_db(y_obs, hop=hop, n_fft=4096, ref=s_ref_n)
    d1_narrow = _stft_db(y_emit, hop=hop, n_fft=4096, ref=s_ref_n)

    s_ref_w = max(
        float(np.max(np.abs(librosa.stft(y_obs, n_fft=2048, hop_length=hop)))),
        float(np.max(np.abs(librosa.stft(y_emit, n_fft=2048, hop_length=hop)))),
        1e-9,
    )
    d0_wide = _stft_db(y_obs, hop=hop, n_fft=2048, ref=s_ref_w)
    d1_wide = _stft_db(y_emit, hop=hop, n_fft=2048, ref=s_ref_w)

    rows = [
        (row_names[0], d0_cqt, d1_cqt, y_lo_cqt),
        (row_names[1], d0_narrow, d1_narrow, 0.0),
        (row_names[2], d0_wide, d1_wide, 0.0),
    ]

    fig = plt.figure(figsize=(10.0, 7.0), facecolor="white", layout="constrained")
    gs = fig.add_gridspec(
        5,
        4,
        height_ratios=(0.10, 1.0, 1.0, 1.0, 0.07),
        width_ratios=(0.06, 1.0, 1.0, 0.034),
        wspace=0.05,
        hspace=0.04,
    )

    # Column headers (row 0)
    for col, title in enumerate(col_titles, start=1):
        ax_h = fig.add_subplot(gs[0, col])
        ax_h.set_axis_off()
        ax_h.text(
            0.5,
            0.4,
            title,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="semibold",
            transform=ax_h.transAxes,
        )
    fig.add_subplot(gs[0, 0]).set_axis_off()
    fig.add_subplot(gs[0, 3]).set_axis_off()

    last_im = None
    for row_idx, (label, d_obs, d_emit, y_lo) in enumerate(rows):
        gs_row = row_idx + 1
        extent = [0.0, t_end, y_lo, FMAX_HZ]
        vmin = float(min(np.min(d_obs), np.min(d_emit)))
        vmax = float(max(np.max(d_obs), np.max(d_emit)))
        norm = Normalize(vmin=vmin, vmax=vmax)

        ax_label = fig.add_subplot(gs[gs_row, 0])
        ax_label.set_axis_off()
        ax_label.text(
            0.95,
            0.5,
            label,
            rotation=90,
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            transform=ax_label.transAxes,
        )

        for col, d in enumerate((d_obs, d_emit), start=1):
            ax = fig.add_subplot(gs[gs_row, col])
            last_im = ax.imshow(
                d,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap="magma",
                norm=norm,
                interpolation="nearest",
            )
            ax.set_xlim(0.0, t_end)
            ax.set_ylim(0.0, FMAX_HZ)
            ax.set_ylabel("Hz", fontsize=8, labelpad=2)
            ax.tick_params(axis="both", labelsize=7, pad=1)
            if gs_row == 3:
                ax.set_xlabel("Time (s)", fontsize=8, labelpad=4)
            else:
                ax.tick_params(axis="x", labelbottom=False)

        cax = fig.add_subplot(gs[gs_row, 3])
        cb = fig.colorbar(last_im, cax=cax, orientation="vertical")
        cb.ax.tick_params(labelsize=6, length=2)
        cb.set_label("dB", fontsize=7, labelpad=2)

    ax_footer = fig.add_subplot(gs[4, 1:3])
    ax_footer.set_axis_off()
    ax_footer.text(
        0.5,
        0.65,
        footer,
        ha="center",
        va="center",
        fontsize=7,
        color="#444",
        transform=ax_footer.transAxes,
    )
    fig.add_subplot(gs[4, 0]).set_axis_off()
    fig.add_subplot(gs[4, 3]).set_axis_off()

    fig.suptitle(f"{vehicle} @ {format_image_number(speed)} m/s", fontsize=12, fontweight="semibold")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
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
        f"Diagnostics — {params.get('path_type')}  v={format_image_number(params.get('speed', 0))} m/s  "
        f"a={format_image_number(params.get('acceleration', 0))} m/s²",
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
        "stacked_spectrogram_comparison": _plot_stacked_spectrogram_comparison(
            y_obs,
            y_emit,
            os.path.join(comp_dir, "stacked_spectrogram_comparison.png"),
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
