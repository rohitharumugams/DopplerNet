"""
Workspace-only: stacked distance comparison spectrograms (prof / Kia flyby style).

Example target figure:
  60 mph drive-by, three CPA distances (50 m, 25 m, 10 m), 30 s, 0–800 Hz STFT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import librosa
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audio.audio_utils import extend_audio_with_overlap, get_speed_of_sound, load_audio
from workspace.quadratic_acceleration.abs_synthesis import synthesize_passby_straight

PLOT_SR = 44100
N_FFT = 4096
# Coarser hop for figures only (~2600 frames @ 30 s). hop=64 → ~20k frames and
# specshow(..., shading="gouraud") can allocate 10+ GiB.
DISPLAY_HOP = 512
WIN_LENGTH = 4096


def mph_to_mps(mph: float) -> float:
    return float(mph) * 0.44704


def doppler_ratio_limits(speed_mps: float, temp_c: float = 20.0, humidity: float = 50.0) -> tuple[float, float]:
    """Approach / recede ratios c/(c-v) and c/(c+v) for perpendicular pass-by at max radial speed."""
    c = get_speed_of_sound(temp_c, humidity)
    v = abs(float(speed_mps))
    approach = c / max(c - v, 1.0)
    recede = c / (c + v)
    return float(approach), float(recede)


def load_source_audio(
    path: str | None,
    vehicle_name: str | None,
    duration_s: float,
    sr: int = PLOT_SR,
) -> np.ndarray:
    if path and os.path.isfile(path):
        y, _ = librosa.load(path, sr=sr, mono=True)
    elif vehicle_name:
        from core.config import UPLOAD_FOLDER, DRONE_SOUNDS_FOLDER

        vehicle_file = None
        for folder in [UPLOAD_FOLDER, DRONE_SOUNDS_FOLDER]:
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
        raise ValueError("Provide --audio path or --vehicle name")

    y = extend_audio_with_overlap(y.astype(np.float32), duration_s * 2.0, sr)
    target_n = int(round(duration_s * sr))
    if len(y) >= target_n:
        return y[:target_n].astype(np.float32)
    out = np.zeros(target_n, dtype=np.float32)
    out[: len(y)] = y
    return out


def plot_distance_panel(
    audios: list[np.ndarray],
    distances_m: list[float],
    sr: int,
    speed_mph: float,
    out_path: str,
    *,
    max_y_freq: float = 800.0,
    duration_s: float | None = None,
    approach_ratio: float | None = None,
    recede_ratio: float | None = None,
    path_label: str = "400 m right -> 400 m left",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display

    n = len(audios)
    if duration_s is None:
        duration_s = len(audios[0]) / float(sr)

    if approach_ratio is None or recede_ratio is None:
        approach_ratio, recede_ratio = doppler_ratio_limits(mph_to_mps(speed_mph))

    fig, axes = plt.subplots(
        n, 1, figsize=(12, 3.2 * n), sharex=True, facecolor="white",
        gridspec_kw={"hspace": 0.35},
    )
    if n == 1:
        axes = [axes]

    suptitle = (
        f"{speed_mph:.0f} mph drive-by, source {path_label} "
        f"(Doppler ratio: {approach_ratio:.3f}x approach, {recede_ratio:.3f}x recede)"
    )
    fig.suptitle(suptitle, fontsize=13, y=0.995)

    for ax, y, d_m in zip(axes, audios, distances_m):
        y_f = np.asarray(y, dtype=np.float32)
        stft = librosa.stft(
            y_f,
            n_fft=N_FFT,
            hop_length=DISPLAY_HOP,
            win_length=WIN_LENGTH,
            window="hann",
            center=True,
        )
        power = (np.abs(stft).astype(np.float32) ** 2)
        d_db = librosa.power_to_db(power, ref=np.max).astype(np.float32, copy=False)
        del stft, power
        vmax = float(np.max(d_db))
        vmin = vmax - 60.0

        img = librosa.display.specshow(
            d_db,
            sr=sr,
            hop_length=DISPLAY_HOP,
            x_axis="time",
            y_axis="hz",
            ax=ax,
            cmap="magma",
            shading="auto",
            rasterized=True,
            vmin=vmin,
            vmax=vmax,
        )
        del d_db
        ax.set_ylim(0, float(max_y_freq))
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Kia Sportage flyby, d = {d_m:.0f} m", fontsize=11, loc="left")
        cbar = fig.colorbar(img, ax=ax, pad=0.01)
        cbar.set_label("Power (dB)")

    axes[-1].set_xlabel("Time (s)")
    for ax in axes:
        ax.set_xlim(0.0, float(duration_s))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def run_distance_spectrogram_panel(
    *,
    distances_m: list[float] | None = None,
    speed_mph: float = 60.0,
    duration_s: float = 30.0,
    audio_path: str | None = None,
    vehicle_name: str | None = "KiaSportage",
    out_dir: str = "static/workspace_outputs/distance_panel",
    max_y_freq: float = 800.0,
    cpa_time_s: float | None = None,
    angle_deg: float = 0.0,
    save_wavs: bool = True,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    if distances_m is None:
        distances_m = [50.0, 25.0, 10.0]

    speed_mps = mph_to_mps(speed_mph)
    if cpa_time_s is None:
        cpa_time_s = duration_s / 2.0

    approach, recede = doppler_ratio_limits(speed_mps)
    half_span_m = speed_mps * (duration_s / 2.0)
    path_label = f"{half_span_m:.0f} m right -> {half_span_m:.0f} m left"

    source = load_source_audio(audio_path, vehicle_name, duration_s, PLOT_SR)
    audios: list[np.ndarray] = []
    meta_clips = []

    try:
        for d_m in distances_m:
            gen = synthesize_passby_straight(
                source,
                PLOT_SR,
                speed_mps,
                float(d_m),
                duration_s,
                cpa_time_s=cpa_time_s,
                angle_deg=angle_deg,
                target_peak_s=cpa_time_s,
            )
            audios.append(gen.astype(np.float32, copy=False))
            if save_wavs:
                import soundfile as sf

                wav_name = f"flyby_d{int(d_m)}m_v{int(speed_mph)}mph.wav"
                sf.write(os.path.join(out_dir, wav_name), gen, PLOT_SR)
                meta_clips.append(wav_name)

        png_path = os.path.join(out_dir, "distance_spectrogram_panel.png")
        plot_distance_panel(
            audios,
            distances_m,
            PLOT_SR,
            speed_mph,
            png_path,
            max_y_freq=max_y_freq,
            duration_s=duration_s,
            approach_ratio=approach,
            recede_ratio=recede,
            path_label=path_label,
        )
    finally:
        audios.clear()

    summary = {
        "speed_mph": speed_mph,
        "speed_mps": speed_mps,
        "distances_m": distances_m,
        "duration_s": duration_s,
        "cpa_time_s": cpa_time_s,
        "max_y_freq_hz": max_y_freq,
        "half_span_m": half_span_m,
        "doppler_approach": approach,
        "doppler_recede": recede,
        "vehicle": vehicle_name,
        "audio_path": audio_path,
        "png": png_path,
        "wav_files": meta_clips,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    p = argparse.ArgumentParser(description="Workspace: distance comparison spectrogram panel")
    p.add_argument("--vehicle", default="KiaSportage", help="Library vehicle name")
    p.add_argument("--audio", default=None, help="Optional source WAV instead of library")
    p.add_argument("--distances", default="50,25,10", help="Comma-separated CPA distances (m)")
    p.add_argument("--speed_mph", type=float, default=60.0)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--max_freq", type=float, default=800.0)
    p.add_argument("--out_dir", default="static/workspace_outputs/distance_panel")
    args = p.parse_args()

    dists = [float(x.strip()) for x in args.distances.split(",") if x.strip()]
    summary = run_distance_spectrogram_panel(
        distances_m=dists,
        speed_mph=args.speed_mph,
        duration_s=args.duration,
        audio_path=args.audio,
        vehicle_name=None if args.audio else args.vehicle,
        out_dir=args.out_dir,
        max_y_freq=args.max_freq,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
