"""
Experimental workspace clip generation (isolated from default simulator behavior).

This module is intentionally opt-in: it is only used when the incoming batch config
includes `workspace.enabled = True`.

Output layout (flat, no Common/Essential duplication):
  audio_clips/sample_XXXXXXX/
    <clip>.wav
    batch-style *.npy + diagnostic PNGs (relocated from Common/)
    extra CQT / wide / narrow spectrogram PNGs, diagnostics overlay, path plot, metadata.json
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import librosa
import numpy as np

from audio.generation import (
    get_doppler_audio_array,
    params_for_json,
    save_numpy_outputs,
    _infer_direction_info,
    _apply_subtle_air_noise,
    _save_numpy_visualization,
    SR,
)
from audio.audio_utils import save_audio


HOP_LENGTH = 512


def _resolve_cpa_time_sec(params: dict, duration_s: float) -> float:
    """CPA time from params; ignore explicit None (common after label derivation)."""
    for key in ("cpa_time_sec", "target_cpa_time", "cpa_time"):
        if key not in params:
            continue
        val = params.get(key)
        if val is None:
            continue
        try:
            t = float(val)
            if np.isfinite(t):
                return t
        except (TypeError, ValueError):
            continue
    return float(duration_s) / 2.0


def _flatten_common_outputs(sample_dir: str) -> list[str]:
    """Move batch-style Common/*.npy (and PNGs) into flat workspace sample folder."""
    common = os.path.join(sample_dir, "Common")
    if not os.path.isdir(common):
        return []
    moved = []
    for name in os.listdir(common):
        src = os.path.join(common, name)
        dst = os.path.join(sample_dir, name)
        if os.path.isfile(dst):
            os.remove(dst)
        shutil.move(src, dst)
        moved.append(name)
    try:
        os.rmdir(common)
    except OSError:
        pass
    return moved


def _workspace_src_pitch_curve(params: dict, workspace_cfg: dict) -> np.ndarray | None:
    """
    Build a time-varying emitted/source frequency scale curve.

    Concept:
      f_obs(t) = f_src(t) * Doppler(t)
      f_src(t) couples to RPM(t) ~ v(t), producing nonlinear frequency evolution under acceleration.
    """
    if not workspace_cfg or not bool(workspace_cfg.get("enabled", False)):
        return None

    model = str(workspace_cfg.get("src_model", "rpm_linear")).lower().strip()
    if model in ("doppler_only", "none", "baseline"):
        return None

    duration = float(params.get("duration", 10.0))
    n = int(max(1, round(SR * duration)))
    t = np.arange(n, dtype=np.float32) / float(SR)

    v_cpa = float(params.get("speed", 0.0))
    a = float(params.get("acceleration", 0.0))

    t_cpa = params.get("target_cpa_time", params.get("cpa_time", duration / 2.0))
    try:
        t_cpa = float(t_cpa)
    except Exception:
        t_cpa = duration / 2.0

    v_t = v_cpa + a * (t - t_cpa)
    v_t = np.maximum(0.1, v_t)

    v_ref = float(workspace_cfg.get("v_ref_mps", 30.0))
    v_ref = max(1e-3, v_ref)

    dv = (v_t - v_cpa) / v_ref
    k1 = float(workspace_cfg.get("coupling_k1", 0.35))
    k2 = float(workspace_cfg.get("coupling_k2", 0.0))

    if model in ("rpm_linear", "linear"):
        k2 = 0.0
    elif model in ("rpm_quadratic", "quadratic"):
        pass
    else:
        return None

    scale = 1.0 + k1 * dv + k2 * (dv * dv)

    clamp_min = float(workspace_cfg.get("pitch_clamp_min", 0.35))
    clamp_max = float(workspace_cfg.get("pitch_clamp_max", 2.5))
    clamp_min = max(0.02, min(clamp_min, clamp_max))
    clamp_max = max(clamp_min, clamp_max)
    return np.clip(scale, clamp_min, clamp_max).astype(np.float32)


def _align_series(series: np.ndarray, target_len: int) -> np.ndarray:
    s = np.asarray(series, dtype=np.float32)
    if len(s) == target_len:
        return s
    if len(s) < 2 or target_len < 1:
        return np.zeros(target_len, dtype=np.float32)
    x_src = np.linspace(0.0, 1.0, len(s))
    x_dst = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_dst, x_src, s).astype(np.float32)


def _save_workspace_spectrograms(
    audio: np.ndarray,
    sample_dir: str,
    base_name: str,
    *,
    cpa_time_s: float | None = None,
) -> dict:
    """
    Three spectrogram views for pass-by diagnostics:
      1. CQT — harmonic / engine-band structure
      2. Wide-band STFT — full roadside energy (approach + CPA splash)
      3. Narrow-band STFT — low-frequency engine body (high freq resolution)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display

    os.makedirs(sample_dir, exist_ok=True)
    duration_s = len(audio) / float(SR)
    saved = {}

    def _mark_cpa(ax):
        if cpa_time_s is not None and np.isfinite(cpa_time_s):
            ax.axvline(float(cpa_time_s), color="#ff7b72", linewidth=1.2, linestyle="--", label="CPA")

    # 1) CQT
    try:
        C = librosa.cqt(audio, sr=SR, hop_length=HOP_LENGTH, n_bins=84, bins_per_octave=12)
        D = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        fig, ax = plt.subplots(figsize=(11, 4.2), facecolor="white")
        librosa.display.specshow(
            D,
            sr=SR,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="hz",
            ax=ax,
            fmin=librosa.note_to_hz("C1"),
            cmap="magma",
        )
        _mark_cpa(ax)
        ax.set_title("CQT spectrogram (harmonic / engine band)")
        ax.set_xlim(0.0, duration_s)
        fig.savefig(os.path.join(sample_dir, f"{base_name}_spectrogram_cqt.png"), dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved["spectrogram_cqt"] = f"{base_name}_spectrogram_cqt.png"
    except Exception as exc:
        print(f"[workspace] CQT spectrogram failed: {exc}")

    # 2) Wide-band STFT (0–8 kHz)
    try:
        n_fft, hop = 2048, 128
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann")
        D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        fig, ax = plt.subplots(figsize=(11, 4.2), facecolor="white")
        librosa.display.specshow(D, sr=SR, hop_length=hop, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
        ax.set_ylim(0, min(8000.0, SR / 2.0))
        _mark_cpa(ax)
        ax.set_title("Wide-band STFT (0–8 kHz)")
        ax.set_xlim(0.0, duration_s)
        fig.savefig(os.path.join(sample_dir, f"{base_name}_spectrogram_wideband.png"), dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved["spectrogram_wideband"] = f"{base_name}_spectrogram_wideband.png"
    except Exception as exc:
        print(f"[workspace] wide-band spectrogram failed: {exc}")

    # 3) Narrow-band STFT (0–1.2 kHz, high resolution)
    try:
        n_fft, hop = 8192, 512
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann")
        D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        fig, ax = plt.subplots(figsize=(11, 4.2), facecolor="white")
        librosa.display.specshow(D, sr=SR, hop_length=hop, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
        ax.set_ylim(0, 1200.0)
        _mark_cpa(ax)
        ax.set_title("Narrow-band STFT (0–1.2 kHz, engine fundamentals)")
        ax.set_xlim(0.0, duration_s)
        fig.savefig(os.path.join(sample_dir, f"{base_name}_spectrogram_narrowband.png"), dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved["spectrogram_narrowband"] = f"{base_name}_spectrogram_narrowband.png"
    except Exception as exc:
        print(f"[workspace] narrow-band spectrogram failed: {exc}")

    return saved


def _save_workspace_diagnostics_overlay(
    audio: np.ndarray,
    sample_dir: str,
    base_name: str,
    freq_ratios: np.ndarray,
    amplitudes: np.ndarray,
    params: dict,
    src_curve: np.ndarray | None,
    *,
    cpa_time_s: float | None = None,
) -> str | None:
    """Aligned overlay: CQT + Doppler ratio + gain envelope + source-pitch / speed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display

    try:
        duration_s = len(audio) / float(SR)
        n_frames = int(max(1, (len(audio) - HOP_LENGTH) // HOP_LENGTH + 1))
        t_frames = np.linspace(0.0, duration_s, n_frames, endpoint=False, dtype=np.float32)

        fr = _align_series(freq_ratios, n_frames)
        amp = _align_series(amplitudes, n_frames)
        amp_n = amp / (np.max(amp) + 1e-8)

        v0 = float(params.get("speed", 0.0))
        acc = float(params.get("acceleration", 0.0))
        t_cpa = float(cpa_time_s) if cpa_time_s is not None and np.isfinite(cpa_time_s) else _resolve_cpa_time_sec(params, duration_s)
        v_t = v0 + acc * (t_frames - t_cpa)

        fig, axes = plt.subplots(4, 1, figsize=(12, 9.5), sharex=True, facecolor="white")
        ax0, ax1, ax2, ax3 = axes

        C = librosa.cqt(audio, sr=SR, hop_length=HOP_LENGTH, n_bins=84, bins_per_octave=12)
        D = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        librosa.display.specshow(
            D,
            sr=SR,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="hz",
            ax=ax0,
            fmin=librosa.note_to_hz("C1"),
            cmap="magma",
        )
        ax0.set_title("CQT (flanks vs CPA region)")
        ax0.set_ylabel("Hz")

        ax1.plot(t_frames, fr, color="#58a6ff", linewidth=1.1)
        ax1.set_ylabel("Doppler ratio")
        ax1.set_title("freq_ratios(t) — geometric Doppler factor")
        ax1.grid(True, alpha=0.35)

        ax2.plot(t_frames, amp_n, color="#2ca02c", linewidth=1.1)
        ax2.set_ylabel("Gain (norm)")
        ax2.set_title("amplitude(t) — path gain / envelope")
        ax2.grid(True, alpha=0.35)

        if src_curve is not None:
            sp = _align_series(src_curve, n_frames)
            ax3.plot(t_frames, sp, color="#d62728", linewidth=1.1, label="g(v) src pitch")
            ax3.set_ylabel("g(v)")
            ax3.set_title("Source pitch coupling g(v) (pre-Doppler)")
        else:
            ax3.plot(t_frames, v_t, color="#d62728", linewidth=1.1, label="v(t)")
            ax3.set_ylabel("m/s")
            ax3.set_title("Kinematic speed v(t)")
        ax3.grid(True, alpha=0.35)
        ax3.set_xlabel("Time (s)")

        if cpa_time_s is not None and np.isfinite(cpa_time_s):
            for ax in axes:
                ax.axvline(float(cpa_time_s), color="#ff7b72", linewidth=1.0, linestyle="--", alpha=0.85)

        fig.tight_layout()
        out_name = f"{base_name}_diagnostics_overlay.png"
        fig.savefig(os.path.join(sample_dir, out_name), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out_name
    except Exception as exc:
        print(f"[workspace] diagnostics overlay failed: {exc}")
        return None


def generate_single_clip_workspace(
    vehicle_name,
    path_type,
    params,
    output_dir,
    batch_id,
    index,
    config,
    custom_filename=None,
):
    """
    Workspace-only clip synthesis (acceleration + optional g(v) source coupling, flat folders):
      - Batch-style .npy / diagnostic PNGs (relocated to sample root, no Common/Essential)
      - CQT + wide-band + narrow-band STFT PNGs; aligned diagnostics overlay
    """
    synth_params = dict(params)
    ws_cfg = (config.get("workspace", {}) or {}) if isinstance(config, dict) else {}
    src_curve = _workspace_src_pitch_curve(synth_params, ws_cfg)

    doppler_audio, freq_ratios, amplitudes = get_doppler_audio_array(
        vehicle_name,
        path_type,
        synth_params,
        src_pitch_curve=src_curve,
    )

    atm_cfg = config.get("atmosphere", {}) if isinstance(config, dict) else {}
    if bool(atm_cfg.get("add_air_noise", False)):
        noise_strength = float(atm_cfg.get("air_noise_strength", 8.0))
        noise_freq_hz = float(atm_cfg.get("air_noise_frequency_hz", 1200.0))
        doppler_audio = _apply_subtle_air_noise(doppler_audio, SR, noise_strength, noise_freq_hz)

    sample_dir = os.path.join(output_dir, f"sample_{index:07d}")
    os.makedirs(sample_dir, exist_ok=True)

    output_format = config.get("output", {}).get("format", "wav")
    _, direction_text_for_name = _infer_direction_info(path_type, params)
    meta_name = (
        f"{vehicle_name}_{path_type}_{direction_text_for_name}_"
        f"{params['speed']}mps_{params.get('distance', params.get('h', 0))}m_{index:07d}"
    )

    if custom_filename:
        try:
            sno = custom_filename.split("_")[-1]
            base_name = f"(test_{sno}_){meta_name}"
        except Exception:
            base_name = f"({custom_filename}_){meta_name}"
    else:
        base_name = meta_name

    filename = f"{base_name}.{output_format}"
    audio_path = os.path.join(sample_dir, filename)
    if output_format == "mp3":
        wav_path = audio_path.replace(".mp3", "_temp.wav")
        save_audio(doppler_audio, wav_path)
        audio_path = audio_path.replace(".mp3", ".wav")
        os.rename(wav_path, audio_path)
        filename = os.path.basename(audio_path)
    else:
        save_audio(doppler_audio, audio_path)

    direction_label, direction_text = _infer_direction_info(path_type, params)
    from physics.recording_labels import derive_recording_labels

    duration_s = len(doppler_audio) / float(SR)
    spectrogram_type = config.get("output", {}).get("spectrogram_type", "cqt")
    features = save_numpy_outputs(
        doppler_audio,
        sample_dir,
        spectrogram_type,
        config,
        base_name=base_name,
        essential_dir=None,
        params=params,
    )
    npy_files = _flatten_common_outputs(sample_dir)

    n_frames = len(features["time"])
    fr = _align_series(freq_ratios, n_frames)
    amp = _align_series(amplitudes, n_frames)
    np.save(os.path.join(sample_dir, "freq_ratios.npy"), fr)
    np.save(os.path.join(sample_dir, "amplitudes.npy"), amp)
    if src_curve is not None:
        np.save(os.path.join(sample_dir, "src_pitch_curve.npy"), _align_series(src_curve, n_frames))
    npy_files.extend(["freq_ratios.npy", "amplitudes.npy"])

    generate_diagnostics = bool(config.get("output", {}).get("generate_diagnostics", True))
    if generate_diagnostics:
        spec = features["spec"]
        freq_bins = librosa.cqt_frequencies(
            84, fmin=librosa.note_to_hz("C1"), bins_per_octave=12
        ).astype(np.float32)
        if spectrogram_type != "cqt":
            freq_bins = None
        _save_numpy_visualization(
            doppler_audio,
            spec,
            features["frequency"],
            features.get("dfdt"),
            features["rms"],
            features.get("spec_topk"),
            features["time"],
            spectrogram_type,
            sample_dir,
            generate_diagnostics=True,
            base_name=base_name,
            freq_bins=freq_bins,
            hop_length=HOP_LENGTH,
        )

    labels = derive_recording_labels(
        path_type,
        params,
        doppler_audio,
        features,
        vehicle_name,
        direction_label,
        direction_text,
    )
    cpa_time_s = _resolve_cpa_time_sec(params, duration_s)
    label_cpa = labels.get("cpa_time_sec")
    if label_cpa is not None:
        try:
            if np.isfinite(float(label_cpa)):
                cpa_time_s = float(label_cpa)
        except (TypeError, ValueError):
            pass
    params["cpa_time_sec"] = cpa_time_s

    spec_files = _save_workspace_spectrograms(
        doppler_audio, sample_dir, base_name, cpa_time_s=cpa_time_s
    )
    overlay_file = None
    if generate_diagnostics:
        overlay_file = _save_workspace_diagnostics_overlay(
            doppler_audio,
            sample_dir,
            base_name,
            freq_ratios,
            amplitudes,
            params,
            src_curve,
            cpa_time_s=cpa_time_s,
        )

    npy_info = {
        "spectrogram_type": spectrogram_type,
        "spec_filename": features.get("spec_filename"),
        "n_frames": int(n_frames),
        "files": sorted(set(npy_files)),
    }

    from visualization.plot_utils import save_path_plot

    plot_params = dict(params)
    if cpa_time_s is not None:
        plot_params["cpa_time_sec"] = cpa_time_s
    plot_file = save_path_plot(path_type, plot_params, sample_dir, base_name)

    workspace_meta = {
        "enabled": True,
        "kind": str(ws_cfg.get("kind", "quadratic_acceleration_testing")),
        "src_model": str(ws_cfg.get("src_model", "rpm_linear")),
        "coupling_k1": float(ws_cfg.get("coupling_k1", 0.0)),
        "coupling_k2": float(ws_cfg.get("coupling_k2", 0.0)),
        "v_ref_mps": float(ws_cfg.get("v_ref_mps", 30.0)),
        "pitch_clamp_min": float(ws_cfg.get("pitch_clamp_min", 0.35)),
        "pitch_clamp_max": float(ws_cfg.get("pitch_clamp_max", 2.5)),
        "output_layout": "flat",
        "spectrograms": spec_files,
        "diagnostics_overlay": overlay_file,
        "npy": npy_info,
    }
    sample_metadata = {
        "batch_id": batch_id,
        "index": index,
        "vehicle": vehicle_name,
        "path_type": path_type,
        "filename": filename,
        "audio_path": filename,
        "path_plot": plot_file,
        "parameters": params_for_json(params),
        "labels": labels,
        "workspace": workspace_meta,
        "freq_ratio_range": {
            "min": float(np.min(freq_ratios)),
            "max": float(np.max(freq_ratios)),
        },
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(sample_metadata, f, indent=2)

    return {
        "filename": filename,
        "index": index,
        "vehicle": vehicle_name,
        "path_type": path_type,
        "direction_text": direction_text,
        "parameters": params_for_json(params),
        "labels": labels,
        "freq_ratio_range": {
            "min": float(np.min(freq_ratios)),
            "max": float(np.max(freq_ratios)),
        },
        "path_plot": plot_file or f"{base_name}.png",
        "sample_dir": f"sample_{index:07d}",
        "acceleration": float(params.get("acceleration", 0.0)),
        "pass_by_in_clip": bool(params.get("pass_by_in_clip", True)),
        "workspace": workspace_meta,
    }
