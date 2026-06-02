#!/usr/bin/env python3
"""
Analysis-by-synthesis (v, d) grid — Workspace-only tool.

Replicates professor-style grid search: for each (speed kph, distance m), synthesize a
pass-by from the first 1 s of a recording (dedopplerized + repeat-stitched), compare
STFT magnitude spectrograms to the original, and export heatmaps + L1/L2 marginals.

CLI:
  python -m workspace.quadratic_acceleration.analysis_by_synthesis_grid --audio path/to.wav --out_dir results/kia_grid
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Any

import librosa
import numpy as np

# Project root on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from workspace.quadratic_acceleration.abs_synthesis import (
    dedopplerize_far_field_segment,
    find_peak_time,
    stitch_repeat_segment,
    synthesize_passby_straight,
)

# STFT params (match visualization/plot_utils.save_spectrogram_to_file)
N_FFT = 4096
HOP_LENGTH = 64
WIN_LENGTH = 4096
WINDOW = "hann"

# Analysis sample rate (prof / user spec)
ANALYSIS_SR = 44100

SOURCE_DURATION_S = 1.0
VELOCITIES_KPH = list(range(30, 95, 5))  # 30..90
DISTANCES_M = [round(1.0 + 0.5 * i, 1) for i in range(13)]  # 1.0..7.0


def load_mono_audio(path: str, sr: int = ANALYSIS_SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def magnitude_stft(y: np.ndarray, sr: int) -> np.ndarray:
    S = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        center=True,
    )
    return np.abs(S).astype(np.float32)


def normalize_spec(S: np.ndarray, mode: str = "global_max") -> np.ndarray:
    S = np.asarray(S, dtype=np.float32)
    if mode == "global_max":
        ref = float(np.max(S)) + 1e-12
        return S / ref
    if mode == "l2":
        norm = float(np.linalg.norm(S)) + 1e-12
        return S / norm
    return S


def align_spec_time(S_ref: np.ndarray, S_gen: np.ndarray) -> np.ndarray:
    """Trim/pad generated STFT columns to match reference width."""
    if S_gen.shape[1] == S_ref.shape[1]:
        return S_gen
    if S_gen.shape[1] > S_ref.shape[1]:
        return S_gen[:, : S_ref.shape[1]]
    pad = np.zeros((S_gen.shape[0], S_ref.shape[1] - S_gen.shape[1]), dtype=S_gen.dtype)
    return np.concatenate([S_gen, pad], axis=1)


def cpa_frame_mask(n_frames: int, sr: int, hop: int, peak_s: float, window_s: float) -> np.ndarray:
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)
    half = window_s / 2.0
    return (np.abs(times - peak_s) <= half).astype(bool)


def compute_errors(S_ref_n: np.ndarray, S_gen_n: np.ndarray, mask: np.ndarray | None) -> dict:
    if mask is not None and np.any(mask):
        ref = S_ref_n[:, mask]
        gen = S_gen_n[:, mask]
    else:
        ref = S_ref_n
        gen = S_gen_n
    diff = ref - gen
    l1 = float(np.sum(np.abs(diff)))
    l2 = float(np.linalg.norm(diff, ord="fro"))
    l2_power = float(np.linalg.norm(ref**2 - gen**2, ord="fro"))
    return {"l1": l1, "l2": l2, "l2_power": l2_power}


def build_grid_vectors(
    v_min: float = 30,
    v_max: float = 90,
    v_step: float = 5,
    d_min: float = 1.0,
    d_max: float = 7.0,
    d_step: float = 0.5,
) -> tuple[list[float], list[float]]:
    velocities = []
    v = v_min
    while v <= v_max + 1e-9:
        velocities.append(round(v, 1))
        v += v_step
    distances = []
    d = d_min
    while d <= d_max + 1e-9:
        distances.append(round(d, 1))
        d += d_step
    return velocities, distances


def save_error_csv(path: str, velocities: list[float], distances: list[float], grid: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["v_kph \\ d_m"] + [f"{d:.1f}" for d in distances])
        for i, v in enumerate(velocities):
            w.writerow([f"{v:.1f}"] + [f"{grid[i, j]:.8f}" for j in range(len(distances))])


def plot_heatmap(
    grid: np.ndarray,
    velocities: list[float],
    distances: list[float],
    title: str,
    out_path: str,
    *,
    best_v: float | None = None,
    best_d: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    # extent: [x0, x1, y0, y1] with origin='upper' -> y decreases upward
    extent = [
        distances[0] - 0.25,
        distances[-1] + 0.25,
        velocities[-1] + 2.5,
        velocities[0] - 2.5,
    ]
    im = ax.imshow(grid, aspect="auto", origin="upper", extent=extent, cmap="viridis")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Velocity (kph)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Error")
    if best_v is not None and best_d is not None:
        ax.plot(best_d, best_v, "r*", markersize=14, label=f"argmin ({best_v:.0f} kph, {best_d:.1f} m)")
        ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_marginals(
    grid: np.ndarray,
    velocities: list[float],
    distances: list[float],
    metric_label: str,
    out_path: str,
) -> dict:
    """1D marginals matching professor reference style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    min_over_v = np.min(grid, axis=0)
    min_over_d = np.min(grid, axis=1)
    best_d_idx = int(np.argmin(min_over_v))
    best_v_idx = int(np.argmin(min_over_d))
    best_d = distances[best_d_idx]
    best_v = velocities[best_v_idx]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")

    ax = axes[0]
    ax.plot(distances, min_over_v, "o-", color="#1f77b4", linewidth=1.5, markersize=6)
    ax.axvline(best_d, color="#d62728", linestyle="--", linewidth=1.2, label=f"best d = {best_d:.1f} m")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel(f"Min {metric_label} (over v)")
    ax.set_title("Min error vs distance")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.plot(velocities, min_over_d, "o-", color="#1f77b4", linewidth=1.5, markersize=6)
    ax.axvline(best_v, color="#d62728", linestyle="--", linewidth=1.2, label=f"best v = {best_v:.0f} kph")
    ax.set_xlabel("Velocity (kph)")
    ax.set_ylabel(f"Min {metric_label} (over d)")
    ax.set_title("Min error vs velocity")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {"best_v_kph": best_v, "best_d_m": best_d, "min_over_v": min_over_v, "min_over_d": min_over_d}


def flat_ridge_report(grid: np.ndarray, tol_frac: float = 0.01) -> dict:
    gmin = float(np.min(grid))
    mask = grid <= gmin * (1.0 + tol_frac)
    return {
        "global_min": gmin,
        "tol_frac": tol_frac,
        "n_within_tol": int(np.sum(mask)),
        "frac_within_tol": float(np.sum(mask) / grid.size),
        "is_broad_ridge": bool(np.sum(mask) > 3),
    }


def run_analysis_by_synthesis_grid(
    audio_path: str,
    out_dir: str,
    *,
    cpa_windows: list[float] | None = None,
    metric: str = "both",
    norm_mode: str = "global_max",
    temp_c: float = 20.0,
    humidity: float = 50.0,
    save_wavs: bool = False,
    synthetic_gt: dict | None = None,
    v_range: tuple[float, float, float] | None = None,
    d_range: tuple[float, float, float] | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    """
    Run full (v,d) grid. Returns summary dict with paths and argmins.
    """
    os.makedirs(out_dir, exist_ok=True)
    if cpa_windows is None:
        cpa_windows = [0.5, 1.0, 2.0]

    if v_range or d_range:
        v0, v1, vs = v_range if v_range else (30.0, 90.0, 5.0)
        d0, d1, ds = d_range if d_range else (1.0, 7.0, 0.5)
        velocities, distances = build_grid_vectors(v0, v1, vs, d0, d1, ds)
    else:
        velocities, distances = build_grid_vectors()

    original = load_mono_audio(audio_path, ANALYSIS_SR)
    duration_s = len(original) / ANALYSIS_SR
    n_src = int(round(SOURCE_DURATION_S * ANALYSIS_SR))
    source_seg = original[: min(n_src, len(original))].copy()
    peak_time_s = find_peak_time(original, ANALYSIS_SR, HOP_LENGTH)

    S_ref = magnitude_stft(original, ANALYSIS_SR)
    S_ref_n = normalize_spec(S_ref, norm_mode)

    n_v, n_d = len(velocities), len(distances)
    err_l1_full = np.zeros((n_v, n_d), dtype=np.float64)
    err_l2_full = np.zeros((n_v, n_d), dtype=np.float64)
    err_l1_cpa: dict[float, np.ndarray] = {w: np.zeros((n_v, n_d)) for w in cpa_windows}
    err_l2_cpa: dict[float, np.ndarray] = {w: np.zeros((n_v, n_d)) for w in cpa_windows}

    method_notes = {
        "synthesis": "DopplerSim straight-line: calculate_straight_line_doppler + apply_doppler_to_audio_fixed",
        "dedopplerize": "f_emit = f_obs * (c + v_r) / c, constant v_r = v_mps (far-field CV)",
        "stitch": "repeat 1 s segment to clip length (periodic copies possible)",
        "stft": {"n_fft": N_FFT, "hop_length": HOP_LENGTH, "win_length": WIN_LENGTH, "window": WINDOW},
        "normalization": norm_mode,
        "analysis_sr": ANALYSIS_SR,
        "source_duration_s": SOURCE_DURATION_S,
        "cpa_peak_time_s": peak_time_s,
        "preprocessing": "resample/mono only; no denoise/EQ on source",
    }

    wav_dir = os.path.join(out_dir, "generated_wavs")
    if save_wavs:
        os.makedirs(wav_dir, exist_ok=True)

    total = n_v * n_d
    step_i = 0
    for i, v_kph in enumerate(velocities):
        v_mps = v_kph / 3.6
        dedoped = dedopplerize_far_field_segment(
            source_seg, ANALYSIS_SR, v_mps, temp_c=temp_c, humidity=humidity
        )
        stitched = stitch_repeat_segment(dedoped, len(original))

        for j, d_m in enumerate(distances):
            step_i += 1
            if progress_callback:
                progress_callback(step_i, total, v_kph, d_m)

            gen = synthesize_passby_straight(
                stitched,
                ANALYSIS_SR,
                v_mps,
                d_m,
                duration_s,
                cpa_time_s=duration_s / 2.0,
                angle_deg=0.0,
                temp_c=temp_c,
                humidity=humidity,
                target_peak_s=peak_time_s,
            )

            if save_wavs:
                import soundfile as sf

                sf.write(
                    os.path.join(wav_dir, f"v{int(v_kph)}_d{d_m:.1f}.wav"),
                    gen,
                    ANALYSIS_SR,
                )

            S_gen = magnitude_stft(gen, ANALYSIS_SR)
            S_gen = align_spec_time(S_ref, S_gen)
            S_gen_n = normalize_spec(S_gen, norm_mode)

            e_full = compute_errors(S_ref_n, S_gen_n, mask=None)
            err_l1_full[i, j] = e_full["l1"]
            err_l2_full[i, j] = e_full["l2"]

            n_frames = S_ref_n.shape[1]
            for w in cpa_windows:
                mask = cpa_frame_mask(n_frames, ANALYSIS_SR, HOP_LENGTH, peak_time_s, w)
                e_cpa = compute_errors(S_ref_n, S_gen_n, mask=mask)
                err_l1_cpa[w][i, j] = e_cpa["l1"]
                err_l2_cpa[w][i, j] = e_cpa["l2"]

    results: dict[str, Any] = {
        "audio_path": audio_path,
        "out_dir": out_dir,
        "velocities_kph": velocities,
        "distances_m": distances,
        "method_notes": method_notes,
        "peak_time_s": peak_time_s,
        "duration_s": duration_s,
        "grid_points": total,
    }

    metrics_to_run = []
    if metric in ("both", "l1", "l2"):
        if metric in ("both", "l2"):
            metrics_to_run.append("l2")
        if metric in ("both", "l1"):
            metrics_to_run.append("l1")
    else:
        metrics_to_run = ["l2", "l1"]

    for m in metrics_to_run:
        grid_full = err_l2_full if m == "l2" else err_l1_full
        label = "L2" if m == "l2" else "L1"
        csv_path = os.path.join(out_dir, f"error_{m}_full.csv")
        save_error_csv(csv_path, velocities, distances, grid_full)

        marg = plot_marginals(
            grid_full,
            velocities,
            distances,
            label,
            os.path.join(out_dir, f"marginals_{m}_full.png"),
        )
        plot_heatmap(
            grid_full,
            velocities,
            distances,
            f"{label} error (full clip)",
            os.path.join(out_dir, f"heatmap_{m}_full.png"),
            best_v=marg["best_v_kph"],
            best_d=marg["best_d_m"],
        )
        results[f"argmin_{m}_full"] = marg
        results[f"ridge_{m}_full"] = flat_ridge_report(grid_full)

        for w in cpa_windows:
            grid_cpa = err_l2_cpa[w] if m == "l2" else err_l1_cpa[w]
            tag = f"{m}_cpa{w:.1f}s"
            save_error_csv(os.path.join(out_dir, f"error_{tag}.csv"), velocities, distances, grid_cpa)
            marg_c = plot_marginals(
                grid_cpa,
                velocities,
                distances,
                label,
                os.path.join(out_dir, f"marginals_{tag}.png"),
            )
            plot_heatmap(
                grid_cpa,
                velocities,
                distances,
                f"{label} error (CPA window {w:.1f} s)",
                os.path.join(out_dir, f"heatmap_{tag}.png"),
                best_v=marg_c["best_v_kph"],
                best_d=marg_c["best_d_m"],
            )
            results[f"argmin_{tag}"] = marg_c
            results[f"ridge_{tag}"] = flat_ridge_report(grid_cpa)

    # Optional diagnostic: best L2 vs best L1 spectrograms
    try:
        import matplotlib.pyplot as plt

        b2 = results.get("argmin_l2_full", {})
        b1 = results.get("argmin_l1_full", {})
        if b2 and b1:
            vi2 = velocities.index(b2["best_v_kph"]) if b2["best_v_kph"] in velocities else 0
            di2 = distances.index(b2["best_d_m"]) if b2["best_d_m"] in distances else 0
            vi1 = velocities.index(b1["best_v_kph"]) if b1["best_v_kph"] in velocities else 0
            di1 = distances.index(b1["best_d_m"]) if b1["best_d_m"] in distances else 0

            def _regen(v_kph, d_m):
                v_mps = v_kph / 3.6
                ded = dedopplerize_far_field_segment(source_seg, ANALYSIS_SR, v_mps, temp_c=temp_c, humidity=humidity)
                st = stitch_repeat_segment(ded, len(original))
                return synthesize_passby_straight(
                    st, ANALYSIS_SR, v_mps, d_m, duration_s,
                    temp_c=temp_c, humidity=humidity, target_peak_s=peak_time_s,
                )

            gen_l2 = _regen(velocities[vi2], distances[di2])
            gen_l1 = _regen(velocities[vi1], distances[di1])

            fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="white")
            for ax, y, title in zip(
                axes,
                [original, gen_l2, gen_l1],
                ["Original", f"Best L2 (v={velocities[vi2]:.0f}, d={distances[di2]:.1f})",
                 f"Best L1 (v={velocities[vi1]:.0f}, d={distances[di1]:.1f})"],
            ):
                S = magnitude_stft(y, ANALYSIS_SR)
                Sn = normalize_spec(S, norm_mode)
                ax.imshow(
                    librosa.amplitude_to_db(Sn, ref=np.max),
                    aspect="auto",
                    origin="lower",
                    cmap="magma",
                )
                ax.set_title(title)
                ax.set_xlabel("Frame")
                ax.set_ylabel("Bin")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "overlay_original_best_l2_l1.png"), dpi=150, facecolor="white")
            plt.close(fig)
            results["overlay_png"] = "overlay_original_best_l2_l1.png"
    except Exception as exc:
        results["overlay_error"] = str(exc)

    if synthetic_gt:
        gt_v = float(synthetic_gt.get("speed_kph", synthetic_gt.get("v_kph", 50)))
        gt_d = float(synthetic_gt.get("distance_m", synthetic_gt.get("d_m", 4.5)))
        vi = min(range(n_v), key=lambda i: abs(velocities[i] - gt_v))
        di = min(range(n_d), key=lambda j: abs(distances[j] - gt_d))
        cal = {}
        for key, grid in [("l2_full", err_l2_full), ("l1_full", err_l1_full)]:
            gmin = float(np.min(grid))
            within = grid <= gmin * 1.01
            cal[key] = {
                "gt_v_kph": gt_v,
                "gt_d_m": gt_d,
                "nearest_grid_v": velocities[vi],
                "nearest_grid_d": distances[di],
                "gt_cell_error": float(grid[vi, di]),
                "global_min": gmin,
                "gt_within_1pct_contour": bool(within[vi, di]),
            }
        results["synthetic_calibration"] = cal

    readme = os.path.join(out_dir, "README_abs_grid.md")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(_readme_text(results, method_notes))
    results["readme"] = readme

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(results), f, indent=2)
    results["summary_json"] = summary_path

    return results


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


def _readme_text(results: dict, notes: dict) -> str:
    lines = [
        "# Analysis-by-synthesis (v, d) grid",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Method",
        f"- Synthesis: {notes.get('synthesis')}",
        f"- Dedopplerize: {notes.get('dedopplerize')}",
        f"- Stitch: {notes.get('stitch')}",
        f"- STFT: {notes.get('stft')}",
        f"- Normalization: {notes.get('normalization')}",
        f"- CPA peak (original RMS): {notes.get('cpa_peak_time_s', '?')} s",
        "",
        "## Argmins (full clip)",
    ]
    for key in ("argmin_l2_full", "argmin_l1_full"):
        if key in results:
            m = results[key]
            lines.append(f"- **{key}**: v = {m.get('best_v_kph')} kph, d = {m.get('best_d_m')} m")
    lines.append("")
    lines.append("## Broad valley")
    for key in ("ridge_l2_full", "ridge_l1_full"):
        if key in results:
            r = results[key]
            lines.append(
                f"- {key}: {r.get('n_within_tol')} cells within 1% of minimum "
                f"({100 * r.get('frac_within_tol', 0):.1f}% of grid)"
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "- L2 often shows a **flat distance valley** for d ≳ 3 m; argmin distance is weakly constrained."
    )
    lines.append(
        "- L1 can show a **sharper distance minimum** near d ≈ 2 m with a flatter velocity band ~45–65 kph."
    )
    lines.append("- Treat argmins as points in a flat region, not ground truth, unless CPA-windowed metrics sharpen the minimum.")
    if "synthetic_calibration" in results:
        lines.append("")
        lines.append("## Synthetic ground truth")
        lines.append(f"```json\n{json.dumps(results['synthetic_calibration'], indent=2)}\n```")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Analysis-by-synthesis (v,d) grid (Workspace tool)")
    p.add_argument("--audio", required=True, help="Path to original pass-by WAV")
    p.add_argument("--out_dir", default="static/workspace_outputs/abs_grid", help="Output directory")
    p.add_argument("--metric", choices=["both", "l1", "l2"], default="both")
    p.add_argument("--cpa_window", type=float, default=1.0, help="Primary CPA window (s); also runs 0.5 and 2.0")
    p.add_argument("--norm", default="global_max", choices=["global_max", "l2"])
    p.add_argument("--save_wavs", action="store_true")
    p.add_argument("--gt_v_kph", type=float, default=None, help="Synthetic GT speed (kph) for calibration")
    p.add_argument("--gt_d_m", type=float, default=None, help="Synthetic GT distance (m) for calibration")
    args = p.parse_args()

    cpa_windows = sorted(set([0.5, 1.0, 2.0, args.cpa_window]))
    gt = None
    if args.gt_v_kph is not None and args.gt_d_m is not None:
        gt = {"speed_kph": args.gt_v_kph, "distance_m": args.gt_d_m}

    def prog(i, t, v, d):
        print(f"[{i}/{t}] v={v} kph, d={d} m")

    results = run_analysis_by_synthesis_grid(
        args.audio,
        args.out_dir,
        cpa_windows=cpa_windows,
        metric=args.metric,
        norm_mode=args.norm,
        save_wavs=args.save_wavs,
        synthetic_gt=gt,
        progress_callback=prog,
    )
    print(json.dumps(_json_safe({k: results[k] for k in results if k.startswith("argmin")}), indent=2))
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
