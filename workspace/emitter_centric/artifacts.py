"""Extra per-clip and per-batch artifacts (emitter-centric workspace only)."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from physics.source_position import compute_source_position_track, save_source_positions_npy
from workspace.emitter_centric.source_frame import speed_profile


def save_scene_sidecars(
    *,
    sample_dir: str,
    common_dir: str,
    essential_dir: str,
    path_type: str,
    params: dict,
    branch: str,
    doppler_arrays: tuple[np.ndarray, np.ndarray] | None = None,
    emitter_frame: dict | None = None,
) -> list[str]:
    """Return list of relative artifact paths written under sample_dir."""
    written: list[str] = []
    duration = float(params.get("duration", 10.0))

    track = compute_source_position_track(path_type, params, duration)
    save_source_positions_npy(track, common_dir, essential_dir)
    written.append("Common/source_positions.npy")

    if branch == "observer_centric" and doppler_arrays is not None:
        fr, amp = doppler_arrays
        for folder in (common_dir, essential_dir):
            np.save(os.path.join(folder, "freq_ratios.npy"), np.asarray(fr, dtype=np.float32))
            np.save(os.path.join(folder, "amplitudes.npy"), np.asarray(amp, dtype=np.float32))
        written.extend(["Common/freq_ratios.npy", "Common/amplitudes.npy"])

    if branch == "emitter_centric" and emitter_frame is not None:
        n = int(round(duration * SR))
        t, v = speed_profile(
            n,
            duration,
            float(emitter_frame.get("speed_mps", params.get("speed", 0))),
            float(emitter_frame.get("acceleration_mps2", params.get("acceleration", 0))),
        )
        frame_kin = np.column_stack([
            t,
            v,
            np.full_like(t, float(params.get("acceleration", 0)), dtype=np.float32),
        ]).astype(np.float32)
        for folder in (common_dir, essential_dir):
            np.save(os.path.join(folder, "emitter_frame_kinematics.npy"), frame_kin)
        written.append("Common/emitter_frame_kinematics.npy")

        if emitter_frame.get("rpm_coupling"):
            v_ref = max(float(emitter_frame.get("v_ref_mps", 1.0)), 0.1)
            pitch = (v / v_ref).astype(np.float32)
            for folder in (common_dir, essential_dir):
                np.save(os.path.join(folder, "src_pitch_curve.npy"), pitch)
            written.append("Common/src_pitch_curve.npy")

    meta_path = os.path.join(sample_dir, "sample_metadata.json")
    payload = {
        "branch": branch,
        "path_type": path_type,
        "parameters": params,
        "scene_note": (
            "source_positions.npy is observer-frame scene geometry (Category A). "
            "Not co-moving microphone position."
        ),
        "emitter_frame_note": (
            "emitter_frame_kinematics.npy is along-track speed on source clock (emitter branch only)."
        ),
        "timestamp": datetime.now().isoformat(),
    }
    if doppler_arrays is not None and branch == "observer_centric":
        fr, _ = doppler_arrays
        payload["doppler_ratio_range"] = {
            "min": float(np.min(fr)),
            "max": float(np.max(fr)),
        }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    written.append("sample_metadata.json")
    return written


def clip_row_for_dataset(batch_id: str, clip_info: dict, branch: str) -> dict:
    """One row for emitter-centric dataset.csv (no CPA/pass-by benchmark columns)."""
    p = clip_info.get("parameters") or {}
    return {
        "sample_id": clip_info.get("sample_dir", ""),
        "batch_id": batch_id,
        "branch": branch,
        "filename": clip_info.get("filename", ""),
        "vehicle": clip_info.get("vehicle", ""),
        "path_type": clip_info.get("path_type", ""),
        "speed_mps": p.get("speed", ""),
        "acceleration_mps2": p.get("acceleration", ""),
        "scene_lateral_offset_m": p.get("distance", p.get("h", "")),
        "angle_deg": p.get("angle", ""),
        "duration_s": p.get("duration", ""),
        "temperature_c": p.get("temperature", ""),
        "humidity_pct": p.get("humidity", ""),
    }


def write_dataset_csv(path: str, batch_id: str, rows: list[dict]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def generate_emitter_batch_statistics(
    clips: list[dict],
    *,
    batch_id: str,
    compare_observer: bool,
) -> str:
    """Text summary analogous to batch statistics_{batch_id}.txt without CPA/pass-by blocks."""
    lines = [
        "=" * 60,
        "EMITTER-CENTRIC WORKSPACE BATCH STATISTICS",
        "=" * 60,
        "",
        f"Batch ID: {batch_id}",
        f"Total clips indexed: {len(clips)}",
        f"Observer comparison branch enabled: {compare_observer}",
        "",
    ]
    if not clips:
        lines.append("No clips — statistics unavailable.")
        return "\n".join(lines)

    vehicles: dict[str, int] = {}
    paths: dict[str, int] = {}
    speeds: list[float] = []
    accels: list[float] = []
    offsets: list[float] = []
    for c in clips:
        vehicles[c.get("vehicle", "unknown")] = vehicles.get(c.get("vehicle", "unknown"), 0) + 1
        paths[c.get("path_type", "unknown")] = paths.get(c.get("path_type", "unknown"), 0) + 1
        p = c.get("parameters") or {}
        if "speed" in p:
            speeds.append(float(p["speed"]))
        if "acceleration" in p:
            accels.append(float(p["acceleration"]))
        if "distance" in p:
            offsets.append(float(p["distance"]))

    lines.append("Vehicle distribution:")
    for v, n in sorted(vehicles.items()):
        lines.append(f"  {v}: {n}")
    lines.append("")
    lines.append("Path type distribution:")
    for p, n in sorted(paths.items()):
        lines.append(f"  {p}: {n}")
    lines.append("")

    def _fmt(label: str, vals: list[float], unit: str) -> None:
        if not vals:
            lines.append(f"{label}: N/A")
            return
        lines.append(f"{label}:")
        lines.append(f"  Min: {min(vals):.2f} {unit}")
        lines.append(f"  Max: {max(vals):.2f} {unit}")
        lines.append(f"  Mean: {np.mean(vals):.2f} {unit}")

    _fmt("Speed", speeds, "m/s")
    lines.append("")
    _fmt("Acceleration", accels, "m/s²")
    lines.append("")
    _fmt("Scene lateral offset (observer-frame geometry)", offsets, "m")
    lines.append("")
    lines.append(
        "Excluded vs standard Batch Generation: B1–B10 benchmark folders, "
        "CPA-time/pass-by labels derived from listener audio, dataset.csv CPA columns."
    )
    return "\n".join(lines)
