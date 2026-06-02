"""Single-clip and batch generation for emitter-centric workspace."""

from __future__ import annotations

import json
import os
import shutil
import traceback
from datetime import datetime
from typing import Any, Callable

import numpy as np

from workspace.emitter_centric.artifacts import (
    clip_row_for_dataset,
    generate_emitter_batch_statistics,
    write_dataset_csv,
)
from workspace.emitter_centric.batch_paths import resolve_batch_root
from workspace.emitter_centric.clip_export import export_clip_artifacts
from workspace.emitter_centric.compare_formulations import compare_waveforms
from workspace.emitter_centric.comparison_plots import generate_all_comparisons
from workspace.emitter_centric.config import SR
from workspace.emitter_centric.kinematics import straight_cv_kinematics_with_c
from workspace.emitter_centric.sampling import build_clip_plan
from workspace.emitter_centric.source_frame import speed_profile, synthesize_source_frame_audio
from workspace.emitter_centric.synthesis import load_source_for_emitter


def _params_for_generation(spec: dict) -> dict:
    return {
        "speed": float(spec["speed"]),
        "distance": float(spec["distance"]),
        "angle": float(spec["angle"]),
        "duration": float(spec.get("duration", 10)),
        "acceleration": float(spec.get("acceleration", 0)),
        "temperature": float(spec.get("temperature", 20)),
        "humidity": float(spec.get("humidity", 50)),
        "path_type": spec.get("path_type", "straight"),
    }


def _maybe_apply_air_noise(audio: np.ndarray, config: dict) -> np.ndarray:
    atm = config.get("atmosphere") or {}
    if not bool(atm.get("add_air_noise", False)):
        return audio
    from audio.generation import _apply_subtle_air_noise

    return _apply_subtle_air_noise(
        np.asarray(audio, dtype=np.float32),
        SR,
        float(atm.get("air_noise_strength", 8.0)),
        float(atm.get("air_noise_frequency_hz", 1200.0)),
    )


def _branch_clip_entry(clip_info: dict, *, branch: str, spec: dict) -> dict[str, Any]:
    """One clip record for branch-level metadata_{batch_id}.json."""
    return {
        "index": clip_info["index"],
        "vehicle": clip_info["vehicle"],
        "path_type": clip_info["path_type"],
        "parameters": clip_info.get("parameters") or _params_for_generation(spec),
        "filename": clip_info["filename"],
        "sample_dir": clip_info["sample_dir"],
        "path_plot": clip_info.get("path_plot"),
        "wav_common": clip_info.get("wav_common"),
        "branch": branch,
    }


def _copy_comparison_wavs(
    comp_root: str,
    *,
    emit_info: dict,
    obs_info: dict | None,
) -> dict[str, str]:
    """Copy branch WAVs into comparison_outputs/sample_XXXXXXX/."""
    os.makedirs(comp_root, exist_ok=True)
    paths: dict[str, str] = {}
    emit_src = emit_info.get("wav_common")
    if emit_src and os.path.isfile(emit_src):
        dst = os.path.join(comp_root, "emitter_centric.wav")
        shutil.copy2(emit_src, dst)
        paths["emitter_centric_wav"] = dst.replace("\\", "/")
    if obs_info:
        obs_src = obs_info.get("wav_common")
        if obs_src and os.path.isfile(obs_src):
            dst = os.path.join(comp_root, "observer_centric.wav")
            shutil.copy2(obs_src, dst)
            paths["observer_centric_wav"] = dst.replace("\\", "/")
    return paths


def synthesize_observer_branch(
    vehicle: str,
    path_type: str,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Production observer-centric audio (read-only call into generation)."""
    from audio.generation import get_doppler_audio_array

    p = dict(params)
    p.setdefault("h", p.get("distance"))
    p.setdefault("angle", p.get("angle", 0))
    p.setdefault("duration", p.get("duration", 10))
    audio, freq_ratios, amplitudes = get_doppler_audio_array(vehicle, path_type, p)
    return (
        np.asarray(audio, dtype=np.float32),
        np.asarray(freq_ratios, dtype=np.float32),
        np.asarray(amplitudes, dtype=np.float32),
    )


def generate_one_clip(
    batch_root: str,
    spec: dict,
    config: dict,
    *,
    compare_observer: bool,
) -> dict[str, Any]:
    """Generate emitter clip; optional observer branch + comparison."""
    index = int(spec["index"])
    vehicle = spec["vehicle"]
    path_type = spec["path_type"]
    params = _params_for_generation(spec)
    duration = params["duration"]

    emit_root = os.path.join(batch_root, "emitter_centric")
    obs_root = os.path.join(batch_root, "observer_centric")
    comp_root = os.path.join(batch_root, "comparison_outputs", f"sample_{index:07d}")

    source = load_source_for_emitter(
        audio_path=None,
        vehicle_name=vehicle,
        duration_s=duration,
    )
    source_file = f"{vehicle} (library)"

    y_emit, emit_meta = synthesize_source_frame_audio(
        source,
        duration_s=duration,
        speed_mps=params["speed"],
        acceleration=params["acceleration"],
        enable_rpm_coupling=bool(config.get("enable_rpm_coupling", True)),
    )
    y_emit = _maybe_apply_air_noise(y_emit, config)

    emit_cfg = dict(config)
    emit_cfg["branch"] = "emitter_centric"
    emit_info = export_clip_artifacts(
        y_emit,
        branch_dir=emit_root,
        sample_index=index,
        vehicle_name=vehicle,
        path_type=path_type,
        params=params,
        config=emit_cfg,
        branch_label="emit",
        branch="emitter_centric",
        emitter_frame_meta={
            **emit_meta,
            "rpm_coupling": bool(config.get("enable_rpm_coupling", True)),
        },
    )

    result: dict[str, Any] = {
        "index": index,
        "spec": spec,
        "emitter": emit_info,
        "emitter_meta": emit_meta,
    }

    t_emit, v_emit = speed_profile(
        int(round(duration * SR)),
        duration,
        params["speed"],
        params["acceleration"],
    )
    kin_emit = {"t": t_emit, "v": v_emit}

    if compare_observer:
        y_obs, freq_ratios, _amps = synthesize_observer_branch(vehicle, path_type, params)
        y_obs = _maybe_apply_air_noise(y_obs, config)
        obs_cfg = dict(config)
        obs_cfg["branch"] = "observer_centric"
        obs_info = export_clip_artifacts(
            y_obs,
            branch_dir=obs_root,
            sample_index=index,
            vehicle_name=vehicle,
            path_type=path_type,
            params=params,
            config=obs_cfg,
            branch_label="obs",
            branch="observer_centric",
            doppler_arrays=(freq_ratios, _amps),
        )
        result["observer"] = obs_info

        n = int(round(duration * SR))
        from audio.audio_utils import get_speed_of_sound

        c_sound = float(get_speed_of_sound(params["temperature"], params["humidity"]))
        kin_obs = straight_cv_kinematics_with_c(
            params["speed"],
            params["distance"],
            params["angle"],
            duration,
            n,
            c_sound=c_sound,
            accel_mps2=params["acceleration"],
        )
        metrics = compare_waveforms(y_obs, y_emit)
        result["comparison_metrics"] = metrics
        plot_paths = generate_all_comparisons(
            comp_root,
            y_obs=y_obs,
            y_emit=y_emit,
            params=params,
            vehicle=vehicle,
            source_file=source_file,
            kin_obs=kin_obs,
            kin_emit=kin_emit,
            freq_ratio=freq_ratios if len(freq_ratios) == n else None,
            metrics=metrics,
        )
        wav_paths = _copy_comparison_wavs(
            comp_root, emit_info=emit_info, obs_info=obs_info
        )
        result["comparison_plots"] = {**plot_paths, **wav_paths}

    return result


def _write_branch_metadata(
    branch_dir: str,
    batch_id: str,
    branch: str,
    clip_entries: list[dict],
    *,
    compare_observer: bool,
) -> str:
    """metadata_{batch_id}.json beside audio_clips/ (Batch Generation convention)."""
    os.makedirs(branch_dir, exist_ok=True)
    meta_path = os.path.join(branch_dir, f"metadata_{batch_id}.json")
    payload = {
        "batch_id": batch_id,
        "branch": branch,
        "mode": "emitter_centric_workspace",
        "audio_clips_dir": "audio_clips",
        "total_clips": len(clip_entries),
        "clips": clip_entries,
        "timestamp": datetime.now().isoformat(),
    }
    if branch == "observer_centric" and not compare_observer:
        payload["note"] = "Observer branch not generated for this batch."
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return meta_path.replace("\\", "/")


def run_batch(
    config: dict,
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run full batch under batch_root."""
    batch_id, batch_root = resolve_batch_root(config)
    os.makedirs(batch_root, exist_ok=True)
    os.makedirs(os.path.join(batch_root, "emitter_centric"), exist_ok=True)
    compare = bool(config.get("compare_observer", False))
    if compare:
        os.makedirs(os.path.join(batch_root, "observer_centric"), exist_ok=True)
        os.makedirs(os.path.join(batch_root, "comparison_outputs"), exist_ok=True)

    plan = build_clip_plan(config)
    total = len(plan)
    clips_meta: list[dict] = []
    failed: list[dict] = []

    for i, spec in enumerate(plan, start=1):
        if progress_callback:
            progress_callback(i, total, f"sample_{spec['index']:07d} {spec['vehicle']}")
        try:
            clips_meta.append(
                generate_one_clip(batch_root, spec, config, compare_observer=compare)
            )
        except Exception as exc:
            traceback.print_exc()
            failed.append({"index": spec["index"], "error": str(exc)})

    emit_clips = [
        _branch_clip_entry(c["emitter"], branch="emitter_centric", spec=c["spec"])
        for c in clips_meta
    ]
    emit_meta_path = _write_branch_metadata(
        os.path.join(batch_root, "emitter_centric"),
        batch_id,
        "emitter_centric",
        emit_clips,
        compare_observer=compare,
    )

    obs_meta_path = None
    if compare:
        obs_clips = [
            _branch_clip_entry(c["observer"], branch="observer_centric", spec=c["spec"])
            for c in clips_meta
            if "observer" in c
        ]
        obs_meta_path = _write_branch_metadata(
            os.path.join(batch_root, "observer_centric"),
            batch_id,
            "observer_centric",
            obs_clips,
            compare_observer=True,
        )

    overall_path = os.path.join(batch_root, "overall_metadata.json")
    payload = {
        "batch_id": batch_id,
        "mode": "emitter_centric_workspace",
        "batch_root": batch_root.replace("\\", "/"),
        "total_requested": total,
        "total_generated": len(clips_meta),
        "failed": failed,
        "compare_observer": compare,
        "config": config,
        "clips": clips_meta,
        "branch_metadata": {
            "emitter_centric": emit_meta_path,
            "observer_centric": obs_meta_path,
        },
        "timestamp": datetime.now().isoformat(),
        "first_principles": "workspace/emitter_centric/plan.md#first-principles",
        "layout": {
            "emitter_centric": "audio_clips/sample_*/ + metadata",
            "observer_centric": "audio_clips/sample_*/ + metadata (if comparison enabled)",
            "comparison_outputs": "sample_*/emitter_centric.wav, observer_centric.wav, plots",
        },
    }
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    dataset_rows = [
        clip_row_for_dataset(batch_id, c["emitter"], "emitter_centric")
        for c in clips_meta
    ]
    if compare:
        dataset_rows.extend(
            clip_row_for_dataset(batch_id, c["observer"], "observer_centric")
            for c in clips_meta
            if "observer" in c
        )
    dataset_path = os.path.join(batch_root, "dataset.csv")
    write_dataset_csv(dataset_path, batch_id, dataset_rows)

    stats_path = os.path.join(batch_root, f"statistics_{batch_id}.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(
            generate_emitter_batch_statistics(
                emit_clips,
                batch_id=batch_id,
                compare_observer=compare,
            )
        )

    log_path = os.path.join(batch_root, f"generation_log_{batch_id}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Emitter-centric workspace batch {batch_id}\n")
        f.write(f"Generated: {len(clips_meta)} / {total}\n")
        if failed:
            f.write("Failures:\n")
            for item in failed:
                f.write(f"  sample_{item['index']:07d}: {item['error']}\n")

    return {
        "success": len(failed) == 0,
        "batch_directory": batch_root.replace("\\", "/"),
        "metadata_file": overall_path.replace("\\", "/"),
        "overall_metadata": overall_path.replace("\\", "/"),
        "dataset_csv": dataset_path.replace("\\", "/"),
        "statistics_file": stats_path.replace("\\", "/"),
        "log_file": log_path.replace("\\", "/"),
        "total_generated": len(clips_meta),
        "total_requested": total,
        "failed": failed,
    }
