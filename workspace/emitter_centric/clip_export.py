"""Batch-style clip export for emitter-centric workspace (isolated)."""

from __future__ import annotations

import os
import shutil
from typing import Any

import numpy as np

from audio.audio_utils import SR, save_audio
from audio.generation import save_numpy_outputs
from visualization.plot_utils import save_path_plot
from workspace.emitter_centric.artifacts import save_scene_sidecars
from workspace.emitter_centric.config import DEFAULT_DURATION_S


def _infer_direction(path_type: str, params: dict) -> str:
    angle = float(params.get("angle", 0))
    if path_type != "straight":
        return "path"
    if angle < 45 or angle > 315:
        return "left_to_right"
    if 135 < angle < 225:
        return "right_to_left"
    return "lateral"


def export_clip_artifacts(
    audio: np.ndarray,
    *,
    branch_dir: str,
    sample_index: int,
    vehicle_name: str,
    path_type: str,
    params: dict,
    config: dict,
    branch_label: str,
    branch: str,
    doppler_arrays: tuple[np.ndarray, np.ndarray] | None = None,
    emitter_frame_meta: dict | None = None,
) -> dict[str, Any]:
    """
    Write sample_XXXXXXX/Common + Essential layout matching batch generation.
    """
    index = int(sample_index)
    sample_dir = os.path.join(branch_dir, "audio_clips", f"sample_{index:07d}")
    common_dir = os.path.join(sample_dir, "Common")
    essential_dir = os.path.join(sample_dir, "Essential")
    os.makedirs(common_dir, exist_ok=True)
    os.makedirs(essential_dir, exist_ok=True)

    direction = _infer_direction(path_type, params)
    meta_name = (
        f"{vehicle_name}_{path_type}_{direction}_{params['speed']:.4g}mps_"
        f"{params['distance']:.4g}m_{index:07d}_{branch_label}"
    )
    output_format = config.get("output", {}).get("format", "wav")
    filename = f"{meta_name}.{output_format}"
    if output_format == "mp3":
        filename = filename.replace(".mp3", ".wav")

    for d in (common_dir, essential_dir):
        save_audio(audio, os.path.join(d, filename))

    spec_cfg = {
        "output": {
            "spectrogram_type": config.get("output", {}).get("spectrogram_type", "cqt"),
            "generate_diagnostics": config.get("output", {}).get("generate_diagnostics", True),
        }
    }
    features = save_numpy_outputs(
        audio,
        sample_dir,
        spectrogram_type=spec_cfg["output"]["spectrogram_type"],
        config=spec_cfg,
        base_name=meta_name,
        essential_dir=essential_dir,
        params=params,
    )

    plot_params = dict(params)
    plot_params.setdefault("duration", params.get("duration", DEFAULT_DURATION_S))
    plot_file = save_path_plot(path_type, plot_params, common_dir, meta_name)
    if plot_file:
        src_plot = os.path.join(common_dir, plot_file)
        dst_plot = os.path.join(essential_dir, plot_file)
        if os.path.abspath(src_plot) != os.path.abspath(dst_plot):
            shutil.copy2(src_plot, dst_plot)

    sidecars = save_scene_sidecars(
        sample_dir=sample_dir,
        common_dir=common_dir,
        essential_dir=essential_dir,
        path_type=path_type,
        params=params,
        branch=branch,
        doppler_arrays=doppler_arrays,
        emitter_frame=emitter_frame_meta,
    )

    wav_common = os.path.join(common_dir, filename)
    return {
        "filename": filename,
        "index": index,
        "vehicle": vehicle_name,
        "path_type": path_type,
        "parameters": params,
        "path_plot": plot_file or f"{meta_name}.png",
        "sample_dir": f"sample_{index:07d}",
        "wav_common": wav_common.replace("\\", "/"),
        "branch": branch,
        "sidecars": sidecars,
        "features": {k: "saved" for k in features.keys()} if isinstance(features, dict) else {},
    }
