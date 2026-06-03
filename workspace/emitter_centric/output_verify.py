"""Verify emitter-centric batch folders on disk (workspace only)."""

from __future__ import annotations

import os
from typing import Any


def _is_file(path: str | None) -> bool:
    if not path:
        return False
    return os.path.isfile(path.replace("/", os.sep))


def verify_clip_outputs(
    batch_root: str,
    clip: dict[str, Any],
    *,
    compare_observer: bool,
) -> list[str]:
    """Return list of issues; empty means clip outputs look complete."""
    issues: list[str] = []
    idx = int(clip.get("index", 0))
    sample_name = f"sample_{idx:07d}"

    emit = clip.get("emitter") or {}
    emit_sample = os.path.join(
        batch_root, "emitter_centric", "audio_clips", emit.get("sample_dir", sample_name)
    )
    if not os.path.isdir(emit_sample):
        issues.append(f"missing emitter sample dir: {emit_sample}")
    else:
        common = os.path.join(emit_sample, "Common")
        if not os.path.isdir(common):
            issues.append(f"missing emitter Common/: {common}")
        elif not _is_file(emit.get("wav_common")):
            wavs = [f for f in os.listdir(common) if f.lower().endswith(".wav")]
            if not wavs:
                issues.append(f"no WAV in {common}")

    if not compare_observer:
        return issues

    obs = clip.get("observer")
    if not obs:
        issues.append("observer branch record missing from clip metadata")
    else:
        obs_sample = os.path.join(
            batch_root, "observer_centric", "audio_clips", obs.get("sample_dir", sample_name)
        )
        if not os.path.isdir(obs_sample):
            issues.append(f"missing observer sample dir: {obs_sample}")
        elif not _is_file(obs.get("wav_common")):
            common = os.path.join(obs_sample, "Common")
            if not os.path.isdir(common) or not any(
                f.lower().endswith(".wav") for f in os.listdir(common)
            ):
                issues.append(f"no observer WAV under {obs_sample}")

    comp_dir = os.path.join(batch_root, "comparison_outputs", sample_name)
    if not os.path.isdir(comp_dir):
        issues.append(f"missing comparison_outputs/{sample_name}/")
    else:
        for name in ("emitter_centric.wav", "observer_centric.wav"):
            if not os.path.isfile(os.path.join(comp_dir, name)):
                issues.append(f"missing {comp_dir}/{name}")

    return issues


def verify_batch_outputs(
    batch_root: str,
    batch_id: str,
    clips: list[dict[str, Any]],
    *,
    compare_observer: bool,
    total_requested: int,
) -> dict[str, Any]:
    """Verify batch root layout and per-clip artifacts."""
    issues: list[str] = []
    root = batch_root.replace("\\", "/")

    required_root = [
        os.path.join(batch_root, "overall_metadata.json"),
        os.path.join(batch_root, "emitter_centric"),
        os.path.join(batch_root, "emitter_centric", "audio_clips"),
        os.path.join(batch_root, "emitter_centric", f"metadata_{batch_id}.json"),
    ]
    if compare_observer:
        required_root.extend([
            os.path.join(batch_root, "observer_centric"),
            os.path.join(batch_root, "observer_centric", "audio_clips"),
            os.path.join(batch_root, "comparison_outputs"),
        ])

    for path in required_root:
        if os.path.isdir(path):
            continue
        if os.path.isfile(path):
            continue
        issues.append(f"missing batch artifact: {path}")

    if len(clips) != total_requested:
        issues.append(
            f"clip count mismatch: {len(clips)} generated vs {total_requested} requested"
        )

    for clip in clips:
        for msg in verify_clip_outputs(batch_root, clip, compare_observer=compare_observer):
            issues.append(f"sample_{int(clip.get('index', 0)):07d}: {msg}")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "clips_verified": len(clips),
        "batch_root": root,
    }
