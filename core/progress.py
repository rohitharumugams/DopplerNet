import os
import json

from core.config import PROGRESS_FILE


def _default_progress():
    return {
        "total_target": 0,
        "generated_so_far": 0,
        "phase": "idle",
        "status": "idle",
    }


def load_progress(batch_dir=None):
    """Load global progress; prefer batch_dir/progress.json when present."""
    if batch_dir:
        batch_progress = load_batch_progress(batch_dir)
        if batch_progress:
            return batch_progress

    if not os.path.exists(PROGRESS_FILE):
        return _default_progress()
    with open(PROGRESS_FILE, "r") as f:
        data = json.load(f)
    for key, default in _default_progress().items():
        data.setdefault(key, default)
    return data


def load_batch_progress(batch_dir):
    path = os.path.join(batch_dir, "progress.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    for key, default in _default_progress().items():
        data.setdefault(key, default)
    return data


def save_progress(
    total_target,
    generated_so_far,
    batch_dir=None,
    *,
    phase=None,
    status=None,
    batch_directory=None,
    batch_id=None,
    wave_start=None,
    error=None,
    metadata_file=None,
    log_file=None,
    stats_file=None,
    formatted_time=None,
    slots_failed=None,
):
    payload = {
        "total_target": int(total_target),
        "generated_so_far": int(generated_so_far),
        "phase": phase or "synthesis",
        "status": status or "running",
    }
    if batch_directory:
        payload["batch_directory"] = batch_directory
    if batch_id:
        payload["batch_id"] = batch_id
    if wave_start is not None:
        payload["wave_start"] = int(wave_start)
    if error:
        payload["error"] = str(error)
    if metadata_file:
        payload["metadata_file"] = metadata_file
    if log_file:
        payload["log_file"] = log_file
    if stats_file:
        payload["stats_file"] = stats_file
    if formatted_time:
        payload["formatted_time"] = formatted_time
    if slots_failed is not None:
        payload["slots_failed"] = int(slots_failed)

    with open(PROGRESS_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    if batch_dir:
        os.makedirs(batch_dir, exist_ok=True)
        with open(os.path.join(batch_dir, "progress.json"), "w") as f:
            json.dump(payload, f, indent=2)
