import json
import os

from core.config import PROGRESS_FILE


def load_progress(batch_dir=None):
    if batch_dir:
        batch_path = os.path.join(batch_dir, 'progress.json')
        if os.path.exists(batch_path):
            with open(batch_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    if not os.path.exists(PROGRESS_FILE):
        return {"total_target": 0, "generated_so_far": 0}
    with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_progress(
    total_target,
    generated_so_far,
    *,
    batch_dir=None,
    batch_id=None,
    phase=None,
    status=None,
    metadata_file=None,
    log_file=None,
    stats_file=None,
    formatted_time=None,
    slots_failed=None,
):
    payload = {
        "total_target": int(total_target),
        "generated_so_far": int(generated_so_far),
    }
    if batch_id is not None:
        payload["batch_id"] = batch_id
    if phase is not None:
        payload["phase"] = phase
    if status is not None:
        payload["status"] = status
    if metadata_file is not None:
        payload["metadata_file"] = metadata_file
    if log_file is not None:
        payload["log_file"] = log_file
    if stats_file is not None:
        payload["stats_file"] = stats_file
    if formatted_time is not None:
        payload["formatted_time"] = formatted_time
    if slots_failed is not None:
        payload["slots_failed"] = int(slots_failed)

    def _write_atomic(path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    _write_atomic(PROGRESS_FILE)

    if batch_dir:
        os.makedirs(batch_dir, exist_ok=True)
        _write_atomic(os.path.join(batch_dir, 'progress.json'))
