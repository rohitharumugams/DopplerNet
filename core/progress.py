import os
import json

from core.config import PROGRESS_FILE


def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {"total_target": 0, "generated_so_far": 0}
    with open(PROGRESS_FILE, "r") as f:
        return json.load(f)


def save_progress(total_target, generated_so_far):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(
            {
                "total_target": total_target,
                "generated_so_far": generated_so_far
            },
            f,
            indent=2
        )
