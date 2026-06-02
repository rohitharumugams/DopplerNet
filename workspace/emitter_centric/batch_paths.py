"""Batch folder naming and layout (emitter-centric workspace only)."""

from __future__ import annotations

import os
import re
from datetime import datetime

from workspace.emitter_centric.config import OUTPUT_ROOT

# Matches legacy stray folders like 20260603_004724 (no batch_ prefix).
_TIMESTAMP_FOLDER = re.compile(r"^\d{8}_\d{6}$")


def normalize_batch_folder_name(name: str | None) -> str:
    """
    Resolve batch folder name under save_path.

    Default matches Batch Generation: ``batch_YYYYMMDD_HHMMSS``.
    Auto-generated timestamp-only names always get a ``batch_`` prefix so
    legacy folders like ``20260603_004724`` are not created again.
    """
    raw = (name or "").strip()
    safe = "".join(c for c in raw if c.isalnum() or c in ("-", "_")).strip()
    if not safe:
        return datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    if _TIMESTAMP_FOLDER.match(safe) or (
        safe[0].isdigit() and not safe.startswith("batch_")
    ):
        return f"batch_{safe}"
    return safe


def resolve_batch_root(config: dict) -> tuple[str, str]:
    """Return (batch_id, batch_root absolute path)."""
    batch_id = normalize_batch_folder_name(config.get("batch_name"))
    save_root = (config.get("save_path") or OUTPUT_ROOT).rstrip("/\\")
    batch_root = os.path.join(save_root, batch_id)
    return batch_id, batch_root
