"""Parallel batch synthesis helpers (wave workers, spawn context)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

DEFAULT_WAVE_SIZE = 5000
DEFAULT_WORKERS = max(1, (os.cpu_count() or 1))
# Always return immediately so the UI can poll per-sample progress (was 10_000 → sync blocked polling).
BACKGROUND_THRESHOLD = 1


def resolve_wave_size(config: Dict[str, Any]) -> int:
    batch = config.get('batch', {}) or {}
    raw = batch.get('wave_size')
    if raw is None:
        raw = os.environ.get('DOPPLERNET_BATCH_WAVE_SIZE', DEFAULT_WAVE_SIZE)
    try:
        size = int(raw)
    except (TypeError, ValueError):
        size = DEFAULT_WAVE_SIZE
    return max(1, size)


def resolve_workers(config: Dict[str, Any]) -> int:
    batch = config.get('batch', {}) or {}
    raw = batch.get('workers')
    if raw is None:
        raw = os.environ.get('DOPPLERNET_BATCH_WORKERS', DEFAULT_WORKERS)
    try:
        workers = int(raw)
    except (TypeError, ValueError):
        workers = DEFAULT_WORKERS
    return max(1, workers)


def make_process_executor(workers: int) -> ProcessPoolExecutor:
    """Process pool using spawn (safe with Flask / loaded libs)."""
    import multiprocessing as mp

    ctx = mp.get_context('spawn')
    return ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
