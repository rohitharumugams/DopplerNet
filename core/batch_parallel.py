"""
Parallel batch clip synthesis.

Parameter planning stays sequential in the Flask handler (preserves SAMPLERS /
cyclic coverage). Workers only run synthesis from fixed job payloads.
"""

from __future__ import annotations

import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.progress import save_progress


def resolve_batch_worker_count(config: Optional[dict] = None) -> int:
    """
    How many clip-synthesis worker processes to use.

    Priority: config['batch']['workers'] > env DOPPLERNET_BATCH_WORKERS > CPU-2.
    Capped to avoid oversubscribing BLAS (each worker keeps OMP_NUM_THREADS=1).
    """
    cfg = config or {}
    batch_cfg = cfg.get('batch') or {}
    if batch_cfg.get('workers') is not None:
        try:
            return max(1, int(batch_cfg['workers']))
        except (TypeError, ValueError):
            pass

    env_val = os.environ.get('DOPPLERNET_BATCH_WORKERS')
    if env_val:
        try:
            return max(1, int(env_val))
        except ValueError:
            pass

    n_cpu = os.cpu_count() or 4
    # Leave headroom for Flask / OS; cap to limit RAM (each worker loads librosa stacks).
    default = max(1, min(n_cpu - 2, 40))
    return default


def _init_worker() -> None:
    """Ensure thread env and dirs exist in child processes (spawn/fork safe)."""
    import core.config  # noqa: F401


def _execute_batch_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one planned job in a worker process.

    Returns {'success': bool, 'result': ..., 'error': str|None, 'traceback': str|None}
    """
    try:
        import core.config  # noqa: F401
        from audio.generation import generate_single_clip, generate_multi_object_clip

        kind = job['kind']
        audio_dir = job['audio_dir']
        batch_id = job['batch_id']
        config = job['config']
        clip_index = int(job['clip_index'])

        if kind == 'single':
            result = generate_single_clip(
                job['vehicle_name'],
                job['path_type'],
                job['params'],
                audio_dir,
                batch_id,
                clip_index,
                config,
            )
        elif kind == 'multi':
            result = generate_multi_object_clip(
                job['v_configs'],
                audio_dir,
                batch_id,
                clip_index,
                config,
                observer_pos=tuple(job['observer_pos']),
                road_curve_a=float(job['road_curve_a']),
                road_y_center=float(job['road_y_center']),
                road_shape=job.get('road_shape', 'parabola'),
                road_bezier_bulge=float(job.get('road_bezier_bulge', 0.0)),
                intersection_angle=float(job.get('intersection_angle', 90.0)),
            )
        else:
            raise ValueError(f"Unknown batch job kind: {kind!r}")

        return {'success': True, 'result': result, 'error': None, 'traceback': None}
    except Exception as exc:
        return {
            'success': False,
            'result': None,
            'error': str(exc),
            'traceback': traceback.format_exc(),
        }


def _rename_sample_folder(audio_dir: str, from_index: int, to_index: int) -> None:
    """Rename sample_XXXXXXX directory after successful synthesis."""
    if from_index == to_index:
        return
    src = os.path.join(audio_dir, f'sample_{from_index:07d}')
    dst = os.path.join(audio_dir, f'sample_{to_index:07d}')
    if not os.path.isdir(src):
        return
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.rename(src, dst)


def _patch_result_indices(result: Dict[str, Any], final_index: int) -> None:
    """Update metadata paths after sample folder renumbering."""
    result['index'] = final_index
    result['sample_dir'] = f'sample_{final_index:07d}'


def finalize_staged_clip(
    audio_dir: str,
    result: Dict[str, Any],
    staging_index: int,
    final_index: int,
) -> Dict[str, Any]:
    """Compact sample folders after parallel synthesis (preserve sequential indexing)."""
    _rename_sample_folder(audio_dir, staging_index, final_index)
    _patch_result_indices(result, final_index)
    return result


def run_planned_jobs_parallel(
    planned_jobs: List[Tuple[int, Dict[str, Any]]],
    *,
    total_clips: int,
    progress_callback: Optional[Callable[[int], None]] = None,
    max_workers: Optional[int] = None,
    config: Optional[dict] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Execute planned jobs concurrently.

    Each job must include ``clip_index`` (staging folder index, unique per slot).
    Callers renumber successful outputs to compact indices in slot order.

    Returns mapping slot_index -> outcome dict from ``_execute_batch_job``.
    """
    if not planned_jobs:
        return {}

    workers = max_workers if max_workers is not None else resolve_batch_worker_count(config)
    workers = max(1, min(workers, len(planned_jobs)))

    outcomes: Dict[int, Dict[str, Any]] = {}
    completed = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
    ) as pool:
        future_to_slot = {
            pool.submit(_execute_batch_job, job): slot
            for slot, job in planned_jobs
        }
        for future in as_completed(future_to_slot):
            slot = future_to_slot[future]
            try:
                outcomes[slot] = future.result()
            except Exception as exc:
                outcomes[slot] = {
                    'success': False,
                    'result': None,
                    'error': str(exc),
                    'traceback': traceback.format_exc(),
                }
            completed += 1
            if progress_callback:
                progress_callback(completed)
            else:
                save_progress(total_clips, completed)

    return outcomes
