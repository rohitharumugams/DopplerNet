"""
Wave-based batch orchestration for large jobs (e.g. 1M clips).

Planning stays sequential (SAMPLERS / cyclic coverage). Synthesis runs in
waves so RAM and pending futures stay bounded. Global slot index i maps to
sample_{i+1:07d}; SAMPLERS are not cleared between waves.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.batch_parallel import (
    resolve_batch_worker_count,
    resolve_wave_size,
    run_planned_jobs_parallel,
)
from core.progress import save_progress
from core.sampler import SAMPLERS, load_sampler_state, save_sampler_state
from audio.generation import (
    generate_multi_object_clip,
    generate_statistics,
    generate_single_clip,
)

STREAMING_METADATA_THRESHOLD = 10_000
BACKGROUND_CLIP_THRESHOLD = 10_000

DATASET_CSV_HEADERS = [
    'sample_id', 'batch_id', 'filename', 'vehicle_class', 'trajectory_type',
    'speed_mps', 'direction_label', 'direction_text', 'cpa_distance_m', 'cpa_time_sec',
    'num_sources', 'is_crossing',
]

_batch_lock = threading.Lock()
_batch_thread: Optional[threading.Thread] = None


def sample_dir_path(audio_dir: str, clip_index: int) -> str:
    return os.path.join(audio_dir, f'sample_{clip_index:07d}')


def is_sample_complete(audio_dir: str, clip_index: int) -> bool:
    """True if this global slot already has a finished clip on disk."""
    sample_dir = sample_dir_path(audio_dir, clip_index)
    if not os.path.isdir(sample_dir):
        return False
    for sub in ('Common', 'Essential'):
        wav_dir = os.path.join(sample_dir, sub)
        if os.path.isdir(wav_dir) and any(
            name.endswith('.wav') for name in os.listdir(wav_dir)
        ):
            return True
    return False


def append_dataset_row(dataset_file: str, batch_id: str, clip: Dict[str, Any]) -> None:
    labels = clip.get('labels', {})
    row = {
        'sample_id': clip.get('sample_dir', ''),
        'batch_id': batch_id,
        'filename': clip.get('filename', ''),
        'vehicle_class': labels.get('vehicle_class', ''),
        'trajectory_type': labels.get('trajectory_type', ''),
        'speed_mps': labels.get('speed_mps', 0.0),
        'direction_label': labels.get('direction_label', 0),
        'direction_text': labels.get('direction_text', ''),
        'cpa_distance_m': labels.get('cpa_distance_m', 0.0),
        'cpa_time_sec': labels.get('cpa_time_sec', 5.0),
        'num_sources': labels.get('num_sources', 1),
        'is_crossing': labels.get('is_crossing', False),
    }
    file_exists = os.path.exists(dataset_file) and os.path.getsize(dataset_file) > 0
    with open(dataset_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_metadata_jsonl(jsonl_path: str, clip: Dict[str, Any]) -> None:
    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(clip) + '\n')


def _remove_staging_sample(audio_dir: str, staging_index: int) -> None:
    import shutil
    staging_dir = sample_dir_path(audio_dir, staging_index)
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir, ignore_errors=True)


def _finalize_wave_slot(
    *,
    slot_index: int,
    job: Dict[str, Any],
    outcome: Optional[Dict[str, Any]],
    audio_dir: str,
    batch_id: str,
    config: dict,
    path_list: List[str],
    motion_pass_by_flags: List[bool],
    apply_direction_variant: Callable,
    run_single_fallback: Callable,
    total_clips: int,
    generation_log: List[str],
    clips_metadata: List[Dict[str, Any]],
    streaming_metadata: bool,
    dataset_file: str,
    metadata_jsonl: str,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Process one slot after parallel synthesis. Uses global index slot_index+1.
    Returns (clip_metadata or None, slot_failed).
    """
    clip_index = slot_index + 1
    path_type = job.get('path_type', path_list[slot_index])

    if outcome and outcome.get('success') and outcome.get('result'):
        result = outcome['result']
        result['index'] = clip_index
        result['sample_dir'] = f'sample_{clip_index:07d}'
        if not streaming_metadata:
            clips_metadata.append(result)
        append_dataset_row(dataset_file, batch_id, result)
        append_metadata_jsonl(metadata_jsonl, result)
        generation_log.append(
            f"Generated clip {clip_index}: {result.get('filename', '')}"
        )
        print(f"Generated clip {clip_index}")
        return result, False

    err_msg = (outcome or {}).get('error', 'unknown error')
    generation_log.append(
        f"Parallel synthesis failed slot {clip_index}: {err_msg}; retrying"
    )
    print(generation_log[-1])

    try:
        if job['kind'] == 'single':
            result, path_type = run_single_fallback(
                config,
                job['vehicle_name'],
                job['path_type'],
                audio_dir,
                batch_id,
                clip_index,
                total_clips,
                motion_pass_by_flags[slot_index],
                slot_index,
                apply_direction_variant,
            )
        elif job['kind'] == 'multi':
            _remove_staging_sample(audio_dir, clip_index)
            result = generate_multi_object_clip(
                job['v_configs'],
                audio_dir,
                batch_id,
                clip_index,
                job['config'],
                observer_pos=tuple(job['observer_pos']),
                road_curve_a=float(job['road_curve_a']),
                road_y_center=float(job['road_y_center']),
                road_shape=job.get('road_shape', 'parabola'),
                road_bezier_bulge=float(job.get('road_bezier_bulge', 0.0)),
                intersection_angle=float(job.get('intersection_angle', 90.0)),
            )
        else:
            raise RuntimeError(f"Unknown job kind: {job.get('kind')}")

        _remove_staging_sample(audio_dir, clip_index)
        result['index'] = clip_index
        result['sample_dir'] = f'sample_{clip_index:07d}'
        if not streaming_metadata:
            clips_metadata.append(result)
        append_dataset_row(dataset_file, batch_id, result)
        append_metadata_jsonl(metadata_jsonl, result)
        generation_log.append(
            f"Generated clip {clip_index}: {result.get('filename', '')} (fallback)"
        )
        print(f"Generated clip {clip_index} (fallback)")
        return result, False
    except Exception as exc:
        _remove_staging_sample(audio_dir, clip_index)
        error_message = f"Error generating slot {clip_index}: {str(exc)}"
        traceback.print_exc()
        generation_log.append(error_message)
        print(error_message)
        return None, True


def run_standard_batch(
    *,
    config: dict,
    batch_dir: str,
    batch_id: str,
    audio_dir: str,
    total_clips: int,
    vehicle_list: List[str],
    path_list: List[str],
    motion_pass_by_flags: List[bool],
    apply_direction_variant: Callable,
    plan_slot_fn: Callable[[int], Dict[str, Any]],
    run_single_fallback: Callable,
    start_time: float,
) -> Dict[str, Any]:
    """Plan and synthesize all clips in waves; preserve global SAMPLERS state."""
    wave_size = resolve_wave_size(config)
    n_workers = resolve_batch_worker_count(config)
    streaming_metadata = total_clips > STREAMING_METADATA_THRESHOLD

    sampler_state_path = os.path.join(batch_dir, 'sampler_state.json')
    if os.path.exists(sampler_state_path):
        load_sampler_state(sampler_state_path)
    else:
        SAMPLERS.clear()

    batch_progress = os.path.join(batch_dir, 'progress.json')
    generated_so_far = 0
    if os.path.exists(batch_progress):
        try:
            with open(batch_progress, 'r') as f:
                generated_so_far = int(json.load(f).get('generated_so_far', 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            generated_so_far = 0

    save_progress(
        total_clips,
        generated_so_far,
        batch_dir,
        phase='starting',
        status='running',
        batch_directory=batch_dir,
    )

    clips_metadata: List[Dict[str, Any]] = []
    generation_log: List[str] = []
    slots_failed = 0
    dataset_file = os.path.join(batch_dir, 'dataset.csv')
    metadata_jsonl = os.path.join(batch_dir, 'clips_metadata.jsonl')

    for wave_start in range(0, total_clips, wave_size):
        wave_end = min(wave_start + wave_size, total_clips)
        save_progress(
            total_clips,
            generated_so_far,
            batch_dir,
            phase='planning',
            status='running',
            batch_directory=batch_dir,
            wave_start=wave_start,
        )

        planned_wave: List[Tuple[int, Dict[str, Any]]] = []
        for i in range(wave_start, wave_end):
            clip_index = i + 1
            if is_sample_complete(audio_dir, clip_index):
                generated_so_far += 1
                save_progress(
                    total_clips,
                    generated_so_far,
                    batch_dir,
                    phase='resume_skip',
                    status='running',
                    batch_directory=batch_dir,
                    wave_start=wave_start,
                )
                continue

            try:
                job = plan_slot_fn(i)
                job.setdefault('audio_dir', audio_dir)
                job.setdefault('batch_id', batch_id)
                job.setdefault('config', config)
                job['clip_index'] = clip_index
                planned_wave.append((i, job))
            except Exception as exc:
                slots_failed += 1
                error_message = f"Error planning slot {clip_index}/{total_clips}: {str(exc)}"
                traceback.print_exc()
                generation_log.append(error_message)
                print(error_message)

        if planned_wave:
            print(
                f"Wave {wave_start}-{wave_end - 1}: synthesizing {len(planned_wave)} clips, "
                f"{n_workers} worker(s)"
            )
            save_progress(
                total_clips,
                generated_so_far,
                batch_dir,
                phase='synthesis',
                status='running',
                batch_directory=batch_dir,
                wave_start=wave_start,
            )

            def _on_wave_progress(done_in_wave: int) -> None:
                save_progress(
                    total_clips,
                    generated_so_far + done_in_wave,
                    batch_dir,
                    phase='synthesis',
                    status='running',
                    batch_directory=batch_dir,
                    wave_start=wave_start,
                )

            outcomes = run_planned_jobs_parallel(
                planned_wave,
                total_clips=total_clips,
                max_workers=n_workers,
                config=config,
                progress_callback=_on_wave_progress,
            )

            job_by_slot = {slot: job for slot, job in planned_wave}
            for i in range(wave_start, wave_end):
                if i not in job_by_slot:
                    continue
                job = job_by_slot[i]
                outcome = outcomes.get(i)
                result, failed = _finalize_wave_slot(
                    slot_index=i,
                    job=job,
                    outcome=outcome,
                    audio_dir=audio_dir,
                    batch_id=batch_id,
                    config=config,
                    path_list=path_list,
                    motion_pass_by_flags=motion_pass_by_flags,
                    apply_direction_variant=apply_direction_variant,
                    run_single_fallback=run_single_fallback,
                    total_clips=total_clips,
                    generation_log=generation_log,
                    clips_metadata=clips_metadata,
                    streaming_metadata=streaming_metadata,
                    dataset_file=dataset_file,
                    metadata_jsonl=metadata_jsonl,
                )
                if failed:
                    slots_failed += 1
                elif result:
                    generated_so_far += 1
                    save_progress(
                        total_clips,
                        generated_so_far,
                        batch_dir,
                        phase='synthesis',
                        status='running',
                        batch_directory=batch_dir,
                        wave_start=wave_start,
                    )

        save_sampler_state(sampler_state_path)

    save_progress(
        total_clips,
        generated_so_far,
        batch_dir,
        phase='done',
        status='completed',
        batch_directory=batch_dir,
    )

    metadata_file = os.path.join(batch_dir, f'metadata_{batch_id}.json')
    metadata_payload = {
        'batch_id': batch_id,
        'config': config,
        'total_generated': generated_so_far,
        'total_requested': total_clips,
        'slots_failed': slots_failed,
        'timestamp': datetime.now().isoformat(),
        'dataset_csv': dataset_file,
        'clips_metadata_jsonl': metadata_jsonl,
    }
    if streaming_metadata:
        metadata_payload['clips'] = []
        metadata_payload['streaming'] = True
    else:
        metadata_payload['clips'] = clips_metadata

    with open(metadata_file, 'w') as f:
        json.dump(metadata_payload, f, indent=2)

    log_file = os.path.join(batch_dir, f'generation_log_{batch_id}.txt')
    with open(log_file, 'w') as f:
        f.write('\n'.join(generation_log))

    stats_file = os.path.join(batch_dir, f'statistics_{batch_id}.txt')
    stats_source = clips_metadata if clips_metadata else []
    stats_text = generate_statistics(stats_source, config) if stats_source else (
        f"Batch {batch_id}: {generated_so_far} clips (see {dataset_file} for labels)."
    )
    with open(stats_file, 'w') as f:
        f.write(stats_text)

    elapsed_time = time.time() - start_time
    formatted_time = f"{elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)"
    print(f"Batch generation finished in {formatted_time}")

    return {
        'success': True,
        'batch_id': batch_id,
        'total_requested': total_clips,
        'total_generated': generated_so_far,
        'slots_failed': slots_failed,
        'parallel_workers': n_workers,
        'wave_size': wave_size,
        'elapsed_time': elapsed_time,
        'formatted_time': formatted_time,
        'batch_directory': batch_dir,
        'metadata_file': metadata_file,
        'log_file': log_file,
        'stats_file': stats_file,
    }


def dispatch_standard_batch(
    *,
    config: dict,
    batch_dir: str,
    batch_id: str,
    audio_dir: str,
    total_clips: int,
    vehicle_list: List[str],
    path_list: List[str],
    motion_pass_by_flags: List[bool],
    apply_direction_variant: Callable,
    plan_slot_fn: Callable[[int], Dict[str, Any]],
    run_single_fallback: Callable,
    start_time: float,
):
    """Run synchronously or in a background thread for very large jobs."""
    global _batch_thread

    kwargs = dict(
        config=config,
        batch_dir=batch_dir,
        batch_id=batch_id,
        audio_dir=audio_dir,
        total_clips=total_clips,
        vehicle_list=vehicle_list,
        path_list=path_list,
        motion_pass_by_flags=motion_pass_by_flags,
        apply_direction_variant=apply_direction_variant,
        plan_slot_fn=plan_slot_fn,
        run_single_fallback=run_single_fallback,
        start_time=start_time,
    )

    if total_clips < BACKGROUND_CLIP_THRESHOLD:
        return run_standard_batch(**kwargs), False

    with _batch_lock:
        if _batch_thread is not None and _batch_thread.is_alive():
            return {'error': 'A batch job is already running'}, True

        def _worker():
            try:
                run_standard_batch(**kwargs)
            except Exception as exc:
                traceback.print_exc()
                save_progress(
                    total_clips,
                    0,
                    batch_dir,
                    phase='failed',
                    status='failed',
                    batch_directory=batch_dir,
                    error=str(exc),
                )

        _batch_thread = threading.Thread(target=_worker, daemon=False)
        _batch_thread.start()

    save_progress(
        total_clips,
        0,
        batch_dir,
        phase='starting',
        status='running',
        batch_directory=batch_dir,
    )
    return {
        'success': True,
        'started': True,
        'batch_id': batch_id,
        'total_requested': total_clips,
        'batch_directory': batch_dir,
        'message': (
            f'Batch started in background ({total_clips} clips, '
            f'wave size {resolve_wave_size(config)}). '
            f'Poll /api/progress or {batch_dir}/progress.json.'
        ),
    }, False
