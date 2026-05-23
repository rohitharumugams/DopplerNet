"""Wave-based standard batch orchestration (plan → parallel synth → finalize)."""

from __future__ import annotations

import csv
import json
import os
import random
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from flask import jsonify

from audio.generation import calculate_distribution, generate_statistics
from core.batch_parallel import (
    BACKGROUND_THRESHOLD,
    make_process_executor,
    resolve_wave_size,
    resolve_workers,
)
from core.batch_planning import plan_slot, synthesize_planned
from core.config import OUTPUT_FOLDER
from core.progress import save_progress
from core.sampler import SAMPLERS, load_sampler_state, save_sampler_state

PLAN_STATE_FILE = 'batch_plan_state.json'
SAMPLER_STATE_NAME = 'sampler_state.json'
LARGE_RUN_THRESHOLD = BACKGROUND_THRESHOLD


def _worker_synthesize(job: dict) -> dict:
    """Top-level worker entry (spawn-safe)."""
    return synthesize_planned(job)


def _is_multi_source_clip(clip: dict) -> bool:
    labels = clip.get('labels', {}) or {}
    num = int(labels.get('num_sources', clip.get('num_sources', 1)) or 1)
    if num > 1:
        return True
    if clip.get('vehicle') == 'multi':
        return True
    if labels.get('vehicle_class') == 'multi':
        return True
    return False


def _fix_list(lst, allowed_values, total_clips):
    lst = list(lst)
    if not allowed_values:
        allowed_values = ['car_1']
    if not lst:
        return [allowed_values[0]] * total_clips
    if len(lst) > total_clips:
        return lst[:total_clips]
    while len(lst) < total_clips:
        lst.append(random.choice(allowed_values))
    return lst


def _build_motion_pass_by_flags(total_clips: int) -> List[bool]:
    flags: List[bool] = []
    for j in range(0, total_clips, 10):
        block_size = min(10, total_clips - j)
        if block_size == 10:
            block = [True] * 8 + [False] * 2
        else:
            n_pass = int(round(block_size * 0.8))
            block = [True] * n_pass + [False] * (block_size - n_pass)
        random.shuffle(block)
        flags.extend(block)
    return flags


def _resolve_batch_paths(config: dict) -> Tuple[str, str, str]:
    base_output_root = config.get('output', {}).get('path', OUTPUT_FOLDER)
    os.makedirs(base_output_root, exist_ok=True)

    custom_name = config.get('batch', {}).get('name', '').strip()
    if custom_name:
        safe_batch_name = ''.join(
            c for c in custom_name if c.isalnum() or c in (' ', '-', '_')
        ).strip().replace(' ', '_')
        batch_id = safe_batch_name
        batch_dir = os.path.join(base_output_root, batch_id)
    else:
        batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_dir = os.path.join(base_output_root, f'batch_{batch_id}')

    audio_dir = os.path.join(batch_dir, 'audio_clips')
    os.makedirs(audio_dir, exist_ok=True)
    return batch_id, batch_dir, audio_dir


def scan_completed_samples(audio_dir: str) -> Set[int]:
    """Return 1-based sample indices that already exist on disk."""
    completed: Set[int] = set()
    if not os.path.isdir(audio_dir):
        return completed
    for name in os.listdir(audio_dir):
        if not name.startswith('sample_'):
            continue
        sample_path = os.path.join(audio_dir, name)
        if not os.path.isdir(sample_path):
            continue
        try:
            completed.add(int(name.split('_', 1)[1]))
        except (IndexError, ValueError):
            continue
    return completed


def _load_or_build_plan_state(
    batch_dir: str,
    config: dict,
    total_clips: int,
    *,
    resume: bool,
) -> Tuple[List[str], List[str], List[bool]]:
    plan_path = os.path.join(batch_dir, PLAN_STATE_FILE)
    if resume and os.path.exists(plan_path):
        with open(plan_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        return (
            saved['vehicle_list'],
            saved['path_list'],
            saved['motion_pass_by_flags'],
        )

    distribution = calculate_distribution(config, total_clips)
    vehicle_dist = distribution['vehicles']
    path_dist = distribution['paths']

    vehicle_list: List[str] = []
    for v, count in vehicle_dist.items():
        vehicle_list.extend([v] * int(count))

    path_list: List[str] = []
    for p, count in path_dist.items():
        path_list.extend([p] * int(count))

    vehicle_list = _fix_list(vehicle_list, list(vehicle_dist.keys()), total_clips)
    path_list = _fix_list(path_list, list(path_dist.keys()), total_clips)
    random.shuffle(vehicle_list)
    random.shuffle(path_list)
    motion_pass_by_flags = _build_motion_pass_by_flags(total_clips)

    os.makedirs(batch_dir, exist_ok=True)
    with open(plan_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'vehicle_list': vehicle_list,
                'path_list': path_list,
                'motion_pass_by_flags': motion_pass_by_flags,
            },
            f,
            indent=2,
        )
    return vehicle_list, path_list, motion_pass_by_flags


def build_batch_context(config: dict) -> dict:
    total_clips = int(config['batch']['total_clips'])
    batch_id, batch_dir, audio_dir = _resolve_batch_paths(config)
    completed = scan_completed_samples(audio_dir)
    resume = os.path.isdir(batch_dir) and (
        os.path.exists(os.path.join(batch_dir, SAMPLER_STATE_NAME))
        or os.path.exists(os.path.join(batch_dir, PLAN_STATE_FILE))
        or bool(completed)
    )

    vehicle_list, path_list, motion_pass_by_flags = _load_or_build_plan_state(
        batch_dir, config, total_clips, resume=resume,
    )

    sampler_path = os.path.join(batch_dir, SAMPLER_STATE_NAME)
    if resume and os.path.exists(sampler_path):
        SAMPLERS.clear()
        load_sampler_state(sampler_path)
    else:
        SAMPLERS.clear()

    return {
        'config': config,
        'total_clips': total_clips,
        'batch_id': batch_id,
        'batch_dir': batch_dir,
        'audio_dir': audio_dir,
        'vehicle_list': vehicle_list,
        'path_list': path_list,
        'motion_pass_by_flags': motion_pass_by_flags,
        'completed_samples': completed,
        'resume': resume,
        'wave_size': resolve_wave_size(config),
        'workers': resolve_workers(config),
        'streaming': total_clips >= LARGE_RUN_THRESHOLD,
        'sampler_path': sampler_path,
    }


def _csv_headers(include_pass_by: bool) -> List[str]:
    headers = [
        'sample_id', 'batch_id', 'filename', 'vehicle_class', 'trajectory_type',
        'speed_mps', 'acceleration', 'direction_label', 'direction_text',
        'cpa_distance_m', 'cpa_time_sec', 'num_sources', 'is_crossing',
    ]
    if include_pass_by:
        headers.append('pass_by_in_clip')
    return headers


def _clip_to_csv_row(clip: dict, batch_id: str, include_pass_by: bool) -> dict:
    labels = clip.get('labels', {}) or {}
    row = {
        'sample_id': clip.get('sample_dir', ''),
        'batch_id': batch_id,
        'filename': clip.get('filename', ''),
        'vehicle_class': labels.get('vehicle_class', ''),
        'trajectory_type': labels.get('trajectory_type', ''),
        'speed_mps': labels.get('speed_mps', 0.0),
        'acceleration': labels.get('acceleration', 0.0),
        'direction_label': labels.get('direction_label', ''),
        'direction_text': labels.get('direction_text', ''),
        'cpa_distance_m': labels.get('cpa_distance_m', 0.0),
        'cpa_time_sec': labels.get('cpa_time_sec', 0.0),
        'num_sources': labels.get('num_sources', 1),
        'is_crossing': labels.get('is_crossing', False),
    }
    if include_pass_by and not _is_multi_source_clip(clip):
        row['pass_by_in_clip'] = str(clip.get('pass_by_in_clip', True)).lower()
    return row


def _load_clips_from_jsonl(jsonl_file: str) -> List[dict]:
    clips: List[dict] = []
    if not os.path.exists(jsonl_file):
        return clips
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                clips.append(json.loads(line))
    return clips


def _synthesize_slot_with_retries(i: int, ctx: dict, executor: Optional[ProcessPoolExecutor]) -> dict:
    last_err = None
    for attempt in range(8):
        try:
            job = plan_slot(i, ctx)
            if executor is not None:
                return executor.submit(_worker_synthesize, job).result()
            return synthesize_planned(job)
        except Exception as exc:
            last_err = exc
            if attempt == 7:
                raise
    raise last_err or RuntimeError('clip generation failed')


def _process_wave(
    wave_indices: List[int],
    ctx: dict,
    executor: Optional[ProcessPoolExecutor],
) -> Tuple[List[Tuple[int, dict]], int, List[str]]:
    results: List[Tuple[int, dict]] = []
    slots_failed = 0
    errors: List[str] = []

    if executor is not None and ctx['workers'] > 1:
        planned: List[Tuple[int, dict]] = []
        for i in wave_indices:
            try:
                planned.append((i, plan_slot(i, ctx)))
            except Exception:
                slots_failed += 1
                err = f"Error planning slot {i + 1}/{ctx['total_clips']}"
                traceback.print_exc()
                errors.append(err)

        futures = {executor.submit(_worker_synthesize, job): i for i, job in planned}
        failed_indices: List[int] = []
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results.append((i, fut.result()))
            except Exception:
                failed_indices.append(i)
                err = f"Error synthesizing slot {i + 1}/{ctx['total_clips']}"
                traceback.print_exc()
                errors.append(err)

        for i in failed_indices:
            try:
                results.append((i, _synthesize_slot_with_retries(i, ctx, None)))
            except Exception:
                slots_failed += 1
                err = f"Error generating slot {i + 1}/{ctx['total_clips']}"
                traceback.print_exc()
                errors.append(err)
    else:
        for i in wave_indices:
            try:
                results.append((i, _synthesize_slot_with_retries(i, ctx, None)))
            except Exception:
                slots_failed += 1
                err = f"Error generating slot {i + 1}/{ctx['total_clips']}"
                traceback.print_exc()
                errors.append(err)

    results.sort(key=lambda pair: pair[0])
    return results, slots_failed, errors


def _finalize_wave_results(
    wave_results: List[Tuple[int, dict]],
    *,
    ctx: dict,
    batch_id: str,
    total_clips: int,
    streaming: bool,
    csv_file,
    csv_writer,
    csv_headers_written: bool,
    jsonl_fp,
    log_fp,
    clips_metadata: Optional[List[dict]],
    batch_dir: str,
    generated_count: int,
) -> Tuple[Any, bool, int]:
    include_pass_by = True
    for i, result in wave_results:
        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=_csv_headers(include_pass_by))
            if not csv_headers_written:
                csv_writer.writeheader()
                csv_headers_written = True

        if streaming:
            csv_writer.writerow(_clip_to_csv_row(result, batch_id, include_pass_by))
            csv_file.flush()
            jsonl_fp.write(json.dumps(result) + '\n')
            jsonl_fp.flush()
        elif clips_metadata is not None:
            clips_metadata.append(result)

        msg = f"Generated clip {i + 1}/{total_clips}: {result.get('filename', '')}"
        log_fp.write(msg + '\n')
        log_fp.flush()
        generated_count += 1
        save_progress(
            total_clips,
            generated_count,
            batch_dir=batch_dir,
            batch_id=batch_id,
            phase='running',
            status='running',
        )
        print(msg)
    return csv_writer, csv_headers_written, generated_count


def run_standard_batch(ctx: dict, start_time: float) -> dict:
    config = ctx['config']
    total_clips = ctx['total_clips']
    batch_id = ctx['batch_id']
    batch_dir = ctx['batch_dir']
    completed = set(ctx['completed_samples'])
    wave_size = ctx['wave_size']
    workers = ctx['workers']
    streaming = ctx['streaming']
    sampler_path = ctx['sampler_path']

    slots_failed = 0
    clips_metadata: Optional[List[dict]] = [] if not streaming else None

    dataset_file = os.path.join(batch_dir, 'dataset.csv')
    jsonl_file = os.path.join(batch_dir, 'clips_metadata.jsonl')
    log_file = os.path.join(batch_dir, f'generation_log_{batch_id}.txt')

    csv_mode = 'a' if ctx['resume'] and os.path.exists(dataset_file) else 'w'
    csv_file = open(dataset_file, csv_mode, newline='', encoding='utf-8')
    csv_writer = None
    csv_headers_written = ctx['resume'] and os.path.exists(dataset_file) and os.path.getsize(dataset_file) > 0
    include_pass_by = True

    jsonl_fp = open(jsonl_file, 'a' if ctx['resume'] else 'w', encoding='utf-8')
    log_fp = open(log_file, 'a' if ctx['resume'] else 'w', encoding='utf-8')

    generated_count = len(completed)
    save_progress(
        total_clips,
        generated_count,
        batch_dir=batch_dir,
        batch_id=batch_id,
        phase='running',
        status='running',
    )

    pending_indices = [i for i in range(total_clips) if (i + 1) not in completed]

    try:
        executor_ctx = make_process_executor(workers) if workers > 1 else nullcontext(None)
        with executor_ctx as executor:
            for wave_start in range(0, len(pending_indices), wave_size):
                wave_indices = pending_indices[wave_start: wave_start + wave_size]
                wave_results, wave_failed, wave_errors = _process_wave(wave_indices, ctx, executor)
                slots_failed += wave_failed
                for err in wave_errors:
                    log_fp.write(err + '\n')
                log_fp.flush()

                csv_writer, csv_headers_written, generated_count = _finalize_wave_results(
                    wave_results,
                    ctx=ctx,
                    batch_id=batch_id,
                    total_clips=total_clips,
                    streaming=streaming,
                    csv_file=csv_file,
                    csv_writer=csv_writer,
                    csv_headers_written=csv_headers_written,
                    jsonl_fp=jsonl_fp,
                    log_fp=log_fp,
                    clips_metadata=clips_metadata,
                    batch_dir=batch_dir,
                    generated_count=generated_count,
                )
                save_sampler_state(sampler_path)
    finally:
        csv_file.close()
        jsonl_fp.close()
        log_fp.close()

    if streaming:
        all_clips = _load_clips_from_jsonl(jsonl_file)
    else:
        all_clips = clips_metadata or []

    has_single_source = any(not _is_multi_source_clip(c) for c in all_clips)
    if not streaming:
        with open(dataset_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=_csv_headers(has_single_source))
            writer.writeheader()
            for clip in all_clips:
                writer.writerow(_clip_to_csv_row(clip, batch_id, has_single_source))

    metadata_file = os.path.join(batch_dir, f'metadata_{batch_id}.json')
    metadata_payload = {
        'batch_id': batch_id,
        'config': config,
        'total_generated': len(all_clips),
        'total_requested': total_clips,
        'slots_failed': slots_failed,
        'timestamp': datetime.now().isoformat(),
    }
    if streaming:
        metadata_payload['clips_jsonl'] = jsonl_file
        metadata_payload['dataset_csv'] = dataset_file
    else:
        metadata_payload['clips'] = all_clips
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_payload, f, indent=2)

    stats_text = generate_statistics(all_clips, config)
    stats_file = os.path.join(batch_dir, f'statistics_{batch_id}.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(stats_text)

    elapsed_time = time.time() - start_time
    formatted_time = f"{elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)"
    print(f"Batch generation finished in {formatted_time}")

    save_progress(
        total_clips,
        generated_count,
        batch_dir=batch_dir,
        batch_id=batch_id,
        phase='done',
        status='completed',
        metadata_file=metadata_file,
        log_file=log_file,
        stats_file=stats_file,
        formatted_time=formatted_time,
        slots_failed=slots_failed,
    )

    return {
        'success': True,
        'batch_id': batch_id,
        'total_requested': total_clips,
        'total_generated': len(all_clips),
        'slots_failed': slots_failed,
        'elapsed_time': elapsed_time,
        'formatted_time': formatted_time,
        'batch_directory': batch_dir,
        'metadata_file': metadata_file,
        'log_file': log_file,
        'stats_file': stats_file,
    }


def dispatch_standard_batch(config: dict, start_time: float):
    ctx = build_batch_context(config)
    total_clips = ctx['total_clips']

    if total_clips >= BACKGROUND_THRESHOLD:
        save_progress(
            total_clips,
            len(ctx['completed_samples']),
            batch_dir=ctx['batch_dir'],
            batch_id=ctx['batch_id'],
            phase='running',
            status='running',
        )

        def _background_run():
            try:
                run_standard_batch(ctx, start_time)
            except Exception:
                traceback.print_exc()
                save_progress(
                    total_clips,
                    len(ctx['completed_samples']),
                    batch_dir=ctx['batch_dir'],
                    batch_id=ctx['batch_id'],
                    phase='done',
                    status='failed',
                )

        threading.Thread(target=_background_run, daemon=True).start()
        return jsonify({
            'success': True,
            'background': True,
            'batch_id': ctx['batch_id'],
            'total_requested': total_clips,
            'batch_directory': ctx['batch_dir'],
            'resume': ctx['resume'],
        })

    result = run_standard_batch(ctx, start_time)
    return jsonify(result)
