# Large-scale batch generation (wave orchestration)

Summary of changes for running very large batches (e.g. 1M clips) in one logical job without breaking physics or randomization.

## Files changed

| File | Change |
|------|--------|
| **`core/batch_runner.py`** | **New.** Wave loop (plan → parallel synth → finalize → clear wave RAM), resume (`sample_*` skip + `sampler_state.json`), streaming `dataset.csv` / `clips_metadata.jsonl`, background thread for jobs ≥ 10k clips. |
| **`core/batch_parallel.py`** | Added `resolve_wave_size()`; `ProcessPoolExecutor` uses **`spawn`** (avoids forked workers inheriting Flask sockets). |
| **`core/progress.py`** | Progress writes to `generation_progress.json` and `{batch_dir}/progress.json`; fields `phase`, `status`, `batch_directory`, `wave_start`. `load_progress()` can read batch-dir file. |
| **`core/sampler.py`** | `save_sampler_state()` / `load_sampler_state()` accept optional filepath (per-batch `sampler_state.json`). |
| **`routes/batch_routes.py`** | Planning moved into `_plan_slot(i)`; synthesis/finalize delegated to `dispatch_standard_batch()`. Removed monolithic “plan all → submit all futures” path. |
| **`templates/index_batch.html`** | Progress bar keeps polling when API returns `started: true` (background jobs); shows `phase` in log. |
| **`README.md`** | Documented waves, workers, background threshold, resume, streaming metadata. |

## What is different (behavior)

1. **Waves (~5000 clips)** — Only one wave of planned jobs and futures is in memory at a time; not all 1M at once.
2. **Same randomization** — `vehicle_list` / `path_list` / pass-by flags built once for `total_clips`; `SAMPLERS` cleared once at start (or restored on resume), **not** reset per wave; global index `sample_{i+1:07d}`.
3. **Large jobs** — ≥ 10k clips: HTTP returns immediately; work runs in a background thread; UI polls `/api/progress`.
4. **Resume** — Restart same batch folder/name; completed samples skipped; sampler state reloaded from `batch_dir/sampler_state.json`.
5. **Metadata** — Large runs append to `dataset.csv` and `clips_metadata.jsonl` instead of holding a giant in-memory list / single huge `metadata.json` clips array.

## Config / env

- `batch.wave_size` or `DOPPLERNET_BATCH_WAVE_SIZE` (default `5000`)
- `batch.workers` or `DOPPLERNET_BATCH_WORKERS`
- Progress: `/api/progress` or `{batch_dir}/progress.json`

## Unchanged (intentionally)

- `audio/generation.py`, `physics/*` — per-clip physics and audio synthesis logic
- `core/sampler.py` — `CyclicIntegerSampler` / `SAMPLERS` algorithm (only persistence path added)
