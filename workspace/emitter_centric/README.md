# Emitter-Centric Analysis

Isolated Workspace sub-mode for **co-moving source-frame** synthesis and optional **observer-centric** comparison batches.

## Documentation

| File | Contents |
|------|----------|
| **[plan.md](plan.md)** | Technical blueprint (Formulation 1), math, integration phases, first principles (§13), output audit (§14), [references](plan.md#references) |
| **This README** | Quick start, layout, API |

Read **plan.md §13** before interpreting outputs: emitter audio is a co-moving microphone (no geometric Doppler), not the same as Formulation 1 “emission grid → roadside observer” (§1–§8).

## Output layout

```text
batch_YYYYMMDD_HHMMSS/
├── overall_metadata.json
├── dataset.csv
├── statistics_{batch_id}.txt
├── generation_log_{batch_id}.txt
├── emitter_centric/
│   ├── audio_clips/sample_XXXXXXX/{Common,Essential}/...
│   └── metadata_{batch_id}.json
├── observer_centric/             # if comparison enabled
└── comparison_outputs/sample_XXXXXXX/
```

Default save root: `static/workspace_outputs/emitter_centric/`

## Modules

| File | Role |
|------|------|
| `source_frame.py` | Co-moving audio |
| `batch_runner.py` | Single + batch orchestration |
| `clip_export.py` | Batch-style Common/Essential export |
| `artifacts.py` | Sidecars (`source_positions`, emitter-frame kinematics) |
| `comparison_plots.py` | Side-by-side figures |
| `sampling.py` | Clip plan builder |
| `synthesis.py` | Legacy straight-CV / Formulation-1 tools |
| `run_synthesis.py` | CLI |

## API

- `POST /api/workspace/emitter_centric/generate` — body `{ "config": { ... } }`
- `GET /api/workspace/emitter_centric/status/<job_id>`

## CLI

```bash
python -m workspace.emitter_centric.run_synthesis --speed-mps 30 --distance-m 15 --duration 10
```

## Isolation

No imports from this package into `audio/generation.py` batch routes. Production Batch Generation and Quadratic Acceleration are unchanged.
