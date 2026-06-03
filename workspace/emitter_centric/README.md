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
    ├── stacked_spectrogram_comparison.png  # CQT + narrow + wide (labeled rows)
    ├── diagnostics_panel.png
    ├── emitter_centric.wav / observer_centric.wav
    └── comparison_report.txt
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

## Comparison metrics (observer vs emitter branch)

When **Compare with observer-centric formulation** is enabled, each sample gets `comparison_metrics` in `overall_metadata.json` and `comparison_outputs/sample_*/comparison_report.txt`. Metrics come from `compare_formulations.compare_waveforms`: the two WAVs are trimmed to the same length, then the **emitter** waveform is **RMS-scaled** to match the **observer** level before subtracting (so loudness does not dominate the residual).

Example:

```json
{
  "n_samples": 220500,
  "max_abs_residual": 0.060000664217146366,
  "rms_residual": 0.00982909142854521,
  "dbfs_peak_residual_vs_peak": 0.5993603541924504,
  "dbfs_rms_residual_vs_peak": -15.113493311821792,
  "correlation": 0.00113480927104977,
  "parity_target_dbfs": -60.0,
  "meets_parity_target": false
}
```

| Field | Meaning | What to expect here |
|-------|---------|---------------------|
| `n_samples` | Number of samples compared (`min(len(observer), len(emitter))`). | `duration × 22050` for a full clip (e.g. 10 s → **220500**). |
| `max_abs_residual` | Peak absolute sample difference after level match: \(\max_k \|a[k] - b[k]\|\). | **Large** when paradigms differ (no geometric Doppler on emitter vs full pass-by on observer). Values \(\sim 10^{-2}\)–\(10^{-1}\) are typical, not a synthesis bug. |
| `rms_residual` | RMS of the residual waveform. | **Non-zero** for the same reason; often \(\sim 10^{-3}\)–\(10^{-2}\) for pass-by scenes. |
| `dbfs_peak_residual_vs_peak` | \(20 \log_{10}(\text{max\_abs\_residual} / \text{peak\_observer})\). | **Near 0 dB or positive** (e.g. **~0.6 dB**) means the worst sample error is a sizable fraction of the observer peak — **expected** for co-moving vs roadside comparison. |
| `dbfs_rms_residual_vs_peak` | \(20 \log_{10}(\text{rms\_residual} / \text{peak\_observer})\). | Often **−10 to −20 dBFS** (e.g. **~−15 dBFS**): average mismatch is smaller than the peak spike but still far from “inaudible.” |
| `correlation` | Pearson correlation between level-matched waveforms. | **Near 0** (e.g. **0.001**) is **normal**: different physics (no $c/(c+v_r)$ swoosh vs full Doppler wings). Do **not** treat low correlation as failed synthesis. |
| `parity_target_dbfs` | Threshold copied from the blueprint (plan §8.10) for **Formulation 1 vs Formulation 2** on the **same** observer WAV. | Fixed at **−60.0** dBFS peak residual vs observer peak. |
| `meets_parity_target` | `dbfs_peak_residual_vs_peak < parity_target_dbfs`. | **`false` is expected** for observer vs co-moving emitter clips. The −60 dBFS gate applies to numerical parity between two **observer-time** synthesizers, not to cross-paradigm comparison. |

**Ideal outcomes depend on what you are testing:**

- **Observer vs co-moving emitter (this UI):** Large residuals, low correlation, `meets_parity_target: false` — confirms the branches are physically distinct. Use plots under `comparison_outputs/` and `plan.md` §13 to interpret.
- **Formulation 1 vs Formulation 2 parity (future / `synthesis.py` tools):** Same scene and observer grid; target **`meets_parity_target: true`**, **`dbfs_peak_residual_vs_peak` < −60**, correlation **close to 1** after alignment.

## Isolation

No imports from this package into `audio/generation.py` batch routes. Production Batch Generation and Quadratic Acceleration are unchanged.
