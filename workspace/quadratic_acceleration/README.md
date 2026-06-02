# Quadratic Acceleration Mode

Workspace sub-mode for **observer-centric** pass-by synthesis with optional **RPM / acceleration-aware source pitch** before geometric Doppler. Use this path to experiment with $g(v)$ coupling, acceleration sweeps, professor reference batches, and analysis-by-synthesis $(v, d)$ grids — without changing production Batch Generation or benchmarks.

## Status

**Active** — full UI under Workspace → **Quadratic Acceleration Mode**.

## Isolation

| Rule | Detail |
|------|--------|
| Enable flag | `workspace.enabled = true` in batch config (set automatically by CLI/UI) |
| Synthesis entry | `audio/workspace_generation.py` → `generate_single_clip_workspace` |
| Not shared with | `workspace/emitter_centric/` (co-moving source frame) |
| Production | CV/CA batch, B1–B10, and `audio/generation.py` unchanged when workspace is off |

## Package layout

```text
workspace/quadratic_acceleration/
├── README.md                      # This file
├── __init__.py                    # Public synthesis helpers
├── abs_synthesis.py               # Dedopplerize, stitch, straight pass-by warp
├── run_batch.py                   # CLI batch + professor --prof-pack
├── analysis_by_synthesis_grid.py  # (v, d) calibration grid on real audio
└── distance_spectrogram_panel.py  # Stacked distance spectrogram figure
```

Shared API gateway: `routes/workspace_routes.py` (quadratic handlers only; emitter-centric has separate routes).

## Physics (summary)

- **Baseline:** $f_{\text{obs}} \approx f_{\text{src}} \times c/(c + v_r(t))$ on a **fixed roadside observer** grid (same core as production).
- **Workspace addition:** optional **$g(v/v_{\text{ref}})$** on the source before Doppler — linear or quadratic RPM models — so acceleration produces **nonlinear** spectral evolution under $v(t) = v_0 + at$.
- **Not emitter-centric:** geometric Doppler is still applied; this is not a co-moving dashboard microphone (see `workspace/emitter_centric/plan.md` §13).

Source models (UI / config): `doppler_only`, `rpm_linear`, `rpm_quadratic` with gains $k_1$, $k_2$.

## Web UI

1. Open **Workspace** tab → **Quadratic Acceleration Mode**.
2. Configure vehicles, acceleration range, source model, spectrogram type, save path.
3. **Generate Workspace Batch** → flat samples under `static/workspace_outputs/`.

Optional: **Run (v, d) Grid** on an uploaded recording (analysis-by-synthesis).

## CLI

### Professor reference pack (60 mph, 30 s, CPA 50 / 25 / 10 m)

```bash
python -m workspace.quadratic_acceleration.run_batch --prof-pack --name prof_kia_60mph
```

Defaults: **26.8224 m/s** (60 mph), **30 s**, `KiaSportage`, kinematic Doppler (`accel=0`). Override: `--speed-mph 55 --distances 40,20,8 --duration 25`.

Optional recording for embedded $(v,d)$ grid: `--audio path/to/passby.wav`

### Exploratory quadratic / RPM batch

```bash
python -m workspace.quadratic_acceleration.run_batch --clips 20 --name quad_test \
  --src-model rpm_quadratic --k1 0.35 --k2 0.25 \
  --accel-min -3.5 --accel-max 2.5
```

### Analysis-by-synthesis $(v, d)$ grid

```bash
python -m workspace.quadratic_acceleration.analysis_by_synthesis_grid \
  --audio path/to/passby.wav \
  --out_dir static/workspace_outputs/abs_grid/kia_run \
  --metric both --cpa_window 1.0
```

Pipeline: first **1.0 s** far-field segment → dedopplerize → repeat-stitch → synthesize per grid cell → L1/L2 vs reference STFT (full clip + CPA windows 0.5 / 1.0 / 2.0 s). Optional synthetic GT: `--gt_v_kph 50 --gt_d_m 4.5`.

### Distance spectrogram panel (offline figure)

```bash
python -m workspace.quadratic_acceleration.distance_spectrogram_panel \
  --distances 50,25,10 --speed-mph 60 --duration 30 --vehicle KiaSportage
```

## Outputs

### Workspace batch (`static/workspace_outputs/` by default)

Per sample: **flat** folder `audio_clips/sample_XXXXXXX/` (no Common/Essential split).

| Artifact | Role |
|----------|------|
| `*.wav` | Synthesized clip |
| `cqt.npy` / `stft.npy` / `mel.npy` | Spectrogram features |
| `frequency.npy`, `dfdt.npy`, `rms.npy`, `spec_topk.npy`, `time.npy`, `kinematics.npy` | Frame traces |
| `freq_ratios.npy`, `amplitudes.npy` | Doppler ratio + path gain |
| `src_pitch_curve.npy` | $g(v)$ when RPM model enabled |
| `*_spectrogram_cqt.png`, `*_spectrogram_wideband.png`, `*_spectrogram_narrowband.png` | Views |
| `*_diagnostics_overlay.png` | CQT + ratio + gain + $g(v)$ or $v(t)$ |
| `*_path_plot.png`, `metadata.json` | Geometry + parameters |

### Professor pack layout

`static/workspace_outputs/<pack_name>/`

- `distance_panel/` — stacked spectrogram PNG + optional WAVs
- `workspace_samples/` — batch with three distance slots
- `abs_grid/` — optional if `--audio` provided (heatmaps, CSVs; local `README_abs_grid.md` may be generated there)

### $(v, d)$ grid

Under `--out_dir`: heatmaps, marginals, CSVs, `README_abs_grid.md` (generated notes for that run).

## Interpretation tips

- Broad $(v, d)$ valleys on **real** recordings are expected; argmin alone is not ground truth.
- Use **CPA-local** metrics and **synthetic GT** runs before trusting global L1/L2 minima.
- Compare with production batch only when `workspace.enabled` is false — otherwise you are on the experimental path.

## Related documentation

- Workspace overview: `workspace/README.md`
- Emitter-centric (separate sub-mode): `workspace/emitter_centric/README.md`
- Simulator state: `ref_docs/context.md` (local; not in git by default)

## Roadmap (workspace-only)

1. CPA-windowed grid + synthetic GT — implemented.
2. Rich spectrogram diagnostics — implemented in batch + grid.
3. Stitching audit (repeat-stitch vs overlap-extend).
4. $g(v)$ + gain-law calibration.
5. Optional move of `audio/workspace_generation.py` into this package (behavior unchanged).
