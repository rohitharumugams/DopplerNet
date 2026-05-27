# Workspace mode (isolated)

Experimental sandbox for **acceleration-aware source frequency** modeling. Does not change Batch Generation, benchmarks, or global CV/CA behavior unless `workspace.enabled` is set.

## Quadratic Acceleration Testing (batch)

**Web:** Workspace tab → configure source model + acceleration range → **Generate Workspace Batch**.

**CLI (professor 60 mph reference — full pack):**

```bash
python -m workspace.run_batch --prof-pack --name prof_kia_60mph
```

Defaults: **60 mph = 26.8224 m/s**, **30 s**, CPA **50 / 25 / 10 m**, `KiaSportage`, kinematic Doppler (`accel=0`).
Override: `--speed-mph 55 --distances 40,20,8 --duration 25`.

Optional real recording for (v,d) grid: `--audio path/to/kia_passby.wav`

**CLI (exploratory quadratic/RPM batch):**

```bash
python -m workspace.run_batch --clips 20 --name quad_test \
  --src-model rpm_quadratic --k1 0.35 --k2 0.25 \
  --accel-min -3.5 --accel-max 2.5
```

### Physics

- Standard path: `f_obs ≈ f_src × Doppler(v_r(t))` with fixed `f_src`.
- Workspace adds optional **g(v/v_ref)** before Doppler: `f_src(t)` tracks RPM/speed under kinematic `v(t)`, producing nonlinear/quadratic evolution under acceleration.

### Per-sample outputs (`audio_clips/sample_XXXXXXX/` — flat, no Common/Essential)

| File | Role |
|------|------|
| `*.wav` | Synthesized clip |
| `cqt.npy` / `stft.npy` / `mel.npy` | Batch-style spectrogram (per UI type) |
| `frequency.npy`, `dfdt.npy`, `rms.npy`, `spec_topk.npy`, `time.npy`, `kinematics.npy` | Frame features |
| `freq_ratios.npy`, `amplitudes.npy` | Doppler ratio + path gain |
| `src_pitch_curve.npy` | g(v) curve when RPM model enabled |
| `*_spectrogram_cqt.png` | CQT view |
| `*_spectrogram_wideband.png` | STFT 0–8 kHz |
| `*_spectrogram_narrowband.png` | STFT 0–1.2 kHz (engine band) |
| `*_diagnostics_overlay.png` | CQT + ratio + gain + g(v) or v(t) |
| `spectrogram.png`, `frequency.png`, … | Batch diagnostics (if enabled) |
| `*_path_plot.png`, `metadata.json` | Path + parameters |

## Analysis-by-synthesis (v, d) calibration

Professor-style grid on a **real** pass-by (or synthetic GT):

1. First 1.0 s far-field source (no heavy preprocessing).
2. For each (v, d) on 13×13 grid: dedopplerize → repeat-stitch → synthesize → CPA peak align.
3. L1 and L2 on |S_ref| − |S_gen| (full clip + CPA windows 0.5 / 1.0 / 2.0 s).
4. Heatmaps, marginals, CSVs; optional synthetic GT overlay.

**Web:** Workspace → **Run (v, d) Grid**.

**CLI:**

```bash
python -m workspace.analysis_by_synthesis_grid \
  --audio path/to/passby.wav \
  --out_dir static/workspace_outputs/abs_grid/kia_run \
  --metric both --cpa_window 1.0
```

Optional synthetic check: `--gt_v_kph 50 --gt_d_m 4.5`.

### Interpretation

- Broad (v, d) valleys are expected on real data; argmins are **not** ground truth alone.
- Use **CPA-local** metrics + **synthetic GT** runs to judge identifiability before trusting L1/L2 minima.

## Roadmap (workspace-only)

1. CPA-windowed grid + synthetic GT (implemented).
2. Spectrogram diagnostics / aligned overlays (batch + workspace PNGs).
3. Stitching audit (repeat-stitch vs overlap-extend).
4. g(v) + gain-law calibration (ongoing via workspace synthesis).
5. Paper-facing limits documented in grid README output.

## Legacy / offline

`distance_spectrogram_panel.py` — stacked distance comparison figure (CLI only; not in UI).
