# Emitter-centric validation tests

Read-only checks on **existing** batch outputs (no regeneration, no synthesis changes).

## Run your `test_batch` dataset

From the **repo root** (`DopplerSim/`):

```bash
python -m workspace.emitter_centric.tests.run_cv_validation
```

That is the only command you need. It validates:

`static/workspace_outputs/emitter_centric/test_batch`

Console output lists each check as **PASS**, **FAIL**, **INCONCLUSIVE**, or **SKIP**, plus an overall line at the end.

Full results are also written to:

`workspace/emitter_centric/tests/results/test_batch_cv_validation.json`

Exit code **0** = no failures; **1** = at least one failed check.

---

## Validation mode (current)

| Mode | Scope | Batch |
|------|--------|--------|
| **Constant velocity (CV)** | `acceleration = 0`, co-moving emitter vs observer comparison | `test_batch` (default) |

**What CV mode checks (summary):**

### A — Metadata & batch integrity
- **A1–A6** CV config, co-moving frame flags, no failed clips, metadata consistency
- **A7** Full emitter/observer `sample_metadata.json` parameter parity (speed, CPA, path, atmosphere, duration)

### B — Emitter (co-moving) physics
- **B1–B4** Constant speed, flat `src_pitch_curve`, no `freq_ratios.npy`
- **B5** `src_pitch_curve` matches recomputed RPM formula `g(v)=v/v_ref`
- **B6** Same vehicle @ CV: emitter spectral centroid stable across speeds (RPM wiring sanity)
- **B7** Emitter WAV spectral centroid near raw vehicle source (CV, g=1)
- **B8** WAV sample count matches `SR * duration` within +/-1 sample

### C — Scene geometry
- **C1–C3** Matching `source_positions.npy`, path arc length, path plots

### D — Cross-branch audio comparison
- **D1–D3** Emitter vs observer waveforms; observer Doppler swing; straight clips not coupled to geometric Doppler
- **D5** Non-parity comparison metrics (expected)

### E / F — Spectral / harmonic heuristics
- **E2, E3, F1** RMS envelope, harmonic stability, frequency.npy std (subset checks)

### G — Comparison artifacts
- **G1** Comparison folder complete per sample

### H — Negative & sidecar hygiene
- **H1** No Doppler sidecars on emitter
- **H2** `time.npy` strictly increasing (monotonic arrival proxy)
- **H3** All `.npy` sidecars finite (both branches)
- **H4** `source_positions.npy` shape `(N, 3)`

### O — Observer physics ground truth (straight clips)
- **O4** `freq_ratios` within analytic `c/(c+/-v)` bounds
- **O5** `freq_ratios` match `straight_cv_kinematics_with_c` recomputation
- **O6** CPA timing: `freq_ratio~1` aligns with nominal CPA time
- **O7** Golden straight analytic regression vs `golden/cv_straight_analytic.json`

### M — Batch coverage
- **M1** Batch exercises configured speed range
- **M2** At least one clip per configured path type

### R — Determinism
- **R1** In-memory co-moving synthesizer bit-identical on repeat

**Not in scope yet:** full batch seed reproducibility (hash all outputs twice), acceleration sweeps, Formulation-1 parity, automated pytest.

---

## Result statuses

| Status | Meaning |
|--------|---------|
| **PASS** | Condition satisfied for this batch |
| **FAIL** | Clear violation — review check ID and JSON `details` |
| **INCONCLUSIVE** | Ambiguous metric (often vehicle- or path-specific); needs manual review |
| **SKIP** | Check not applicable (e.g. no straight clips for a straight-only test) |

Thresholds live in `cv_validation_helpers.py` if you tune later.

---

## Files

| File | Role |
|------|------|
| `run_cv_validation.py` | Entry point (default batch = `test_batch`) |
| `cv_validation_checks.py` | Check definitions |
| `cv_validation_helpers.py` | Shared helpers and thresholds |
| `cv_validation_physics.py` | Physics helpers (centroid, kinematics, determinism) |
| `golden/cv_straight_analytic.json` | Reference spec for O7 |
| `results/` | JSON output (gitignored) |

After you run validation, use the JSON to produce a human-readable verification report (e.g. `verification_results.md`).
