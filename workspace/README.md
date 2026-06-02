# Workspace mode (isolated)

Experimental sandbox for research prototypes. Does **not** change Batch Generation, benchmarks, or global CV/CA behavior unless `workspace.enabled` is set.

## Sub-modes

| Folder | Mode | Documentation |
|--------|------|----------------|
| [`quadratic_acceleration/`](quadratic_acceleration/README.md) | **Active** — RPM + acceleration, $(v,d)$ grid, prof pack | Full guide in subfolder README |
| [`emitter_centric/`](emitter_centric/README.md) | Co-moving source frame + optional observer comparison | `plan.md` for blueprint & references |

Select a sub-mode from the **Workspace** landing page. Sub-modes must not cross-import or share output directories.

## Quick links

**Quadratic Acceleration**

```bash
python -m workspace.quadratic_acceleration.run_batch --prof-pack --name prof_kia_60mph
```

**Emitter-Centric**

```bash
python -m workspace.emitter_centric.run_synthesis --speed-mps 30 --distance-m 15 --duration 10
```

## Default output roots

| Sub-mode | Path |
|----------|------|
| Quadratic | `static/workspace_outputs/` |
| Emitter-centric | `static/workspace_outputs/emitter_centric/` |
| ABS grid | `static/workspace_outputs/abs_grid/` |

All under `.gitignore` as generated artifacts.
