"""
CLI: quadratic-acceleration workspace batch (flat samples + full artifacts).

Professor reference defaults (60 mph figure):
  speed = 26.8224 m/s  (60 × 0.44704)
  duration = 30 s
  CPA distances = 50, 25, 10 m
  straight path, KiaSportage, kinematic Doppler (accel = 0)

Examples:
  # Full pack for prof (panel PNG + 3 flat samples with all spectrograms/npy)
  python -m workspace.quadratic_acceleration.run_batch --prof-pack --name prof_kia_60mph

  # Override speed/distance/duration from CLI
  python -m workspace.quadratic_acceleration.run_batch --prof-pack --speed-mph 55 --distances 40,20,8

  # Custom exploratory batch (RPM coupling)
  python -m workspace.quadratic_acceleration.run_batch --clips 10 --src-model rpm_quadratic --accel-min -2 --accel-max 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import core.config  # noqa: F401

# 60 mph → m/s (prof uses 26.8224 m/s)
MPH_TO_MPS = 0.44704
PROF_SPEED_MPH = 60.0
PROF_SPEED_MPS = round(PROF_SPEED_MPH * MPH_TO_MPS, 4)  # 26.8224
PROF_DURATION_S = 30.0
PROF_DISTANCES_M = [50.0, 25.0, 10.0]
PROF_VEHICLE = "KiaSportage"


def mph_to_mps(mph: float) -> float:
    return round(float(mph) * MPH_TO_MPS, 6)


def _parse_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_floats(raw: str | None) -> list[float]:
    return [float(x.strip()) for x in _parse_list(raw)]


def _resolve_speed_mps(args: argparse.Namespace) -> float:
    if args.speed_mps is not None:
        return float(args.speed_mps)
    mph = PROF_SPEED_MPH if args.speed_mph is None else float(args.speed_mph)
    return mph_to_mps(mph)


def build_fixed_slots(
    distances_m: list[float],
    speed_mps: float,
    duration_s: float,
    *,
    vehicle: str,
    acceleration: float,
    angle_deg: float,
    temp_c: float,
    humidity: float,
) -> list[dict]:
    slots = []
    for d_m in distances_m:
        slots.append({
            "vehicle_name": vehicle,
            "path_type": "straight",
            "speed": speed_mps,
            "distance": float(d_m),
            "duration": duration_s,
            "acceleration": acceleration,
            "angle": angle_deg,
            "temperature": temp_c,
            "humidity": humidity,
            "simulation_mode": "cv" if abs(acceleration) < 1e-9 else "ca",
        })
    return slots


def build_config(args: argparse.Namespace) -> dict:
    vehicles = _parse_list(args.vehicles) or [args.vehicle]
    paths = _parse_list(args.paths) or ["straight"]

    speed_mps = _resolve_speed_mps(args)
    distances_m = _parse_floats(args.distances) if args.distances else None

    use_fixed = bool(
        args.prof_reference
        or args.prof_pack
        or args.fixed_kinematics
        or distances_m
    )

    if use_fixed and not distances_m:
        distances_m = list(PROF_DISTANCES_M)

    if use_fixed:
        n_clips = len(distances_m)
        speed_cfg = {"randomize": False, "min": speed_mps, "max": speed_mps, "value": speed_mps}
        dist_cfg = {"randomize": False, "min": 0.0, "max": 0.0, "value": 0.0}
        dur = args.duration if args.duration is not None else PROF_DURATION_S
        duration_cfg = {"randomize": False, "min": dur, "max": dur, "value": dur}
        accel = float(args.accel_fixed if args.accel_fixed is not None else 0.0)
        accel_cfg = {"randomize": False, "min": accel, "max": accel, "value": accel}
        sim_mode = "cv" if abs(accel) < 1e-9 else "ca"
        fixed_slots = build_fixed_slots(
            distances_m,
            speed_mps,
            dur,
            vehicle=args.vehicle,
            acceleration=accel,
            angle_deg=float(args.angle),
            temp_c=float(args.temp),
            humidity=float(args.humidity),
        )
        paths = ["straight"]
    else:
        n_clips = args.clips
        speed_cfg = {
            "randomize": True,
            "min": args.speed_min if args.speed_min is not None else 10.0,
            "max": args.speed_max if args.speed_max is not None else 50.0,
        }
        dist_cfg = {
            "randomize": True,
            "min": args.distance_min,
            "max": args.distance_max,
        }
        duration_cfg = {
            "randomize": True,
            "min": args.duration or 10.0,
            "max": args.duration or 10.0,
        }
        accel_cfg = {
            "randomize": True,
            "min": args.accel_min,
            "max": args.accel_max,
            "value": 0.0,
        }
        sim_mode = "ca"
        fixed_slots = None

    src_model = args.src_model
    if args.prof_reference or args.prof_pack:
        if args.src_model == "rpm_linear" and not args.keep_rpm_model:
            src_model = "doppler_only"

    cfg = {
        "simulation_mode": sim_mode,
        "output": {
            "format": args.format,
            "path": args.out_root,
            "spectrogram_type": args.spectrogram,
            "generate_diagnostics": not args.no_diagnostics,
        },
        "vehicles": {"selected": vehicles},
        "paths": {"selected": paths},
        "speed": speed_cfg,
        "distance": dist_cfg,
        "duration": duration_cfg,
        "angle": {
            "randomize": not use_fixed,
            "min": args.angle_min,
            "max": args.angle_max,
        },
        "atmosphere": {
            "randomize": not use_fixed,
            "temp_min": args.temp_min,
            "temp_max": args.temp_max,
            "hum_min": args.hum_min,
            "hum_max": args.hum_max,
            "add_air_noise": args.air_noise,
        },
        "acceleration": accel_cfg,
        "batch": {
            "total_clips": n_clips,
            "mode": "auto",
            "name": args.name or "",
        },
        "benchmarks": {"enabled": False, "selected": [], "params": {}},
        "workspace": {
            "enabled": True,
            "kind": "quadratic_acceleration_testing",
            "src_model": src_model,
            "coupling_k1": args.k1,
            "coupling_k2": args.k2,
            "v_ref_mps": args.v_ref,
            "pitch_clamp_min": args.pitch_clamp_min,
            "pitch_clamp_max": args.pitch_clamp_max,
        },
    }
    if fixed_slots:
        cfg["workspace"]["fixed_slots"] = fixed_slots
    return cfg


def run_prof_pack(args: argparse.Namespace) -> int:
    """Distance panel figure + flat workspace samples (+ optional abs grid)."""
    speed_mps = _resolve_speed_mps(args)
    mph = args.speed_mph if args.speed_mph is not None else PROF_SPEED_MPH
    distances_m = _parse_floats(args.distances) or list(PROF_DISTANCES_M)
    duration_s = float(args.duration if args.duration is not None else PROF_DURATION_S)
    pack_name = args.name or f"prof_{int(mph)}mph_{int(duration_s)}s"
    pack_dir = os.path.join(args.out_root, pack_name)
    os.makedirs(pack_dir, exist_ok=True)

    manifest = {
        "pack": pack_name,
        "speed_mph": mph,
        "speed_mps": speed_mps,
        "duration_s": duration_s,
        "distances_m": distances_m,
        "vehicle": args.vehicle,
        "outputs": {},
    }

    print(f"=== Prof pack → {pack_dir}")
    print(f"    {mph} mph = {speed_mps} m/s, duration {duration_s} s, d = {distances_m}")

    # 1) Stacked spectrogram panel (reference figure)
    panel_dir = os.path.join(pack_dir, "distance_panel")
    from workspace.quadratic_acceleration.distance_spectrogram_panel import run_distance_spectrogram_panel

    panel_summary = run_distance_spectrogram_panel(
        distances_m=distances_m,
        speed_mph=mph,
        duration_s=duration_s,
        vehicle_name=args.vehicle,
        audio_path=args.audio,
        out_dir=panel_dir,
        max_y_freq=float(args.max_freq),
        save_wavs=True,
    )
    manifest["outputs"]["distance_panel"] = panel_dir
    manifest["outputs"]["distance_panel_png"] = panel_summary.get("png")

    # 2) Three flat workspace samples (WAV + npy + 3 spectrograms + overlay + path plot)
    args.prof_reference = True
    args.out_root = pack_dir
    args.name = "workspace_samples"
    args.distances = ",".join(str(d) for d in distances_m)
    config = build_config(args)
    ws = config.setdefault("workspace", {})
    ws["enabled"] = True

    from core.batch_runner import build_batch_context, run_standard_batch

    ctx = build_batch_context(config)
    print(f"=== Workspace samples → {ctx['batch_dir']}")
    batch_result = run_standard_batch(ctx, time.time())
    manifest["outputs"]["workspace_batch"] = ctx["batch_dir"]
    manifest["outputs"]["batch_result"] = batch_result

    # 3) Optional analysis-by-synthesis grid on uploaded recording
    if args.audio and os.path.isfile(args.audio):
        abs_dir = os.path.join(pack_dir, "abs_grid")
        from workspace.quadratic_acceleration.analysis_by_synthesis_grid import run_analysis_by_synthesis_grid

        print(f"=== (v,d) grid → {abs_dir}")
        abs_result = run_analysis_by_synthesis_grid(
            args.audio,
            abs_dir,
            cpa_windows=sorted({0.5, 1.0, 2.0, float(args.cpa_window)}),
            metric=args.abs_metric,
            save_wavs=args.abs_save_wavs,
            synthetic_gt=(
                {"speed_kph": float(args.gt_v_kph), "distance_m": float(args.gt_d_m)}
                if args.gt_v_kph is not None and args.gt_d_m is not None
                else None
            ),
        )
        manifest["outputs"]["abs_grid"] = abs_dir
        manifest["outputs"]["abs_grid_summary"] = abs_result

    manifest_path = os.path.join(pack_dir, "prof_pack_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== Done. Manifest: {manifest_path}")
    print("Send to prof:")
    print(f"  • {panel_dir}/distance_spectrogram_panel.png")
    print(f"  • {panel_dir}/flyby_d*.wav")
    print(f"  • {ctx['batch_dir']}/audio_clips/sample_*/  (flat: wav, npy, spectrograms, overlay)")
    if args.audio:
        print(f"  • {pack_dir}/abs_grid/  (heatmaps, CSVs, README_abs_grid.md)")
    return 0 if batch_result.get("success") else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description="Workspace batch CLI (quadratic/RPM sandbox + professor reference preset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", help="JSON config (overrides flags; workspace.enabled forced on)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--prof-pack",
        action="store_true",
        help="Build full prof deliverable: distance panel + 3 flat samples (+ abs grid if --audio)",
    )
    mode.add_argument(
        "--prof-reference",
        action="store_true",
        help="Fixed kinematics only: 60 mph, 30 s, distances 50/25/10 (overridable)",
    )

    p.add_argument("--clips", type=int, default=3, help="Clip count (ignored if --distances or --prof-*)")
    p.add_argument("--name", default="", help="Output batch / pack folder name")
    p.add_argument("--out-root", default="static/workspace_outputs", dest="out_root")

    p.add_argument("--vehicle", default=PROF_VEHICLE, help="Library vehicle for synthesis")
    p.add_argument("--vehicles", default="", help="Comma-separated vehicles (batch distribution)")
    p.add_argument("--paths", default="straight", help="Path types (prof reference uses straight only)")
    p.add_argument("--audio", default=None, help="Pass-by WAV for abs grid inside --prof-pack")

    p.add_argument(
        "--speed-mph",
        type=float,
        default=None,
        help=f"Speed in mph (prof modes default {PROF_SPEED_MPH} → {PROF_SPEED_MPS} m/s)",
    )
    p.add_argument("--speed-mps", type=float, default=None, help="Speed in m/s (overrides --speed-mph)")
    p.add_argument("--speed-min", type=float, default=None)
    p.add_argument("--speed-max", type=float, default=None)
    p.add_argument(
        "--distances",
        default=",".join(str(int(d)) for d in PROF_DISTANCES_M),
        help="Comma-separated CPA distances (m); sets clip count when fixed",
    )
    p.add_argument("--duration", type=float, default=PROF_DURATION_S, help="Clip duration (s)")
    p.add_argument("--max-freq", type=float, default=800.0, dest="max_freq", help="Panel STFT max Hz")
    p.add_argument("--angle", type=float, default=0.0, help="Straight-path angle (deg) for fixed runs")
    p.add_argument("--angle-min", type=float, default=-45.0)
    p.add_argument("--angle-max", type=float, default=45.0)
    p.add_argument("--temp", type=float, default=20.0)
    p.add_argument("--humidity", type=float, default=50.0)
    p.add_argument("--temp-min", type=float, default=15.0)
    p.add_argument("--temp-max", type=float, default=35.0)
    p.add_argument("--hum-min", type=float, default=30.0)
    p.add_argument("--hum-max", type=float, default=70.0)

    p.add_argument("--accel-min", type=float, default=-3.5)
    p.add_argument("--accel-max", type=float, default=2.5)
    p.add_argument(
        "--accel-fixed",
        type=float,
        default=None,
        help="Fixed acceleration (m/s²); prof reference uses 0 (CV Doppler)",
    )
    p.add_argument("--fixed-kinematics", action="store_true", help="Use --speed-mph and --distances as fixed")

    p.add_argument(
        "--src-model",
        dest="src_model",
        choices=("doppler_only", "rpm_linear", "rpm_quadratic"),
        default="doppler_only",
    )
    p.add_argument("--keep-rpm-model", action="store_true", help="Do not force doppler_only in prof modes")
    p.add_argument("--k1", type=float, default=0.35)
    p.add_argument("--k2", type=float, default=0.25)
    p.add_argument("--v-ref", type=float, default=30.0, dest="v_ref")
    p.add_argument("--pitch-clamp-min", type=float, default=0.35, dest="pitch_clamp_min")
    p.add_argument("--pitch-clamp-max", type=float, default=2.5, dest="pitch_clamp_max")
    p.add_argument("--distance-min", type=float, default=5.0)
    p.add_argument("--distance-max", type=float, default=100.0)

    p.add_argument("--cpa-window", type=float, default=1.0, dest="cpa_window")
    p.add_argument("--abs-metric", default="both", dest="abs_metric", choices=("both", "l1", "l2"))
    p.add_argument("--abs-save-wavs", action="store_true", help="Save all 169 grid WAVs (large)")
    p.add_argument("--gt-v-kph", type=float, default=None, dest="gt_v_kph")
    p.add_argument("--gt-d-m", type=float, default=None, dest="gt_d_m")

    p.add_argument("--air-noise", action="store_true")
    p.add_argument("--format", choices=("wav", "mp3"), default="wav")
    p.add_argument("--spectrogram", choices=("cqt", "stft", "mel"), default="cqt")
    p.add_argument("--no-diagnostics", action="store_true")
    args = p.parse_args()

    if args.prof_pack:
        return run_prof_pack(args)

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = build_config(args)

    ws = config.setdefault("workspace", {})
    ws["enabled"] = True
    ws.setdefault("kind", "quadratic_acceleration_testing")

    from core.batch_runner import build_batch_context, run_standard_batch

    ctx = build_batch_context(config)
    speed_note = _resolve_speed_mps(args)
    print(f"Batch: {ctx['batch_id']}  clips: {ctx['total_clips']}  v={speed_note} m/s  out: {ctx['batch_dir']}")
    result = run_standard_batch(ctx, time.time())
    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
