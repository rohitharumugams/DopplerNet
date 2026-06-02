#!/usr/bin/env python
"""CLI for emitter-centric straight CV synthesis (workspace sandbox)."""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from workspace.emitter_centric.synthesis import run_straight_cv_job


def main() -> None:
    p = argparse.ArgumentParser(description="Emitter-centric straight CV synthesis")
    p.add_argument("--speed-mps", type=float, default=30.0)
    p.add_argument("--distance-m", type=float, default=15.0)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--angle-deg", type=float, default=0.0)
    p.add_argument("--vehicle", default="KiaSportage")
    p.add_argument("--audio", default=None, help="Optional source WAV path")
    p.add_argument("--temp-c", type=float, default=20.0)
    p.add_argument("--humidity", type=float, default=50.0)
    p.add_argument("--propagation-delay", action="store_true")
    p.add_argument("--job-name", default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    summary = run_straight_cv_job(
        speed_mps=args.speed_mps,
        distance_m=args.distance_m,
        duration_s=args.duration,
        angle_deg=args.angle_deg,
        vehicle_name=None if args.audio else args.vehicle,
        audio_path=args.audio,
        temp_c=args.temp_c,
        humidity=args.humidity,
        use_propagation_delay=args.propagation_delay,
        out_dir=args.out_dir,
        job_name=args.job_name,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
