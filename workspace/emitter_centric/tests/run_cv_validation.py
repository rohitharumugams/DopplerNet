#!/usr/bin/env python3
"""
Run constant-velocity validation on an emitter-centric batch folder.

Usage (from repo root):
  python -m workspace.emitter_centric.tests.run_cv_validation
  python -m workspace.emitter_centric.tests.run_cv_validation --batch static/workspace_outputs/emitter_centric/test_batch
  python -m workspace.emitter_centric.tests.run_cv_validation --json workspace/emitter_centric/tests/results/test_batch.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Allow running as script: python workspace/emitter_centric/tests/run_cv_validation.py
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from workspace.emitter_centric.tests.cv_validation_checks import run_all_checks

DEFAULT_BATCH = os.path.join(
    _REPO_ROOT,
    "static",
    "workspace_outputs",
    "emitter_centric",
    "test_batch",
)


def _default_json_path(batch_root: str) -> str:
    batch_id = os.path.basename(os.path.normpath(batch_root)) or "batch"
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{batch_id}_cv_validation.json")


def _print_report(payload: dict) -> None:
    summary = payload["summary"]
    print("=" * 72)
    print("Emitter-Centric CV Validation")
    print("=" * 72)
    print(f"Batch:   {summary['batch_root']}")
    print(f"ID:      {summary.get('batch_id')}")
    print(f"Clips:   {summary['n_clips']}")
    print(
        f"Results: PASS={summary['pass']}  FAIL={summary['fail']}  "
        f"INCONCLUSIVE={summary['inconclusive']}  SKIP={summary['skip']}"
    )
    print("-" * 72)
    for chk in payload["checks"]:
        print(f"[{chk['status']:12}] {chk['check_id']}  {chk['title']}")
        print(f"             {chk['justification']}")
    print("-" * 72)
    if summary["fail"]:
        print("OVERALL: FAIL - see failed checks above.")
        return
    if summary["inconclusive"]:
        print("OVERALL: INCONCLUSIVE - core checks passed; review flagged items.")
        return
    print("OVERALL: PASS - all executed checks passed.")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Validate emitter-centric constant-velocity batch outputs.",
    )
    parser.add_argument(
        "--batch",
        default=DEFAULT_BATCH,
        help=f"Path to batch root (default: {DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Write full JSON results to this path (default: tests/results/<batch_id>_cv_validation.json)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not write JSON output.",
    )
    args = parser.parse_args()

    batch_root = os.path.abspath(args.batch)
    meta_path = os.path.join(batch_root, "overall_metadata.json")
    if not os.path.isfile(meta_path):
        print(f"Error: not a batch folder (missing overall_metadata.json): {batch_root}", file=sys.stderr)
        return 2

    payload = run_all_checks(batch_root)
    payload["run"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "constant_velocity",
    }

    _print_report(payload)

    if not args.no_json:
        json_path = args.json_path or _default_json_path(batch_root)
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"JSON written: {json_path}")

    summary = payload["summary"]
    if summary["fail"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
