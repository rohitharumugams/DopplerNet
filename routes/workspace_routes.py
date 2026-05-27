"""
Workspace-only API routes.

These endpoints are only called from the Workspace UI tab. No other mode uses them.
"""

from __future__ import annotations

import os
import threading
import traceback
import uuid
from datetime import datetime

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

workspace_bp = Blueprint("workspace", __name__)

_ABS_JOBS: dict[str, dict] = {}
_ABS_LOCK = threading.Lock()


def _abs_job_update(job_id: str, **kwargs) -> None:
    with _ABS_LOCK:
        if job_id in _ABS_JOBS:
            _ABS_JOBS[job_id].update(kwargs)


@workspace_bp.route("/api/workspace/abs_grid", methods=["POST"])
def workspace_abs_grid():
    """Run analysis-by-synthesis (v,d) grid on an uploaded recording (Workspace only)."""
    try:
        audio_file = request.files.get("audio")
        if not audio_file or not audio_file.filename:
            return jsonify({"error": "No audio file uploaded"}), 400

        out_root = request.form.get("out_dir", "static/workspace_outputs/abs_grid").strip()
        metric = request.form.get("metric", "both")
        norm_mode = request.form.get("norm", "global_max")
        cpa_window = float(request.form.get("cpa_window", 1.0))
        save_wavs = request.form.get("save_wavs", "false").lower() in ("1", "true", "yes")
        gt_v = request.form.get("gt_v_kph")
        gt_d = request.form.get("gt_d_m")

        job_name = request.form.get("job_name", "").strip()
        if not job_name:
            job_name = f"abs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        safe_name = "".join(c for c in job_name if c.isalnum() or c in ("-", "_")).strip() or "abs_grid"
        job_dir = os.path.join(out_root, safe_name)
        os.makedirs(job_dir, exist_ok=True)

        upload_name = secure_filename(audio_file.filename) or "recording.wav"
        audio_path = os.path.join(job_dir, upload_name)
        audio_file.save(audio_path)

        job_id = uuid.uuid4().hex[:12]
        with _ABS_LOCK:
            _ABS_JOBS[job_id] = {
                "status": "running",
                "progress": 0,
                "total": 169,
                "message": "Starting grid...",
                "out_dir": job_dir,
            }

        cpa_windows = sorted(set([0.5, 1.0, 2.0, cpa_window]))
        synthetic_gt = None
        if gt_v and gt_d:
            try:
                synthetic_gt = {"speed_kph": float(gt_v), "distance_m": float(gt_d)}
            except ValueError:
                pass

        def _run():
            try:
                from workspace.analysis_by_synthesis_grid import run_analysis_by_synthesis_grid

                def prog(i, t, v, d):
                    _abs_job_update(
                        job_id,
                        progress=i,
                        total=t,
                        message=f"v={v} kph, d={d} m ({i}/{t})",
                    )

                results = run_analysis_by_synthesis_grid(
                    audio_path,
                    job_dir,
                    cpa_windows=cpa_windows,
                    metric=metric,
                    norm_mode=norm_mode,
                    save_wavs=save_wavs,
                    synthetic_gt=synthetic_gt,
                    progress_callback=prog,
                )
                _abs_job_update(
                    job_id,
                    status="completed",
                    progress=results.get("total", 1) if isinstance(results.get("total"), int) else 1,
                    total=1,
                    message="Grid complete",
                    results=_safe_json(results),
                )
            except Exception as exc:
                traceback.print_exc()
                _abs_job_update(job_id, status="failed", message=str(exc))

        threading.Thread(target=_run, daemon=True).start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "out_dir": job_dir.replace("\\", "/"),
            "status_url": f"/api/workspace/abs_grid/status/{job_id}",
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@workspace_bp.route("/api/workspace/abs_grid/status/<job_id>", methods=["GET"])
def workspace_abs_grid_status(job_id: str):
    with _ABS_LOCK:
        job = _ABS_JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


@workspace_bp.route("/api/workspace/distance_panel", methods=["POST"])
def workspace_distance_panel():
    """Stacked distance spectrograms (50/25/10 m style figure) — Workspace only."""
    try:
        vehicle = request.form.get("vehicle", "KiaSportage").strip()
        distances_raw = request.form.get("distances", "50,25,10")
        speed_mph = float(request.form.get("speed_mph", 60))
        duration_s = float(request.form.get("duration", 30))
        max_freq = float(request.form.get("max_freq", 800))
        out_root = request.form.get("out_dir", "static/workspace_outputs/distance_panel").strip()
        job_name = request.form.get("job_name", "").strip() or f"panel_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        safe_name = "".join(c for c in job_name if c.isalnum() or c in ("-", "_")).strip()
        job_dir = os.path.join(out_root, safe_name)
        os.makedirs(job_dir, exist_ok=True)

        audio_path = None
        audio_file = request.files.get("audio")
        if audio_file and audio_file.filename:
            audio_path = os.path.join(job_dir, secure_filename(audio_file.filename) or "source.wav")
            audio_file.save(audio_path)
            vehicle = None

        from workspace.distance_spectrogram_panel import run_distance_spectrogram_panel

        summary = run_distance_spectrogram_panel(
            distances_m=[float(x.strip()) for x in distances_raw.split(",") if x.strip()],
            speed_mph=speed_mph,
            duration_s=duration_s,
            audio_path=audio_path,
            vehicle_name=vehicle if not audio_path else None,
            out_dir=job_dir,
            max_y_freq=max_freq,
        )
        return jsonify({"success": True, "out_dir": job_dir.replace("\\", "/"), "results": _safe_json(summary)})
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


def _safe_json(obj):
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]
    return obj
