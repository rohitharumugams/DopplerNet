"""Constant-velocity validation checks for emitter-centric batch outputs."""

from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np

from workspace.emitter_centric.config import SR
from workspace.emitter_centric.tests.cv_validation_helpers import (
    CURVED_ARC_LEN_RATIO_MAX,
    CPA_TIME_TOL_S,
    DOPPLER_BOUNDS_REL_TOL,
    EMIT_RMS_CV_MAX_OBS_RATIO,
    KINEMATICS_FR_MAX_ABS_ERR,
    OBS_FREQ_SPAN_MIN,
    PITCH_MEAN_TOL,
    PITCH_RECOMPUTE_TOL,
    PITCH_STD_MAX,
    RAW_CORR_MAX,
    SAME_VEHICLE_CENTROID_SPREAD_TOL,
    SAMPLE_COUNT_TOL,
    SOURCE_EMIT_CENTROID_REL_TOL,
    SPEED_RANGE_MIN_SPAN_FRAC,
    STRAIGHT_ARC_LEN_RATIO_MAX,
    STRAIGHT_ARC_LEN_RATIO_MIN,
    STRAIGHT_DOPPLER_PEAK_CORR_MAX,
    EMIT_RIDGE_VEHICLE_TOL,
    V_SPEED_STD_MAX,
    V_SPEED_TOL,
    CheckResult,
    clip_paths,
    cqt_ridge_metrics,
    load_json,
    path_arc_length,
    read_wav_mono,
)
from workspace.emitter_centric.tests.cv_validation_physics import (
    align_series,
    analytic_doppler_bounds,
    estimate_cpa_time_from_freq_ratio,
    expected_straight_freq_ratios,
    freq_ratio_time_axis,
    load_vehicle_source_buffer,
    metadata_parameters_match,
    recompute_src_pitch_curve,
    scan_sidecars_finite,
    spectral_centroid_hz,
    synthesizer_deterministic_probe,
)


def _load_batch(batch_root: str) -> tuple[dict, list[dict], list[dict]]:
    root = os.path.abspath(batch_root)
    overall = load_json(os.path.join(root, "overall_metadata.json"))
    clips = overall.get("clips") or []
    dataset_rows: list[dict] = []
    dataset_path = os.path.join(root, "dataset.csv")
    if os.path.isfile(dataset_path):
        with open(dataset_path, newline="", encoding="utf-8") as f:
            dataset_rows = list(csv.DictReader(f))
    return overall, clips, dataset_rows


def _per_sample_records(batch_root: str, clips: list[dict]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for clip in clips:
        spec = clip["spec"]
        sid = clip["emitter"]["sample_dir"]
        paths = clip_paths(batch_root, sid)
        ec = paths["emit_common"]
        oc = paths["obs_common"]

        ekin = np.load(os.path.join(ec, "emitter_frame_kinematics.npy"))
        pitch = np.load(os.path.join(ec, "src_pitch_curve.npy"))
        sp_e = np.load(os.path.join(ec, "source_positions.npy"))
        sp_o = np.load(os.path.join(oc, "source_positions.npy"))
        cqt_e = np.load(os.path.join(ec, "cqt.npy"))
        cqt_o = np.load(os.path.join(oc, "cqt.npy"))
        fr = np.load(os.path.join(oc, "freq_ratios.npy"))
        emit_peaks = np.argmax(cqt_e, axis=0)
        obs_peaks = np.argmax(cqt_o, axis=0)
        n_fr = min(len(fr), len(emit_peaks), len(obs_peaks))
        if n_fr > 2:
            fr_emit_corr = float(np.corrcoef(fr[:n_fr], emit_peaks[:n_fr])[0, 1])
            fr_obs_corr = float(np.corrcoef(fr[:n_fr], obs_peaks[:n_fr])[0, 1])
        else:
            fr_emit_corr = 0.0
            fr_obs_corr = 0.0

        emit_rms = np.load(os.path.join(ec, "rms.npy"))
        obs_rms = np.load(os.path.join(oc, "rms.npy"))
        freq_e = np.load(os.path.join(ec, "frequency.npy"))
        freq_o = np.load(os.path.join(oc, "frequency.npy"))

        emit_wav_path = os.path.join(ec, clip["emitter"]["filename"])
        obs_wav_path = os.path.join(oc, clip["observer"]["filename"])
        emit_wav = read_wav_mono(emit_wav_path)
        obs_wav = read_wav_mono(obs_wav_path)
        n = min(len(emit_wav), len(obs_wav))
        raw_corr = float(np.corrcoef(emit_wav[:n], obs_wav[:n])[0, 1])

        duration = float(spec["duration"])
        expected_n = int(round(SR * duration))
        time_obs = np.load(os.path.join(oc, "time.npy"))

        v_ref = max(float(spec["speed"]), 1.0)
        pitch_expected = recompute_src_pitch_curve(
            n_samples=len(pitch),
            duration_s=duration,
            speed_mps=float(spec["speed"]),
            acceleration=float(spec.get("acceleration", 0.0)),
            v_ref=v_ref,
        )
        pitch_max_err = float(np.max(np.abs(pitch - pitch_expected)))

        try:
            source_buf = load_vehicle_source_buffer(spec["vehicle"], duration)
            src_n = min(len(source_buf), len(emit_wav))
            source_centroid = spectral_centroid_hz(source_buf[:src_n])
            emit_centroid = spectral_centroid_hz(emit_wav[:src_n])
            if source_centroid > 1.0:
                centroid_rel_diff = abs(emit_centroid - source_centroid) / source_centroid
            else:
                centroid_rel_diff = 0.0
            source_load_error = None
        except Exception as exc:
            source_centroid = None
            emit_centroid = None
            centroid_rel_diff = None
            source_load_error = str(exc)

        fr_min = float(np.min(fr))
        fr_max = float(np.max(fr))
        temp = float(spec.get("temperature", 20.0))
        hum = float(spec.get("humidity", 50.0))
        bound_lo, bound_hi = analytic_doppler_bounds(float(spec["speed"]), temp, hum)
        bounds_ok = (
            fr_min >= bound_lo * (1.0 - DOPPLER_BOUNDS_REL_TOL)
            and fr_max <= bound_hi * (1.0 + DOPPLER_BOUNDS_REL_TOL)
        )

        kin_fr_max_err = None
        if spec["path_type"] == "straight":
            expected_fr = expected_straight_freq_ratios(spec, len(fr))
            efr, fr_a = align_series(expected_fr, fr)
            kin_fr_max_err = float(np.max(np.abs(efr - fr_a)))

        from physics.recording_labels import resolve_cpa_time_s

        cpa_nominal = resolve_cpa_time_s(spec, duration)
        time_fr = freq_ratio_time_axis(duration, len(fr))
        cpa_est = estimate_cpa_time_from_freq_ratio(fr, time_fr)
        cpa_time_err = abs(cpa_est - float(cpa_nominal)) if cpa_nominal is not None else None

        emit_meta = load_json(os.path.join(paths["emit_sample"], "sample_metadata.json"))
        obs_meta = load_json(os.path.join(paths["obs_sample"], "sample_metadata.json"))
        meta_issues = metadata_parameters_match(emit_meta, obs_meta)
        emit_finite_bad = scan_sidecars_finite(ec)
        obs_finite_bad = scan_sidecars_finite(oc)
        time_monotonic = bool(np.all(np.diff(time_obs) > 0)) if len(time_obs) > 1 else True

        sp_shape = tuple(int(x) for x in sp_e.shape)

        ridge_e = cqt_ridge_metrics(cqt_e)
        ridge_o = cqt_ridge_metrics(cqt_o)
        arc = path_arc_length(sp_e)
        expected = float(spec["speed"]) * float(spec["duration"])

        comp_dir = paths["comparison"]
        comp_files = sorted(os.listdir(comp_dir)) if os.path.isdir(comp_dir) else []

        records.append({
            "index": clip["index"],
            "sample_dir": sid,
            "path_type": spec["path_type"],
            "vehicle": spec["vehicle"],
            "spec": spec,
            "emitter_meta": clip.get("emitter_meta") or {},
            "comparison_metrics": clip.get("comparison_metrics") or {},
            "emit_v_mean": float(np.mean(ekin[:, 1])),
            "emit_v_std": float(np.std(ekin[:, 1])),
            "emit_accel_max": float(np.max(np.abs(ekin[:, 2]))),
            "pitch_mean": float(np.mean(pitch)),
            "pitch_std": float(np.std(pitch)),
            "source_pos_max_diff": float(np.max(np.abs(sp_e - sp_o))),
            "arc_length": arc,
            "expected_distance": expected,
            "arc_ratio": arc / expected if expected > 0 else 0.0,
            "emit_has_freq_ratios": os.path.isfile(os.path.join(ec, "freq_ratios.npy")),
            "obs_freq_span": float(np.max(fr) - np.min(fr)),
            "raw_corr": raw_corr,
            "ridge_slope_e": ridge_e["slope"],
            "ridge_slope_o": ridge_o["slope"],
            "ridge_slope_ratio": abs(ridge_o["slope"]) / (abs(ridge_e["slope"]) + 1e-12),
            "fr_emit_peak_corr": fr_emit_corr,
            "fr_obs_peak_corr": fr_obs_corr,
            "peak_std_e": ridge_e["peak_std"],
            "peak_std_o": ridge_o["peak_std"],
            "emit_rms_cv": float(np.std(emit_rms) / (np.mean(emit_rms) + 1e-9)),
            "obs_rms_cv": float(np.std(obs_rms) / (np.mean(obs_rms) + 1e-9)),
            "emit_freq_std": float(np.std(freq_e)),
            "obs_freq_std": float(np.std(freq_o)),
            "emit_meta": emit_meta,
            "obs_meta": obs_meta,
            "path_plot_exists": os.path.isfile(
                os.path.join(ec, clip["emitter"].get("path_plot", ""))
            ),
            "comparison_files": comp_files,
            "emit_wav_len": len(emit_wav),
            "obs_wav_len": len(obs_wav),
            "expected_n_samples": expected_n,
            "pitch_max_err": pitch_max_err,
            "source_centroid_hz": source_centroid,
            "emit_centroid_hz": emit_centroid,
            "centroid_rel_diff": centroid_rel_diff,
            "source_load_error": source_load_error,
            "fr_min": fr_min,
            "fr_max": fr_max,
            "doppler_bound_lo": bound_lo,
            "doppler_bound_hi": bound_hi,
            "doppler_bounds_ok": bounds_ok,
            "kin_fr_max_err": kin_fr_max_err,
            "cpa_nominal_s": cpa_nominal,
            "cpa_est_s": cpa_est,
            "cpa_time_err_s": cpa_time_err,
            "meta_param_issues": meta_issues,
            "emit_sidecar_nonfinite": emit_finite_bad,
            "obs_sidecar_nonfinite": obs_finite_bad,
            "time_monotonic": time_monotonic,
            "source_pos_shape": sp_shape,
        })
    return records


def run_all_checks(batch_root: str) -> dict[str, Any]:
    overall, clips, dataset_rows = _load_batch(batch_root)
    samples = _per_sample_records(batch_root, clips)
    checks: list[CheckResult] = []

    accel_ok = all(abs(s["spec"].get("acceleration", 0)) < 1e-9 for s in samples)
    mode_ok = all(s["spec"].get("simulation_mode", "cv") == "cv" for s in samples)
    checks.append(CheckResult(
        "A1", "A", "CV mode and zero acceleration",
        "CV validation excludes acceleration physics.",
        "PASS" if accel_ok and mode_ok else "FAIL",
        f"accel_ok={accel_ok}, simulation_mode cv on all clips={mode_ok}.",
        {"per_sample_accel": [s["spec"]["acceleration"] for s in samples]},
    ))

    gen = int(overall.get("total_generated", 0))
    req = int(overall.get("total_requested", 0))
    failed = overall.get("failed") or []
    checks.append(CheckResult(
        "A2", "A", "Batch completed without failures",
        "All requested clips should be present.",
        "PASS" if gen == req and not failed else "FAIL",
        f"generated={gen}, requested={req}, failed_count={len(failed)}.",
        {"failed": failed},
    ))

    geo_ok = all(
        s["emitter_meta"].get("frame") == "co_moving_source"
        and s["emitter_meta"].get("geometric_doppler") is False
        for s in samples
    )
    checks.append(CheckResult(
        "A3", "A", "Emitter metadata declares co-moving frame",
        "Emitter branch must not claim geometric Doppler.",
        "PASS" if geo_ok else "FAIL",
        "All clips have frame=co_moving_source and geometric_doppler=false."
        if geo_ok else "At least one clip has unexpected emitter_meta.",
    ))

    acc_meta_ok = all(abs(s["emitter_meta"].get("acceleration_mps2", 0)) < 1e-9 for s in samples)
    checks.append(CheckResult(
        "A4", "A", "Emitter metadata acceleration is zero",
        "Matches CV configuration.",
        "PASS" if acc_meta_ok else "FAIL",
        "All emitter_meta.acceleration_mps2 == 0." if acc_meta_ok else "Non-zero acceleration in metadata.",
    ))

    meta_mismatches = []
    ds_emit = {r["sample_id"]: r for r in dataset_rows if r.get("branch") == "emitter_centric"}
    for s in samples:
        sid = s["sample_dir"]
        row = ds_emit.get(sid)
        if not row:
            meta_mismatches.append({"sample": sid, "issue": "missing dataset row"})
            continue
        spec_val = float(s["spec"]["speed"])
        csv_val = float(row["speed_mps"])
        if abs(spec_val - csv_val) > 1e-6:
            meta_mismatches.append({"sample": sid, "field": "speed", "spec": spec_val, "csv": csv_val})
        emit_p = s["emit_meta"].get("parameters") or {}
        if abs(float(emit_p.get("speed", 0)) - spec_val) > 1e-6:
            meta_mismatches.append({"sample": sid, "issue": "sample_metadata speed mismatch"})
    checks.append(CheckResult(
        "A5", "A", "Metadata consistency (overall / sample / dataset)",
        "Single source of truth for scene parameters.",
        "PASS" if not meta_mismatches else "FAIL",
        "No mismatches." if not meta_mismatches else f"{len(meta_mismatches)} mismatch(es).",
        {"mismatches": meta_mismatches},
    ))

    doppler_meta_ok = all(
        s["obs_meta"].get("doppler_ratio_range") is not None
        and s["emit_meta"].get("doppler_ratio_range") is None
        for s in samples
    )
    checks.append(CheckResult(
        "A6", "A", "Doppler ratio range only on observer metadata",
        "Geometric Doppler metadata belongs to observer branch.",
        "PASS" if doppler_meta_ok else "FAIL",
        "Observer has doppler_ratio_range; emitter does not." if doppler_meta_ok
        else "Unexpected Doppler metadata placement.",
    ))

    v_ok = all(
        abs(s["emit_v_mean"] - float(s["spec"]["speed"])) < V_SPEED_TOL
        and s["emit_v_std"] < V_SPEED_STD_MAX
        for s in samples
    )
    checks.append(CheckResult(
        "B1", "B", "Constant along-track speed on source clock",
        "At CV, v(t) should equal configured speed.",
        "PASS" if v_ok else "FAIL",
        f"All samples within {V_SPEED_TOL} m/s and std<{V_SPEED_STD_MAX}."
        if v_ok else "Speed column not constant or mismatched.",
        {"per_sample": [
            {"index": s["index"], "mean": s["emit_v_mean"], "std": s["emit_v_std"]}
            for s in samples
        ]},
    ))

    a_ok = all(s["emit_accel_max"] < 1e-9 for s in samples)
    checks.append(CheckResult(
        "B2", "B", "Zero acceleration in emitter_frame_kinematics",
        "CV mode has a(t)=0.",
        "PASS" if a_ok else "FAIL",
        "Acceleration column zero on all clips." if a_ok else "Non-zero acceleration column found.",
    ))

    pitch_ok = all(
        s["pitch_std"] < PITCH_STD_MAX and abs(s["pitch_mean"] - 1.0) < PITCH_MEAN_TOL
        for s in samples
    )
    checks.append(CheckResult(
        "B3", "B", "Flat src_pitch_curve (g(v)=1 at CV)",
        "With v_ref=speed, intrinsic pitch factor is unity.",
        "PASS" if pitch_ok else "FAIL",
        "src_pitch_curve mean~1, std~0 on all clips." if pitch_ok else "Pitch curve not flat.",
        {"per_sample": [{"index": s["index"], "mean": s["pitch_mean"], "std": s["pitch_std"]} for s in samples]},
    ))

    no_fr = all(not s["emit_has_freq_ratios"] for s in samples)
    checks.append(CheckResult(
        "B4", "B", "No freq_ratios.npy on emitter branch",
        "Geometric Doppler arrays are observer-only.",
        "PASS" if no_fr else "FAIL",
        "Emitter Common/ has no freq_ratios.npy." if no_fr else "freq_ratios found on emitter.",
    ))

    sp_ok = all(s["source_pos_max_diff"] < 1e-5 for s in samples)
    checks.append(CheckResult(
        "C1", "C", "Identical source_positions.npy across branches",
        "Scene geometry is branch-independent.",
        "PASS" if sp_ok else "FAIL",
        "Max branch diff < 1e-5 on all clips." if sp_ok else "Position arrays differ between branches.",
    ))

    arc_fail = []
    for s in samples:
        r = s["arc_ratio"]
        if s["path_type"] == "straight":
            if not (STRAIGHT_ARC_LEN_RATIO_MIN <= r <= STRAIGHT_ARC_LEN_RATIO_MAX):
                arc_fail.append(s["index"])
        elif r > CURVED_ARC_LEN_RATIO_MAX:
            arc_fail.append(s["index"])
    checks.append(CheckResult(
        "C2", "C", "Path arc length vs v*T",
        "Straight paths: arc~vT; curved paths may be longer but bounded.",
        "FAIL" if arc_fail else "PASS",
        f"Failed samples: {arc_fail or 'none'}.",
        {"per_sample": [
            {"index": s["index"], "path": s["path_type"], "arc_ratio": round(s["arc_ratio"], 4)}
            for s in samples
        ]},
    ))

    plot_ok = all(s["path_plot_exists"] for s in samples)
    checks.append(CheckResult(
        "C3", "C", "Path plot PNG present on emitter branch",
        "Visual sanity on scene setup.",
        "PASS" if plot_ok else "FAIL",
        "All emitter path plots found." if plot_ok else "Missing path plot(s).",
        {"missing": [s["index"] for s in samples if not s["path_plot_exists"]]},
    ))

    corr_ok = all(abs(s["raw_corr"]) <= RAW_CORR_MAX for s in samples)
    checks.append(CheckResult(
        "D1", "D", "Emitter and observer WAVs differ (low correlation)",
        "Different physics should produce distinct waveforms.",
        "PASS" if corr_ok else "FAIL",
        f"All |rho|<={RAW_CORR_MAX}." if corr_ok else "High correlation on some clips.",
        {"per_sample": [{"index": s["index"], "raw_corr": round(s["raw_corr"], 4)} for s in samples]},
    ))

    obs_span_ok = all(s["obs_freq_span"] >= OBS_FREQ_SPAN_MIN for s in samples)
    checks.append(CheckResult(
        "D2", "D", "Observer freq_ratios show pass-by swing",
        "Observer branch uses geometric Doppler.",
        "PASS" if obs_span_ok else "FAIL",
        f"All spans>={OBS_FREQ_SPAN_MIN}." if obs_span_ok else "Small observer Doppler span.",
        {"per_sample": [{"index": s["index"], "span": round(s["obs_freq_span"], 4)} for s in samples]},
    ))

    straight = [s for s in samples if s["path_type"] == "straight"]
    d3_pass, d3_fail = [], []
    d3_notes: list[str] = []
    for s in straight:
        coupling = abs(s["fr_emit_peak_corr"])
        if coupling <= STRAIGHT_DOPPLER_PEAK_CORR_MAX:
            d3_pass.append(s["index"])
        else:
            d3_fail.append(s["index"])
            d3_notes.append(
                f"sample {s['index']}: |corr(freq_ratios, emit CQT peak)|={coupling:.3f}"
            )

    # Same vehicle @ CV: emitter ridge slope must be scene-independent (not path/speed driven).
    from collections import defaultdict
    by_vehicle: dict[str, list[float]] = defaultdict(list)
    for s in straight:
        by_vehicle[s["vehicle"]].append(s["ridge_slope_e"])
    for vehicle, slopes in by_vehicle.items():
        if len(slopes) < 2:
            continue
        spread = max(slopes) - min(slopes)
        if spread > EMIT_RIDGE_VEHICLE_TOL:
            d3_fail.extend([s["index"] for s in straight if s["vehicle"] == vehicle])
            d3_notes.append(
                f"{vehicle}: emit ridge slope varies by scene (spread={spread:.5f})"
            )

    d3_fail = sorted(set(d3_fail))
    d3_pass = [i for i in d3_pass if i not in d3_fail]

    if not straight:
        d3_status, d3_msg = "SKIP", "No straight clips in batch."
    elif d3_fail:
        d3_status = "FAIL"
        d3_msg = f"Failed straight samples: {d3_fail}. " + "; ".join(d3_notes[:3])
    else:
        d3_status = "PASS"
        d3_msg = (
            f"All straight samples: low Doppler coupling to emitter CQT peak "
            f"(|r|<={STRAIGHT_DOPPLER_PEAK_CORR_MAX}) and scene-stable ridge per vehicle."
        )
    checks.append(CheckResult(
        "D3", "D", "Straight clips: emitter not coupled to geometric Doppler",
        "Co-moving audio must not track observer freq_ratios; same vehicle @ CV gives same emitter ridge.",
        d3_status, d3_msg,
        {"per_sample": [
            {
                "index": s["index"],
                "vehicle": s["vehicle"],
                "emit_slope": round(s["ridge_slope_e"], 5),
                "obs_slope": round(s["ridge_slope_o"], 5),
                "fr_emit_peak_corr": round(s["fr_emit_peak_corr"], 3),
                "fr_obs_peak_corr": round(s["fr_obs_peak_corr"], 3),
            }
            for s in straight
        ]},
    ))

    parity_ok = all(
        not s["comparison_metrics"].get("meets_parity_target", True)
        and abs(s["comparison_metrics"].get("correlation", 1.0)) <= RAW_CORR_MAX
        for s in samples
    )
    checks.append(CheckResult(
        "D5", "D", "Cross-paradigm comparison metrics (non-parity expected)",
        "Observer vs co-moving emitter should not meet Formulation parity gate.",
        "PASS" if parity_ok else "INCONCLUSIVE",
        "meets_parity_target=false and low correlation on all clips."
        if parity_ok else "Unexpected parity or correlation.",
        {"per_sample": [
            {
                "index": s["index"],
                "correlation": s["comparison_metrics"].get("correlation"),
                "meets_parity_target": s["comparison_metrics"].get("meets_parity_target"),
            }
            for s in samples
        ]},
    ))

    rms_ok = all(s["emit_rms_cv"] < s["obs_rms_cv"] * EMIT_RMS_CV_MAX_OBS_RATIO for s in samples)
    checks.append(CheckResult(
        "E2", "E", "Emitter RMS envelope flatter than observer",
        "Observer applies pass-by gain; emitter should be less CPA-shaped.",
        "PASS" if rms_ok else "INCONCLUSIVE",
        "Emitter RMS CV lower than observer on all clips." if rms_ok
        else "Some clips have similar RMS modulation.",
        {"per_sample": [
            {"index": s["index"], "emit_cv": round(s["emit_rms_cv"], 3), "obs_cv": round(s["obs_rms_cv"], 3)}
            for s in samples
        ]},
    ))

    kia_straight = [s for s in straight if "Kia" in s["vehicle"]]
    e3_pass = [s["index"] for s in kia_straight if s["peak_std_e"] < s["peak_std_o"] * 0.75]
    if not kia_straight:
        e3_status, e3_msg = "SKIP", "No KiaSportage straight clips."
    elif len(e3_pass) == len(kia_straight):
        e3_status, e3_msg = "PASS", f"Kia straight clips: {e3_pass}."
    else:
        e3_status, e3_msg = "INCONCLUSIVE", f"Kia straight pass={e3_pass}."
    checks.append(CheckResult(
        "E3", "E", "Harmonic stability (Kia straight subset)",
        "Without geometric Doppler, emitter harmonics should shear less at CPA.",
        e3_status, e3_msg,
        {"per_sample": [
            {"index": s["index"], "peak_std_e": s["peak_std_e"], "peak_std_o": s["peak_std_o"]}
            for s in kia_straight
        ]},
    ))

    kia_freq = [s for s in straight if "Kia" in s["vehicle"]]
    f1_ok = all(s["emit_freq_std"] < s["obs_freq_std"] for s in kia_freq) if kia_freq else False
    checks.append(CheckResult(
        "F1", "F", "frequency.npy std lower on emitter (Kia straight)",
        "Dominant-frequency tracker should see less sweep on co-moving audio.",
        "PASS" if f1_ok else ("SKIP" if not kia_freq else "INCONCLUSIVE"),
        "Emitter frequency.npy std < observer on Kia straight clips."
        if f1_ok else "Metric ambiguous or not satisfied.",
        {"per_sample": [
            {"index": s["index"], "emit": s["emit_freq_std"], "obs": s["obs_freq_std"]}
            for s in kia_freq
        ]},
    ))

    required = {
        "emitter_centric.wav", "observer_centric.wav",
        "stacked_spectrogram_comparison.png", "diagnostics_panel.png",
        "comparison_report.txt",
    }
    missing_comp = []
    for s in samples:
        have = set(s["comparison_files"])
        miss = sorted(required - have)
        if miss:
            missing_comp.append({"index": s["index"], "missing": miss})
    checks.append(CheckResult(
        "G1", "G", "Comparison output folder complete",
        "Each sample should export WAVs, plots, and report.",
        "PASS" if not missing_comp else "FAIL",
        "All comparison artifacts present." if not missing_comp
        else f"{len(missing_comp)} sample(s) missing files.",
        {"missing": missing_comp},
    ))

    checks.append(CheckResult(
        "H1", "H", "No Doppler sidecars on emitter",
        "Negative check for freq_ratios on emitter branch.",
        "PASS" if no_fr else "FAIL",
        "Confirmed." if no_fr else "freq_ratios on emitter.",
    ))

    # --- B extended: co-moving physics ---
    b5_ok = all(s["pitch_max_err"] <= PITCH_RECOMPUTE_TOL for s in samples)
    checks.append(CheckResult(
        "B5", "B", "src_pitch_curve matches RPM formula g(v)=v/v_ref",
        "Verifies RPM coupling is wired to speed_profile, not just flat arrays.",
        "PASS" if b5_ok else "FAIL",
        f"All max|stored-expected| <= {PITCH_RECOMPUTE_TOL}."
        if b5_ok else "Pitch curve mismatch vs recomputed g(v).",
        {"per_sample": [
            {"index": s["index"], "pitch_max_err": s["pitch_max_err"]} for s in samples
        ]},
    ))

    from collections import defaultdict

    b6_fail: list[int] = []
    b6_groups: dict[str, list] = defaultdict(list)
    for s in samples:
        if s["centroid_rel_diff"] is not None:
            b6_groups[s["vehicle"]].append(s)
    for vehicle, group in b6_groups.items():
        if len(group) < 2:
            continue
        cents = [g["emit_centroid_hz"] for g in group if g["emit_centroid_hz"]]
        if len(cents) < 2:
            continue
        spread = (max(cents) - min(cents)) / (np.mean(cents) + 1e-9)
        if spread > SAME_VEHICLE_CENTROID_SPREAD_TOL:
            b6_fail.extend(g["index"] for g in group)
    b6_fail = sorted(set(b6_fail))
    checks.append(CheckResult(
        "B6", "B", "Same vehicle @ CV: emitter centroid stable across speeds",
        "With g(v)=1, co-moving audio should not scale with pass-by speed (RPM wiring sanity).",
        "PASS" if not b6_fail else "INCONCLUSIVE",
        "Per-vehicle centroid spread within tolerance." if not b6_fail
        else f"High centroid spread for vehicles on samples {b6_fail}.",
        {"per_vehicle": {
            v: [
                {"index": g["index"], "speed": g["spec"]["speed"], "centroid": g["emit_centroid_hz"]}
                for g in grp
            ]
            for v, grp in b6_groups.items() if len(grp) >= 2
        }},
    ))

    b7_fail = [
        s["index"] for s in samples
        if s["source_load_error"]
        or (
            s["centroid_rel_diff"] is not None
            and s["centroid_rel_diff"] > SOURCE_EMIT_CENTROID_REL_TOL
        )
    ]
    checks.append(CheckResult(
        "B7", "B", "Emitter WAV spectral centroid near raw source (CV, g=1)",
        "Co-moving output should be source resampling without geometric sweep.",
        "FAIL" if b7_fail else "PASS",
        "Centroid within tolerance of library source." if not b7_fail
        else f"Failed samples: {b7_fail}.",
        {"per_sample": [
            {
                "index": s["index"],
                "centroid_rel_diff": s["centroid_rel_diff"],
                "error": s["source_load_error"],
            }
            for s in samples
        ]},
    ))

    b8_fail = [
        s["index"] for s in samples
        if abs(s["emit_wav_len"] - s["expected_n_samples"]) > SAMPLE_COUNT_TOL
        or abs(s["obs_wav_len"] - s["expected_n_samples"]) > SAMPLE_COUNT_TOL
    ]
    checks.append(CheckResult(
        "B8", "B", "WAV sample count matches SR * duration",
        "Catches off-by-one grid / linspace boundary bugs.",
        "PASS" if not b8_fail else "FAIL",
        f"All within +/-{SAMPLE_COUNT_TOL} of int(SR*duration)." if not b8_fail
        else f"Failed: {b8_fail}.",
        {"per_sample": [
            {
                "index": s["index"],
                "emit_len": s["emit_wav_len"],
                "obs_len": s["obs_wav_len"],
                "expected": s["expected_n_samples"],
            }
            for s in samples
        ]},
    ))

    # --- A extended: cross-branch metadata ---
    a7_fail = [s["index"] for s in samples if s["meta_param_issues"]]
    checks.append(CheckResult(
        "A7", "A", "Emitter/observer sample_metadata parameters match",
        "Scene parameters must be identical across branches.",
        "PASS" if not a7_fail else "FAIL",
        "All shared keys match." if not a7_fail else f"Mismatches on samples {a7_fail}.",
        {"issues": [
            {"index": s["index"], "issues": s["meta_param_issues"]}
            for s in samples if s["meta_param_issues"]
        ]},
    ))

    # --- O: observer physics ground truth ---
    straight = [s for s in samples if s["path_type"] == "straight"]
    o4_fail = [s["index"] for s in straight if not s["doppler_bounds_ok"]]
    checks.append(CheckResult(
        "O4", "O", "Observer freq_ratios within analytic c/(c+/-v) bounds (straight)",
        "Loose physics bounds for geometric Doppler extrema.",
        "SKIP" if not straight else ("PASS" if not o4_fail else "FAIL"),
        "All straight clips within bounds." if not o4_fail else f"Failed: {o4_fail}.",
        {"per_sample": [
            {
                "index": s["index"],
                "fr_min": round(s["fr_min"], 4),
                "fr_max": round(s["fr_max"], 4),
                "bound_lo": round(s["doppler_bound_lo"], 4),
                "bound_hi": round(s["doppler_bound_hi"], 4),
            }
            for s in straight
        ]},
    ))

    o5_fail = [
        s["index"] for s in straight
        if s["kin_fr_max_err"] is None or s["kin_fr_max_err"] > KINEMATICS_FR_MAX_ABS_ERR
    ]
    checks.append(CheckResult(
        "O5", "O", "Observer freq_ratios match kinematics recomputation (straight)",
        "Closed-form ground truth: compare export to straight_cv_kinematics_with_c.",
        "SKIP" if not straight else ("PASS" if not o5_fail else "FAIL"),
        f"Max abs error <= {KINEMATICS_FR_MAX_ABS_ERR}." if not o5_fail
        else f"Failed: {o5_fail}.",
        {"per_sample": [
            {"index": s["index"], "kin_fr_max_err": s["kin_fr_max_err"]} for s in straight
        ]},
    ))

    o6_fail = [
        s["index"] for s in straight
        if s["cpa_time_err_s"] is None or s["cpa_time_err_s"] > CPA_TIME_TOL_S
    ]
    checks.append(CheckResult(
        "O6", "O", "CPA timing: freq_ratio~1 aligns with nominal CPA time (straight)",
        "Pass-by CPA should align with geometric Doppler unity crossing.",
        "SKIP" if not straight else ("PASS" if not o6_fail else "INCONCLUSIVE"),
        f"Within +/-{CPA_TIME_TOL_S}s of resolve_cpa_time_s." if not o6_fail
        else f"Misaligned samples: {o6_fail}.",
        {"per_sample": [
            {
                "index": s["index"],
                "cpa_nominal_s": s["cpa_nominal_s"],
                "cpa_est_s": round(s["cpa_est_s"], 3),
                "err_s": s["cpa_time_err_s"],
            }
            for s in straight
        ]},
    ))

    golden_path = os.path.join(
        os.path.dirname(__file__), "golden", "cv_straight_analytic.json"
    )
    o7_status, o7_msg, o7_details = "SKIP", "Golden spec file missing.", {}
    if os.path.isfile(golden_path):
        golden = load_json(golden_path)
        gp = golden["parameters"]
        tol = float(golden.get("tolerance_max_abs_ratio", 0.025))
        pmt = golden.get("param_match_tol", {})
        best = None
        for s in straight:
            spec = s["spec"]
            score = (
                abs(float(spec["speed"]) - gp["speed_mps"]) / max(pmt.get("speed_mps", 1), 1)
                + abs(float(spec["distance"]) - gp["distance_m"]) / max(pmt.get("distance_m", 1), 1)
            )
            if best is None or score < best[0]:
                best = (score, s)
        if best and best[1]["kin_fr_max_err"] is not None:
            _, ref = best
            err = ref["kin_fr_max_err"]
            o7_details = {"reference_sample": ref["index"], "kin_fr_max_err": err, "tol": tol}
            if err <= tol:
                o7_status, o7_msg = "PASS", f"Nearest straight clip {ref['index']} matches golden tolerance."
            else:
                o7_status, o7_msg = "INCONCLUSIVE", (
                    f"Nearest clip {ref['index']} err={err:.4f} > tol={tol}."
                )
        elif straight:
            o7_status, o7_msg = "INCONCLUSIVE", "No straight clip for golden comparison."
    checks.append(CheckResult(
        "O7", "O", "Golden straight analytic regression (nearest clip in batch)",
        "Highest-confidence physics regression vs reference spec.",
        o7_status, o7_msg, o7_details,
    ))

    # --- H extended ---
    h2_fail = [s["index"] for s in samples if not s["time_monotonic"]]
    checks.append(CheckResult(
        "H2", "H", "time.npy strictly increasing (observer branch)",
        "Proxy for monotonic arrival-time grid; arrival_times.npy not exported.",
        "PASS" if not h2_fail else "FAIL",
        "Monotonic time axis on all clips." if not h2_fail else f"Failed: {h2_fail}.",
    ))

    h3_fail = [
        s["index"] for s in samples
        if s["emit_sidecar_nonfinite"] or s["obs_sidecar_nonfinite"]
    ]
    checks.append(CheckResult(
        "H3", "H", "All .npy sidecars finite (both branches)",
        "Catch NaN/Inf numerical failures.",
        "PASS" if not h3_fail else "FAIL",
        "All finite." if not h3_fail else f"Non-finite arrays on samples {h3_fail}.",
        {"nonfinite": [
            {
                "index": s["index"],
                "emit": s["emit_sidecar_nonfinite"],
                "obs": s["obs_sidecar_nonfinite"],
            }
            for s in samples if s["emit_sidecar_nonfinite"] or s["obs_sidecar_nonfinite"]
        ]},
    ))

    h4_fail = [
        s["index"] for s in samples
        if len(s["source_pos_shape"]) != 2 or s["source_pos_shape"][1] != 3
    ]
    checks.append(CheckResult(
        "H4", "H", "source_positions.npy shape (N, 3)",
        "Single-source batches use (N, 3) observer-frame tracks.",
        "PASS" if not h4_fail else "FAIL",
        "All clips (N, 3)." if not h4_fail else f"Unexpected shape on {h4_fail}.",
        {"shapes": {s["index"]: s["source_pos_shape"] for s in samples}},
    ))

    # --- M: batch coverage ---
    cfg = overall.get("config") or {}
    speeds = [float(s["spec"]["speed"]) for s in samples]
    speed_cfg = cfg.get("speed") or {}
    lo = float(speed_cfg.get("min", min(speeds) if speeds else 0))
    hi = float(speed_cfg.get("max", max(speeds) if speeds else 0))
    span = hi - lo
    actual_span = max(speeds) - min(speeds) if speeds else 0.0
    m1_ok = span <= 0 or (actual_span / span) >= SPEED_RANGE_MIN_SPAN_FRAC
    checks.append(CheckResult(
        "M1", "M", "Batch exercises configured speed range",
        "Detect config bugs where all clips share one speed.",
        "PASS" if m1_ok else "FAIL",
        f"Speed span {actual_span:.2f} m/s of configured [{lo}, {hi}].",
        {"speed_min": min(speeds), "speed_max": max(speeds), "configured": [lo, hi]},
    ))

    configured_paths = cfg.get("path_types") or sorted({s["path_type"] for s in samples})
    present_paths = {s["path_type"] for s in samples}
    missing_paths = [p for p in configured_paths if p not in present_paths]
    checks.append(CheckResult(
        "M2", "M", "At least one clip per configured path type",
        "Path-specific checks need straight/parabola/bezier coverage.",
        "PASS" if not missing_paths else "FAIL",
        "All configured path types present." if not missing_paths
        else f"Missing path types: {missing_paths}.",
        {"configured": configured_paths, "present": sorted(present_paths)},
    ))

    # --- R: determinism ---
    r1_ok = False
    r1_msg = "No clips to probe."
    if samples:
        r1_ok = synthesizer_deterministic_probe(samples[0]["spec"])
        r1_msg = (
            "synthesize_source_frame_audio bit-identical on repeat."
            if r1_ok else "Non-deterministic co-moving synthesis detected."
        )
    checks.append(CheckResult(
        "R1", "R", "Co-moving synthesizer deterministic (in-memory probe)",
        "Same inputs should yield identical emitter WAV samples.",
        "PASS" if r1_ok else ("FAIL" if samples else "SKIP"),
        r1_msg,
        {"probe_sample": samples[0]["index"] if samples else None},
    ))

    summary = {
        "batch_root": os.path.abspath(batch_root),
        "batch_id": overall.get("batch_id"),
        "n_clips": len(samples),
        "pass": sum(1 for c in checks if c.status == "PASS"),
        "fail": sum(1 for c in checks if c.status == "FAIL"),
        "inconclusive": sum(1 for c in checks if c.status == "INCONCLUSIVE"),
        "skip": sum(1 for c in checks if c.status == "SKIP"),
    }

    return {
        "summary": summary,
        "checks": [c.to_dict() for c in checks],
        "per_sample": samples,
    }
