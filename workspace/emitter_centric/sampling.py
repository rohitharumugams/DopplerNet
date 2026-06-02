"""Parameter sampling for emitter-centric batches (isolated)."""

from __future__ import annotations

import random
from typing import Any


def _rand_float(rng: random.Random, lo: float, hi: float) -> float:
    return float(rng.uniform(min(lo, hi), max(lo, hi)))


def build_clip_plan(config: dict) -> list[dict]:
    """Build ordered list of clip specs from UI/batch config."""
    total = int(config.get("total_clips", 1))
    vehicles = list(config.get("vehicles") or ["KiaSportage"])
    paths = list(config.get("path_types") or ["straight"])
    if not paths:
        paths = ["straight"]

    rng = random.Random(config.get("seed"))
    sim_mode = config.get("simulation_mode", "cv")
    accel_range = config.get("acceleration", {"min": 0.0, "max": 0.0})
    if sim_mode == "cv":
        accel_range = {"min": 0.0, "max": 0.0}

    dist_mode = config.get("distribution_mode", "auto")
    manual = config.get("manual_distribution") or {}

    slots: list[dict] = []
    if dist_mode == "manual" and manual.get("vehicles"):
        for veh, count in manual["vehicles"].items():
            for _ in range(int(count)):
                slots.append({"vehicle": veh})
        path_counts = {
            "straight": int(manual.get("straight", 0)),
            "parabola": int(manual.get("parabola", 0)),
            "bezier": int(manual.get("bezier", 0)),
        }
        path_list = []
        for p, c in path_counts.items():
            path_list.extend([p] * c)
        while len(path_list) < len(slots):
            path_list.append(rng.choice(paths))
        for i, slot in enumerate(slots):
            slot["path_type"] = path_list[i] if i < len(path_list) else rng.choice(paths)
    else:
        per_vehicle = max(1, total // max(len(vehicles), 1))
        vi = 0
        for i in range(total):
            if i > 0 and i % per_vehicle == 0:
                vi = (vi + 1) % len(vehicles)
            slots.append({
                "vehicle": vehicles[vi % len(vehicles)],
                "path_type": paths[i % len(paths)],
            })

    duration = float(config.get("duration_s", 10.0))
    speed_r = config.get("speed", {"min": 10, "max": 50})
    dist_r = config.get("distance", {"min": 5, "max": 100})
    angle_r = config.get("angle", {"min": -45, "max": 45})
    temp_r = config.get("temperature", {"min": 15, "max": 35})
    hum_r = config.get("humidity", {"min": 30, "max": 70})

    plan: list[dict] = []
    for idx, slot in enumerate(slots[:total], start=1):
        accel = 0.0 if sim_mode == "cv" else _rand_float(rng, accel_range["min"], accel_range["max"])
        plan.append({
            "index": idx,
            "vehicle": slot["vehicle"],
            "path_type": slot.get("path_type", "straight"),
            "duration": duration,
            "speed": _rand_float(rng, speed_r["min"], speed_r["max"]),
            "distance": _rand_float(rng, dist_r["min"], dist_r["max"]),
            "angle": _rand_float(rng, angle_r["min"], angle_r["max"]),
            "acceleration": accel,
            "temperature": _rand_float(rng, temp_r["min"], temp_r["max"]),
            "humidity": _rand_float(rng, hum_r["min"], hum_r["max"]),
            "simulation_mode": sim_mode,
        })
    return plan
