"""Fill path-specific geometry keys for emitter-centric clips (isolated)."""

from __future__ import annotations

import random
from typing import Any

from core.config import DEFAULT_RANGES


def enrich_path_params(
    path_type: str,
    params: dict[str, Any],
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Ensure params contain keys required by ``source_position``, path plots, and
    ``get_doppler_audio_array`` for parabola / bezier (mirrors batch sampling).
    """
    p = dict(params)
    path_type = str(path_type or "straight").lower()
    p["path_type"] = path_type
    p.setdefault("h", float(p.get("distance", p.get("h", 30.0))))
    p.setdefault("duration", float(p.get("duration", 10.0)))

    rng = random.Random(seed)

    if path_type == "parabola":
        if "a" not in p:
            a_lo, a_hi = DEFAULT_RANGES.get("parabola_a", (5, 20))
            p["a"] = rng.randint(int(a_lo), int(a_hi)) / 10000.0
        p["h"] = float(p.get("h", p.get("distance", 30.0)))

    elif path_type == "bezier":
        needed = ("x0", "x1", "x2", "x3", "y0", "y1", "y2", "y3")
        if not all(k in p for k in needed):
            span = float(p["speed"]) * float(p["duration"])
            half = 0.5 * span
            if rng.random() > 0.5:
                x0, x3 = -half, half
            else:
                x0, x3 = half, -half
            lo_x, hi_x = min(x0, x3), max(x0, x3)
            dist = float(p.get("distance", 30.0))
            y_off = rng.uniform(10.0, 50.0)
            p.setdefault("x0", x0)
            p.setdefault("x3", x3)
            p.setdefault("x1", rng.uniform(lo_x, hi_x))
            p.setdefault("x2", rng.uniform(lo_x, hi_x))
            p.setdefault("y0", dist + y_off)
            p.setdefault("y3", dist + y_off)
            p.setdefault("y1", dist + rng.uniform(-2.0, 5.0))
            p.setdefault("y2", dist + rng.uniform(-2.0, 5.0))

    return p
