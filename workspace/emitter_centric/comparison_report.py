"""Text comparison report for observer vs emitter clips."""

from __future__ import annotations

import os
from typing import Any


def write_comparison_report(
    out_path: str,
    *,
    params: dict,
    vehicle: str,
    comparison_metrics: dict,
    notes: list[str] | None = None,
) -> str:
    lines = [
        "Emitter-Centric Analysis — Comparison Report",
        "=" * 60,
        "",
        f"Vehicle: {vehicle}",
        f"Path type: {params.get('path_type', 'straight')}",
        f"Speed: {params.get('speed', 0):.4f} m/s",
        f"Acceleration: {params.get('acceleration', 0):.4f} m/s²",
        f"Scene lateral offset: {params.get('distance', 0):.4f} m",
        f"Angle (straight): {params.get('angle', 0):.2f}°",
        f"Duration: {params.get('duration', 10):.2f} s",
        f"Temperature: {params.get('temperature', 20):.1f} °C",
        f"Humidity: {params.get('humidity', 50):.0f} %",
        "",
        "Frame interpretation",
        "-" * 40,
        "emitter_centric/: co-moving source microphone — no geometric Doppler.",
        "observer_centric/: fixed roadside observer — production Doppler + gain.",
        "",
        "Waveform metrics (level-matched)",
        "-" * 40,
    ]
    for k, v in comparison_metrics.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Observations")
    lines.append("-" * 40)
    for n in notes or []:
        lines.append(f"  • {n}")
    lines.extend([
        "",
        "Implementation notes",
        "-" * 40,
        "  • See workspace/emitter_centric/plan.md#first-principles for paradigm definitions.",
        "  • Formulation-1 (emission grid → observer) is documented in workspace/emitter_centric/plan.md.",
    ])
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path
