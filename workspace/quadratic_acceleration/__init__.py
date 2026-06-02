"""
Quadratic Acceleration workspace sub-mode.

Observer-centric pass-by synthesis with optional RPM / acceleration-aware
source pitch coupling before geometric Doppler. Isolated from production
batch routes and from ``workspace.emitter_centric``.
"""

from workspace.quadratic_acceleration.abs_synthesis import (
    dedopplerize_far_field_segment,
    find_peak_time,
    stitch_repeat_segment,
    synthesize_passby_straight,
)

__all__ = [
    "dedopplerize_far_field_segment",
    "find_peak_time",
    "stitch_repeat_segment",
    "synthesize_passby_straight",
]
