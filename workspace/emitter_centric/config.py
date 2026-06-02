"""Emitter-centric workspace configuration (isolated from production)."""

from __future__ import annotations

import os

# All generated artifacts stay under this tree only.
OUTPUT_ROOT = os.path.join("static", "workspace_outputs", "emitter_centric")
DEFAULT_SAVE_PATH = OUTPUT_ROOT

# Match production observer export rate.
SR = 22050

# Radial Mach guard (per emission sample); production-style clamp.
MR_MAX = 0.9

# Worst-case propagation padding (§2.10 plan) — used when use_propagation_delay=True.
R_MAX_M = 1000.0
C_MIN_MPS = 319.0

DEFAULT_DURATION_S = 10.0
DEFAULT_SPEED_MPS = 30.0
DEFAULT_DISTANCE_M = 15.0
DEFAULT_ANGLE_DEG = 0.0
DEFAULT_TEMP_C = 20.0
DEFAULT_HUMIDITY = 50.0
