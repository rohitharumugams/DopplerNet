import json
import random
import numpy as np

from core.config import SAMPLER_STATE_FILE

# ============================================================
# GLOBAL SAMPLER CACHE (PER BATCH)
# ============================================================

SAMPLERS = {}


# ============================================================
# FAST INTEGER CYCLIC SAMPLER (O(1), FULL COVERAGE)
# ============================================================

class CyclicIntegerSampler:
    def __init__(self, low, high, seed=None):
        self.low = int(low)
        self.high = int(high)
        self.range = self.high - self.low + 1
        self.k = 0

        if self.range <= 1:
            self.step = 1
        else:
            # pick step coprime with range
            self.step = random.choice(
                [s for s in range(1, self.range) if np.gcd(s, self.range) == 1]
            )

        self.offset = random.randint(0, self.range - 1)

    def next(self):
        val = self.low + (self.offset + self.k * self.step) % self.range
        self.k += 1
        return int(val)


def save_sampler_state(filepath=None):
    state = {}
    for key, sampler in SAMPLERS.items():
        state[key] = {
            "low": sampler.low,
            "high": sampler.high,
            "step": sampler.step,
            "offset": sampler.offset,
            "k": sampler.k
        }
    target = filepath or SAMPLER_STATE_FILE
    with open(target, "w") as f:
        json.dump(state, f, indent=2)


def load_sampler_state(filepath=None):
    import os
    target = filepath or SAMPLER_STATE_FILE
    if not os.path.exists(target):
        return
    with open(target, "r") as f:
        state = json.load(f)

    for key, s in state.items():
        sampler = CyclicIntegerSampler(s["low"], s["high"])
        sampler.step = s["step"]
        sampler.offset = s["offset"]
        sampler.k = s["k"]
        SAMPLERS[key] = sampler
