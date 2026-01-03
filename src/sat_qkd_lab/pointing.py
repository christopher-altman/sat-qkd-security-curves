"""
Pointing acquisition/tracking/dropout dynamics for pass modeling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class PointingParams:
    acq_seconds: float = 0.0
    dropout_prob: float = 0.0
    relock_seconds: float = 0.0
    pointing_jitter_urad: float = 2.0
    seed: int = 0


def simulate_pointing_profile(
    time_s: np.ndarray,
    params: PointingParams,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    Simulate lock state and transmittance multiplier from pointing error.
    """
    rng = np.random.default_rng(params.seed)
    if time_s.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=float), 0, 0.0

    dt = float(time_s[1] - time_s[0]) if time_s.size > 1 else 1.0
    lock_state = np.ones(time_s.size, dtype=bool)
    lock_state[time_s < params.acq_seconds] = False

    dropout_count = 0
    relock_steps = int(max(0.0, params.relock_seconds) / dt) if dt > 0 else 0
    locked = True if time_s[0] >= params.acq_seconds else False
    remaining_relock = 0

    sigma_rad = max(1e-12, params.pointing_jitter_urad * 1e-6)
    tau = 1.0
    alpha = np.exp(-dt / tau)
    sigma_step = sigma_rad * np.sqrt(1.0 - alpha ** 2)
    x = 0.0
    trans_mult = np.zeros(time_s.size, dtype=float)

    for i, t in enumerate(time_s):
        if t < params.acq_seconds:
            locked = False
        elif remaining_relock > 0:
            locked = False
            remaining_relock -= 1
        else:
            locked = True

        if locked and params.dropout_prob > 0.0:
            if rng.random() < params.dropout_prob * dt:
                dropout_count += 1
                remaining_relock = relock_steps
                locked = False

        lock_state[i] = locked

        if locked:
            x = alpha * x + sigma_step * rng.normal()
            trans_mult[i] = np.exp(-0.5 * (x / sigma_rad) ** 2)
        else:
            trans_mult[i] = 0.0

    lock_fraction = float(np.mean(lock_state)) if lock_state.size > 0 else 0.0
    return lock_state, trans_mult, dropout_count, lock_fraction
