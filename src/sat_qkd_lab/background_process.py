"""
Time-correlated background process utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BackgroundProcessParams:
    enabled: bool = False
    mean: float = 1.0
    sigma: float = 0.2
    tau_seconds: float = 60.0
    seed: int = 0


def simulate_background_scales(
    time_s: Sequence[float],
    params: BackgroundProcessParams,
) -> np.ndarray:
    """
    Simulate a time-correlated background scale factor via OU dynamics.
    """
    times = np.array(time_s, dtype=float)
    if times.size == 0:
        return np.array([], dtype=float)
    if not params.enabled:
        return np.ones(times.size, dtype=float)

    mean = float(params.mean)
    sigma = float(params.sigma)
    tau = float(params.tau_seconds)
    rng = np.random.default_rng(params.seed)

    if sigma <= 0.0:
        return np.full(times.size, max(0.0, mean), dtype=float)

    values = np.zeros_like(times, dtype=float)
    values[0] = mean
    if tau <= 0.0:
        noise = rng.normal(0.0, sigma, size=times.size)
        values = mean + noise
    else:
        theta = 1.0 / tau
        for idx in range(1, times.size):
            dt = float(times[idx] - times[idx - 1])
            if dt < 0.0:
                dt = 0.0
            drift = theta * (mean - values[idx - 1]) * dt
            diffusion = sigma * np.sqrt(dt) * rng.normal(0.0, 1.0)
            values[idx] = values[idx - 1] + drift + diffusion

    return np.maximum(0.0, values)
