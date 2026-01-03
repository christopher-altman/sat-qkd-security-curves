"""
Ornstein-Uhlenbeck fading evolution utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass(frozen=True)
class OutageStats:
    count: int
    mean_duration_s: float
    durations_s: List[float]


def simulate_ou_transmittance(
    duration_s: float,
    dt_s: float,
    mu: float,
    theta: float,
    sigma: float,
    seed: int = 0,
    t0: float = 0.0,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate OU process for transmittance evolution.
    """
    if duration_s <= 0 or dt_s <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    n_steps = int(np.floor(duration_s / dt_s)) + 1
    times = np.linspace(0.0, duration_s, n_steps, dtype=float)
    values = np.zeros(n_steps, dtype=float)
    values[0] = float(t0)

    rng = np.random.default_rng(seed)
    for i in range(1, n_steps):
        prev = values[i - 1]
        drift = theta * (mu - prev) * dt_s
        diffusion = sigma * np.sqrt(dt_s) * rng.normal()
        values[i] = prev + drift + diffusion

    if clamp_min is not None or clamp_max is not None:
        values = np.clip(values, clamp_min, clamp_max)

    return times, values


def compute_outage_clusters(
    times: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> OutageStats:
    """
    Compute outage clusters where values drop below threshold.
    """
    durations: List[float] = []
    in_outage = False
    start_time = 0.0

    for t, v in zip(times, values):
        below = v < threshold
        if below and not in_outage:
            in_outage = True
            start_time = float(t)
        elif not below and in_outage:
            in_outage = False
            durations.append(float(t) - start_time)

    if in_outage and times.size > 0:
        durations.append(float(times[-1]) - start_time)

    mean_duration = float(np.mean(durations)) if durations else 0.0
    return OutageStats(count=len(durations), mean_duration_s=mean_duration, durations_s=durations)
