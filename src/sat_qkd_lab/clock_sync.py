"""
Clock beacon sync utilities for offset/drift estimation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class ClockSyncResult:
    offset_s: float
    drift_ppm: float
    residual_std_s: float
    residuals_s: np.ndarray


def generate_beacon_times(
    duration_s: float,
    rate_hz: float,
    jitter_sigma_s: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate deterministic beacon timestamps with optional jitter.
    """
    if duration_s <= 0 or rate_hz <= 0:
        return np.array([], dtype=float)
    period = 1.0 / rate_hz
    times = np.arange(0.0, duration_s, period, dtype=float)
    if jitter_sigma_s > 0:
        rng = np.random.default_rng(seed)
        times = times + rng.normal(0.0, jitter_sigma_s, size=times.size)
    return times


def apply_clock_model(
    times: np.ndarray,
    offset_s: float,
    drift_ppm: float,
) -> np.ndarray:
    """
    Apply clock offset and linear drift to time stamps.
    """
    return times * (1.0 + drift_ppm * 1e-6) + offset_s


def estimate_offset_drift(
    times_a: np.ndarray,
    times_b: np.ndarray,
) -> ClockSyncResult:
    """
    Estimate offset and drift using linear regression on beacon times.
    """
    if times_a.size == 0 or times_b.size == 0:
        return ClockSyncResult(0.0, 0.0, 0.0, np.array([], dtype=float))
    n = min(times_a.size, times_b.size)
    x = times_a[:n]
    y = times_b[:n]
    slope, intercept = np.polyfit(x, y, 1)
    drift_ppm = (slope - 1.0) * 1e6
    residuals = y - (slope * x + intercept)
    residual_std = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
    return ClockSyncResult(
        offset_s=float(intercept),
        drift_ppm=float(drift_ppm),
        residual_std_s=residual_std,
        residuals_s=residuals,
    )
