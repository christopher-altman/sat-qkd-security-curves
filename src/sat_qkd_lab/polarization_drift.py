"""
Polarization drift and compensation utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PolarizationDriftParams:
    enabled: bool = False
    sigma_deg: float = 1.0
    seed: int = 0


@dataclass(frozen=True)
class CompensationParams:
    enabled: bool = False
    lag_seconds: float = 5.0


def simulate_polarization_drift(
    time_s: Sequence[float],
    params: PolarizationDriftParams,
) -> np.ndarray:
    """
    Simulate polarization angle drift as a random walk in radians.
    """
    times = np.array(time_s, dtype=float)
    if times.size == 0:
        return np.array([], dtype=float)
    if not params.enabled:
        return np.zeros(times.size, dtype=float)

    sigma_rad = np.deg2rad(float(params.sigma_deg))
    if sigma_rad <= 0.0:
        return np.zeros(times.size, dtype=float)

    rng = np.random.default_rng(params.seed)
    angles = np.zeros_like(times, dtype=float)
    for idx in range(1, times.size):
        dt = float(times[idx] - times[idx - 1])
        if dt < 0.0:
            dt = 0.0
        angles[idx] = angles[idx - 1] + sigma_rad * np.sqrt(dt) * rng.normal(0.0, 1.0)
    return angles


def compensate_polarization_drift(
    time_s: Sequence[float],
    angle_rad: np.ndarray,
    params: CompensationParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a lagged compensation estimator to the drift angle.
    """
    times = np.array(time_s, dtype=float)
    if times.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if not params.enabled:
        return angle_rad, np.zeros_like(angle_rad)

    lag = max(0.0, float(params.lag_seconds))
    est = np.zeros_like(angle_rad, dtype=float)
    for idx in range(1, times.size):
        dt = float(times[idx] - times[idx - 1])
        if dt < 0.0:
            dt = 0.0
        alpha = 1.0 if lag == 0.0 else dt / (lag + dt)
        est[idx] = est[idx - 1] + alpha * (angle_rad[idx] - est[idx - 1])
    residual = angle_rad - est
    return residual, est


def adjust_qber_for_angle(
    qber_mean: float,
    angle_rad: float,
    phase_offset_rad: float = 0.0,
) -> float:
    """
    Adjust QBER with polarization misalignment modeled as sin^2(angle).
    """
    drift_error = float(np.sin(angle_rad + phase_offset_rad) ** 2)
    adjusted = qber_mean + (1.0 - qber_mean) * drift_error
    return min(0.5, max(0.0, adjusted))


def coincidence_matrix_from_qber(
    n_sifted: int,
    qber: float,
) -> list[list[int]]:
    """
    Build a 2x2 coincidence matrix given a QBER.
    """
    n_basis = max(0, int(round(0.5 * n_sifted)))
    errors = int(round(n_basis * qber)) if n_basis > 0 else 0
    correct = max(0, n_basis - errors)
    return [
        [correct, errors],
        [errors, correct],
    ]
