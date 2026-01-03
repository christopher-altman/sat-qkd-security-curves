"""
Motion-induced basis bias and polarization rotation utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class BasisBiasParams:
    max_bias: float = 0.15
    rotation_deg_at_zenith: float = 8.0


def basis_bias_from_elevation(
    elevation_deg: np.ndarray,
    params: BasisBiasParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute basis bias and polarization rotation as a function of elevation.
    """
    elev = np.clip(elevation_deg, 0.0, 90.0)
    norm = np.sin(np.deg2rad(elev))
    bias = params.max_bias * norm
    rotation_deg = params.rotation_deg_at_zenith * norm
    return bias, rotation_deg


def basis_probs_from_bias(bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert bias into P(Z) and P(X) probabilities.
    """
    bias = np.clip(bias, -0.9, 0.9)
    pz = 0.5 + 0.5 * bias
    px = 1.0 - pz
    return pz, px
