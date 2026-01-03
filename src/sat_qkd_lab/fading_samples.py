"""
Fading sample utilities for transmittance variance inspection.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt


def sample_fading_transmittance(
    sigma_ln: float,
    n_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Sample lognormal fading transmittance factors with unit mean.
    """
    if n_samples <= 0:
        return np.array([], dtype=float)
    if sigma_ln <= 0:
        return np.ones(n_samples, dtype=float)
    rng = np.random.default_rng(seed)
    mu = -0.5 * sigma_ln ** 2
    return rng.lognormal(mean=mu, sigma=sigma_ln, size=n_samples)


def plot_eta_samples(
    samples: Sequence[float],
    out_path: str,
) -> str:
    """Plot a small histogram of fading transmittance samples."""
    fig, ax = plt.subplots()
    ax.hist(samples, bins=24, color="steelblue", alpha=0.8)
    ax.set_xlabel("Transmittance sample")
    ax.set_ylabel("Count")
    ax.set_title("Fading transmittance samples")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path
