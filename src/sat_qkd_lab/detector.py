"""
Detector and background noise model for QKD simulations.

This module provides a minimal but realistic detector model that captures
the operationally dominant effect at high channel loss: background/dark
clicks that contribute random bits and degrade QBER.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorParams:
    """
    Minimal detector model parameters.

    Attributes
    ----------
    eta : float
        Overall detection efficiency (0..1). Includes detector quantum
        efficiency, optical coupling losses, and any filtering. Typical
        values for superconducting nanowire detectors (SNSPDs) range from
        0.7-0.9; for InGaAs APDs, 0.1-0.25.
        Default: 0.2 (conservative, representative of gated APD systems).

    p_bg : float
        Background/dark click probability per pulse window (0..1). This
        lumps together detector dark counts, stray light, and timing-window
        effects. For a dark count rate D (Hz) and gate width T (s):
        p_bg ≈ D * T. Typical values: 1e-5 to 1e-3.
        Default: 1e-4 (order-of-magnitude placeholder).

    p_afterpulse : float
        Afterpulsing probability (0..1). Probability that a detection
        triggers a spurious subsequent detection. Included as a stub for
        future modeling; not currently used in simulations.
        Default: 0.0.

    Notes
    -----
    This is a simplified model suitable for demonstrating the key effect:
    at high channel loss, the signal-to-noise ratio degrades because
    background clicks (which yield random bits) become comparable to or
    dominate signal clicks. This causes QBER to approach 0.5.

    A more complete model would include:
    - Time-resolved detection with jitter and dead time
    - Afterpulsing correlations
    - Multiphoton detection effects
    - Detector saturation at high rates
    """
    eta: float = 0.2
    p_bg: float = 1e-4
    p_afterpulse: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError(f"eta must be in [0, 1], got {self.eta}")
        if not 0.0 <= self.p_bg <= 1.0:
            raise ValueError(f"p_bg must be in [0, 1], got {self.p_bg}")
        if not 0.0 <= self.p_afterpulse <= 1.0:
            raise ValueError(f"p_afterpulse must be in [0, 1], got {self.p_afterpulse}")


# Convenient default instance
DEFAULT_DETECTOR = DetectorParams()
