"""
Optical chain background and dark count models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DayNight = Literal["day", "night"]


@dataclass(frozen=True)
class OpticalParams:
    filter_bandwidth_nm: float = 1.0
    detector_temp_c: float = 20.0
    mode: DayNight = "night"


def background_rate_hz(params: OpticalParams, base_rate_hz: float) -> float:
    """
    Scale background rate with filter bandwidth and day/night mode.
    """
    if params.filter_bandwidth_nm <= 0:
        raise ValueError("filter_bandwidth_nm must be > 0")
    mode_factor = 1.0 if params.mode == "night" else 100.0
    return base_rate_hz * params.filter_bandwidth_nm * mode_factor


def dark_count_rate_hz(params: OpticalParams, base_rate_hz: float) -> float:
    """
    Linear temperature scaling for dark count rate.
    """
    slope = 0.02  # per degree C
    delta = max(-50.0, min(50.0, params.detector_temp_c - 20.0))
    scale = 1.0 + slope * delta
    if scale < 0.0:
        scale = 0.0
    return base_rate_hz * scale
