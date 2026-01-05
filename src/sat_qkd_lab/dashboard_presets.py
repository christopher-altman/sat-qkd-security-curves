"""
Scenario presets for the dashboard operator console.

Each preset bundles a coherent set of parameter values for common
operational scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DashboardPreset:
    """A named preset with parameter values for dashboard controls."""

    name: str
    description: str
    sweep_params: Dict[str, Any]
    pass_params: Dict[str, Any]


# Preset catalog
PRESETS = [
    DashboardPreset(
        name="Night / Low Background / Low Turbulence",
        description="Baseline night-time pass with minimal atmospheric disturbance",
        sweep_params={
            "loss_min": 20.0,
            "loss_max": 60.0,
            "steps": 21,
            "flip_prob": 0.005,
            "pulses": 200000,
            "eta": 0.2,
            "p_bg": 1e-4,
        },
        pass_params={
            "max_elevation": 60.0,
            "min_elevation": 10.0,
            "pass_seconds": 300.0,
            "dt_seconds": 5.0,
            "rep_rate": 1e8,
            "turbulence": False,
            "day": False,
        },
    ),
    DashboardPreset(
        name="Day / High Background / Moderate Turbulence",
        description="Daytime pass with increased solar background and atmospheric turbulence",
        sweep_params={
            "loss_min": 20.0,
            "loss_max": 60.0,
            "steps": 21,
            "flip_prob": 0.005,
            "pulses": 200000,
            "eta": 0.2,
            "p_bg": 1e-2,  # 100x higher background
        },
        pass_params={
            "max_elevation": 70.0,
            "min_elevation": 15.0,
            "pass_seconds": 300.0,
            "dt_seconds": 5.0,
            "rep_rate": 1e8,
            "turbulence": True,
            "day": True,
        },
    ),
    DashboardPreset(
        name="High Jitter / Timing Stress",
        description="Low detector efficiency with high background to stress timing margins",
        sweep_params={
            "loss_min": 20.0,
            "loss_max": 60.0,
            "steps": 21,
            "flip_prob": 0.01,  # Higher intrinsic errors
            "pulses": 200000,
            "eta": 0.1,  # Low efficiency
            "p_bg": 5e-4,
        },
        pass_params={
            "max_elevation": 50.0,
            "min_elevation": 10.0,
            "pass_seconds": 300.0,
            "dt_seconds": 5.0,
            "rep_rate": 5e7,  # Lower rep rate
            "turbulence": True,
            "day": False,
        },
    ),
    DashboardPreset(
        name="High Loss / Near Cliff",
        description="Extended loss range to probe security boundary and abort threshold",
        sweep_params={
            "loss_min": 40.0,
            "loss_max": 70.0,  # Extended to probe cliff
            "steps": 31,
            "flip_prob": 0.005,
            "pulses": 200000,
            "eta": 0.2,
            "p_bg": 1e-4,
        },
        pass_params={
            "max_elevation": 45.0,  # Lower max elev → higher loss
            "min_elevation": 10.0,
            "pass_seconds": 300.0,
            "dt_seconds": 5.0,
            "rep_rate": 1e8,
            "turbulence": False,
            "day": False,
        },
    ),
]


def get_preset_by_name(name: str) -> DashboardPreset:
    """Retrieve a preset by name."""
    for preset in PRESETS:
        if preset.name == name:
            return preset
    raise ValueError(f"Unknown preset: {name}")


def list_preset_names() -> list[str]:
    """Return list of all preset names."""
    return [p.name for p in PRESETS]
