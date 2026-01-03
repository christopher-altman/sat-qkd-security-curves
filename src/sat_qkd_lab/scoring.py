"""
Forecast scoring utilities.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np

from .forecast import Forecast


def robust_z_score(value: float, values: Sequence[float]) -> Optional[float]:
    """Compute robust z-score using median and MAD."""
    vals = np.array([v for v in values if v is not None and v == v], dtype=float)
    if vals.size == 0:
        return None
    median = float(np.median(vals))
    mad = float(np.median(np.abs(vals - median)))
    if mad == 0:
        return None
    return (float(value) - median) / mad


def score_forecast(
    forecast: Forecast,
    outcome: Optional[float],
    baseline: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """
    Score a forecast against a numeric outcome.

    Returns:
        dict with hit (bool or None) and error (float or None).
    """
    if outcome is None:
        return {"hit": None, "error": None}

    if forecast.operator == ">=":
        return {"hit": bool(outcome >= forecast.value), "error": None}
    if forecast.operator == "<=":
        return {"hit": bool(outcome <= forecast.value), "error": None}
    if forecast.operator == "==":
        return {"hit": None, "error": abs(outcome - float(forecast.value))}
    if forecast.operator in ("increases", "decreases"):
        if baseline is None:
            return {"hit": None, "error": None}
        if forecast.operator == "increases":
            return {"hit": bool(outcome > baseline), "error": None}
        return {"hit": bool(outcome < baseline), "error": None}

    raise ValueError(f"Unsupported operator: {forecast.operator}")
