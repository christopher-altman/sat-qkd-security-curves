"""
Forecast scoring utilities.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import math
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


def normal_cdf(value: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def two_sided_p_value(z_score: float) -> float:
    """Two-sided p-value from a z-score."""
    return max(0.0, min(1.0, 2.0 * (1.0 - normal_cdf(abs(z_score)))))


def bh_fdr(p_values: Sequence[Optional[float]], alpha: float) -> List[Optional[float]]:
    """
    Benjamini-Hochberg FDR adjustment returning q-values aligned to input order.
    """
    indexed = [(idx, p) for idx, p in enumerate(p_values) if p is not None]
    if not indexed:
        return [None for _ in p_values]
    indexed.sort(key=lambda item: item[1])
    n = len(indexed)
    q_values = [None for _ in p_values]
    prev = 1.0
    for rank, (idx, p_val) in enumerate(reversed(indexed), start=1):
        k = n - rank + 1
        q = min(prev, p_val * n / float(k))
        prev = q
        q_values[idx] = min(1.0, q)
    return q_values


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
