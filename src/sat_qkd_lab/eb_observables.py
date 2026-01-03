"""
Entanglement observables: visibility, CHSH S, and correlation coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math


@dataclass(frozen=True)
class EBObservableResult:
    visibility: float
    chsh_s: float
    chsh_sigma: float
    correlations: Dict[str, float]
    aborted: bool


def correlation_from_counts(counts: List[List[int]]) -> float:
    """
    Compute correlation coefficient E from 2x2 coincidence counts.
    """
    n00 = counts[0][0]
    n01 = counts[0][1]
    n10 = counts[1][0]
    n11 = counts[1][1]
    total = n00 + n01 + n10 + n11
    if total == 0:
        return 0.0
    return (n00 + n11 - n01 - n10) / total


def visibility_from_correlations(correlations: Dict[str, float]) -> float:
    """
    Approximate visibility from correlation magnitudes.
    """
    if not correlations:
        return 0.0
    values = [abs(v) for v in correlations.values()]
    return float(sum(values) / len(values))


def _sigma_chsh(n_pairs: int) -> float:
    """
    Analytic standard deviation for CHSH S using binomial approximation.
    """
    if n_pairs <= 0:
        return 0.0
    return math.sqrt(4.0 / n_pairs)


def chsh_s_from_correlations(correlations: Dict[str, float]) -> float:
    """
    Compute CHSH S from correlations for AB, AB', A'B, A'B'.
    """
    e_ab = correlations.get("AB", 0.0)
    e_abp = correlations.get("ABp", 0.0)
    e_apb = correlations.get("ApB", 0.0)
    e_apbp = correlations.get("ApBp", 0.0)
    return e_ab + e_abp + e_apb - e_apbp


def compute_observables(
    correlation_counts: Dict[str, List[List[int]]],
    n_pairs: int,
    min_visibility: float = 0.0,
    min_chsh_s: float = 2.0,
) -> EBObservableResult:
    """
    Compute visibility, CHSH S, and correlation coefficients.
    """
    correlations = {
        key: correlation_from_counts(counts)
        for key, counts in correlation_counts.items()
    }
    visibility = visibility_from_correlations(correlations)
    chsh_s = chsh_s_from_correlations(correlations)
    chsh_sigma = _sigma_chsh(n_pairs)
    aborted = visibility < min_visibility or chsh_s < min_chsh_s
    return EBObservableResult(
        visibility=visibility,
        chsh_s=chsh_s,
        chsh_sigma=chsh_sigma,
        correlations=correlations,
        aborted=aborted,
    )
