"""
Fisher information identifiability utilities for calibration uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

from .telemetry import TelemetryRecord
from .pass_model import expected_qber


PARAM_ORDER = ("eta_scale", "p_bg", "flip_prob")


@dataclass(frozen=True)
class IdentifiabilityResult:
    params: List[str]
    covariance: List[List[float]]
    condition_number: float
    is_degenerate: bool
    warnings: List[str]


def _param_vector(params: Dict[str, float]) -> np.ndarray:
    return np.array([float(params[name]) for name in PARAM_ORDER], dtype=float)


def _vector_to_params(values: Sequence[float]) -> Dict[str, float]:
    return {name: float(values[idx]) for idx, name in enumerate(PARAM_ORDER)}


def _finite_diff_grad(
    func: Callable[[Dict[str, float]], float],
    params: Dict[str, float],
    rel_step: float = 1e-3,
    abs_step: float = 1e-8,
) -> np.ndarray:
    base = _param_vector(params)
    grad = np.zeros_like(base)
    for idx, name in enumerate(PARAM_ORDER):
        step = max(abs_step, abs(base[idx]) * rel_step)
        plus = base.copy()
        minus = base.copy()
        plus[idx] += step
        minus[idx] -= step
        grad[idx] = (func(_vector_to_params(plus)) - func(_vector_to_params(minus))) / (2.0 * step)
    return grad


def compute_fim_identifiability(
    records: Iterable[TelemetryRecord],
    eta_base: float,
    params: Dict[str, float],
    sigma: float = 0.005,
    cond_threshold: float = 1e14,
) -> IdentifiabilityResult:
    """
    Compute Fisher information covariance and identify degeneracies.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive for identifiability analysis.")

    records_list = list(records)
    if not records_list:
        return IdentifiabilityResult(
            params=list(PARAM_ORDER),
            covariance=[[float("nan")] * len(PARAM_ORDER)] * len(PARAM_ORDER),
            condition_number=float("inf"),
            is_degenerate=True,
            warnings=["no telemetry records available"],
        )

    def qber_model(param_values: Dict[str, float], loss_db: float) -> float:
        eta_ch = 10 ** (-loss_db / 10.0)
        p_sig = eta_base * param_values["eta_scale"] * eta_ch
        return expected_qber(p_sig, param_values["p_bg"], param_values["flip_prob"])

    jacobian = []
    for record in records_list:
        grad = _finite_diff_grad(
            lambda p: qber_model(p, record.loss_db),
            params,
        )
        jacobian.append(grad)
    jacobian_mat = np.vstack(jacobian)
    scale = np.maximum(np.abs(_param_vector(params)), 1e-6)
    jacobian_scaled = jacobian_mat * scale
    fim_scaled = (jacobian_scaled.T @ jacobian_scaled) / (sigma ** 2)

    warnings: List[str] = []
    try:
        cond = float(np.linalg.cond(fim_scaled))
    except np.linalg.LinAlgError:
        cond = float("inf")

    rank = int(np.linalg.matrix_rank(fim_scaled))
    is_degenerate = rank < len(PARAM_ORDER) or not np.isfinite(cond) or cond > cond_threshold
    if rank < len(PARAM_ORDER):
        warnings.append("fim_rank_deficient")
    if not np.isfinite(cond) or cond > cond_threshold:
        warnings.append("fim_condition_number_high")

    if rank == len(PARAM_ORDER) and np.isfinite(cond):
        cov_scaled = np.linalg.inv(fim_scaled)
    else:
        cov_scaled = np.linalg.pinv(fim_scaled)
    scale_mat = np.diag(scale)
    cov = scale_mat @ cov_scaled @ scale_mat

    return IdentifiabilityResult(
        params=list(PARAM_ORDER),
        covariance=cov.tolist(),
        condition_number=cond,
        is_degenerate=is_degenerate,
        warnings=warnings,
    )


def propagate_uncertainty(
    metric_func: Callable[[Dict[str, float]], float],
    params: Dict[str, float],
    covariance: Sequence[Sequence[float]],
    rel_step: float = 1e-3,
    abs_step: float = 1e-8,
) -> Tuple[float, float]:
    """
    Propagate parameter covariance to metric uncertainty via linearization.
    """
    cov = np.array(covariance, dtype=float)
    mean = float(metric_func(params))
    grad = _finite_diff_grad(metric_func, params, rel_step=rel_step, abs_step=abs_step)
    variance = float(grad.T @ cov @ grad)
    return mean, float(np.sqrt(max(0.0, variance)))
