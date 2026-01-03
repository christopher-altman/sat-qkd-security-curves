"""
Telemetry calibration fitting for detector/background parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .telemetry import TelemetryRecord
from .pass_model import expected_qber
from .fim_identifiability import compute_fim_identifiability


@dataclass(frozen=True)
class FitResult:
    eta_scale: float
    p_bg: float
    flip_prob: float
    rmse: float
    residual_std: float
    clock_offset_s: Optional[float] = None
    pointing_jitter_sigma: Optional[float] = None
    background_rate: Optional[float] = None


def predict_qber(
    loss_db: float,
    eta_scale: float,
    p_bg: float,
    flip_prob: float,
    eta_base: float,
) -> float:
    eta_ch = 10 ** (-loss_db / 10.0)
    p_sig = eta_base * eta_scale * eta_ch
    return expected_qber(p_sig, p_bg, flip_prob)


def fit_telemetry_parameters(
    records: List[TelemetryRecord],
    eta_base: float,
    p_bg_grid: np.ndarray,
    flip_grid: np.ndarray,
    eta_scale_grid: np.ndarray,
) -> FitResult:
    """
    Fit parameters using grid-search least squares.
    """
    loss_vals = np.array([r.loss_db for r in records], dtype=float)
    qber_vals = np.array([r.qber_mean for r in records], dtype=float)

    best = None
    best_rmse = float("inf")

    for eta_scale in eta_scale_grid:
        for p_bg in p_bg_grid:
            for flip_prob in flip_grid:
                pred = np.array([
                    predict_qber(loss, eta_scale, p_bg, flip_prob, eta_base)
                    for loss in loss_vals
                ])
                residuals = qber_vals - pred
                rmse = float(np.sqrt(np.mean(residuals ** 2)))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best = (eta_scale, p_bg, flip_prob, residuals)

    eta_scale, p_bg, flip_prob, residuals = best
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    return FitResult(
        eta_scale=float(eta_scale),
        p_bg=float(p_bg),
        flip_prob=float(flip_prob),
        rmse=float(best_rmse),
        residual_std=residual_std,
        clock_offset_s=_estimate_clock_offset(records),
        pointing_jitter_sigma=_estimate_pointing_jitter(records),
        background_rate=_estimate_background_rate(records),
    )


def predict_with_uncertainty(
    records: List[TelemetryRecord],
    fit: FitResult,
    eta_base: float,
) -> List[Dict[str, float]]:
    """
    Generate predicted QBER with simple uncertainty bounds from residuals.
    """
    out = []
    for r in records:
        pred = predict_qber(
            r.loss_db,
            fit.eta_scale,
            fit.p_bg,
            fit.flip_prob,
            eta_base,
        )
        ci_low = max(0.0, pred - 1.96 * fit.residual_std)
        ci_high = min(0.5, pred + 1.96 * fit.residual_std)
        out.append({
            "loss_db": float(r.loss_db),
            "qber_mean": float(pred),
            "qber_ci_low": float(ci_low),
            "qber_ci_high": float(ci_high),
        })
    return out


def compute_fit_quality(
    records: List[TelemetryRecord],
    fit: FitResult,
    eta_base: float,
) -> Dict[str, float | Dict[str, float]]:
    """
    Compute fit-quality metrics and identifiability diagnostics.
    """
    observed = np.array([r.qber_mean for r in records], dtype=float)
    predicted = np.array([
        predict_qber(r.loss_db, fit.eta_scale, fit.p_bg, fit.flip_prob, eta_base)
        for r in records
    ], dtype=float)
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    ident = compute_fim_identifiability(
        records=records,
        eta_base=eta_base,
        params={
            "eta_scale": fit.eta_scale,
            "p_bg": fit.p_bg,
            "flip_prob": fit.flip_prob,
        },
    )
    cov = np.array(ident.covariance, dtype=float)
    diag = np.sqrt(np.maximum(np.diag(cov), 0.0))
    param_uncertainty = {
        "eta_scale": float(diag[0]),
        "p_bg": float(diag[1]),
        "flip_prob": float(diag[2]),
    }

    return {
        "r2": float(r2),
        "condition_number": float(ident.condition_number),
        "identifiable": bool(not ident.is_degenerate),
        "parameter_uncertainty": param_uncertainty,
    }


def compute_residual_diagnostics(
    records: List[TelemetryRecord],
    fit: FitResult,
    eta_base: float,
    autocorr_lag: int = 1,
    warning_threshold: float = 0.4,
) -> Dict[str, float | list[float] | bool]:
    """
    Compute residual diagnostics with an autocorrelation proxy.
    """
    observed = np.array([r.qber_mean for r in records], dtype=float)
    predicted = np.array([
        predict_qber(r.loss_db, fit.eta_scale, fit.p_bg, fit.flip_prob, eta_base)
        for r in records
    ], dtype=float)
    residuals = observed - predicted
    autocorr = 0.0
    if residuals.size > autocorr_lag and np.std(residuals) > 0:
        x = residuals[:-autocorr_lag]
        y = residuals[autocorr_lag:]
        autocorr = float(np.corrcoef(x, y)[0, 1])
    structure = abs(autocorr)
    return {
        "residuals": [float(r) for r in residuals],
        "predicted": [float(p) for p in predicted],
        "observed": [float(o) for o in observed],
        "autocorr_lag1": autocorr,
        "structure_score": float(structure),
        "overfit_warning": bool(structure > warning_threshold),
    }


def _estimate_clock_offset(records: List[TelemetryRecord]) -> Optional[float]:
    offsets = []
    for record in records:
        hist = record.coincidence_histogram
        bin_s = record.coincidence_bin_seconds
        if not hist or bin_s is None:
            continue
        center = (len(hist) - 1) / 2.0
        peak_idx = int(np.argmax(hist))
        offsets.append((peak_idx - center) * float(bin_s))
    if not offsets:
        return None
    return float(np.mean(offsets))


def _estimate_pointing_jitter(records: List[TelemetryRecord]) -> Optional[float]:
    sigmas = []
    for record in records:
        series = record.transmittance_series
        if not series:
            continue
        sigmas.append(float(np.std(series, ddof=1)) if len(series) > 1 else 0.0)
    if not sigmas:
        return None
    return float(np.mean(sigmas))


def _estimate_background_rate(records: List[TelemetryRecord]) -> Optional[float]:
    rates = []
    for record in records:
        if record.background_rate is not None:
            rates.append(float(record.background_rate))
            continue
        if record.off_window_counts is not None and record.off_window_seconds:
            rates.append(float(record.off_window_counts) / float(record.off_window_seconds))
    if not rates:
        return None
    return float(np.mean(rates))
