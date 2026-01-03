"""
Telemetry calibration fitting for detector/background parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .telemetry import TelemetryRecord
from .pass_model import expected_qber


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
