import numpy as np

from sat_qkd_lab.calibration_fit import FitResult, compute_residual_diagnostics, predict_qber
from sat_qkd_lab.telemetry import TelemetryRecord


def _make_records(loss_db, qber):
    return [TelemetryRecord(loss_db=float(l), qber_mean=float(q)) for l, q in zip(loss_db, qber)]


def test_residuals_low_structure_for_good_model():
    loss_db = np.linspace(20.0, 40.0, 15)
    eta_base = 0.2
    fit = FitResult(eta_scale=1.0, p_bg=1e-4, flip_prob=0.01, rmse=0.0, residual_std=0.0)
    pred = np.array([predict_qber(l, fit.eta_scale, fit.p_bg, fit.flip_prob, eta_base) for l in loss_db])
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1e-4, size=loss_db.size)
    records = _make_records(loss_db, pred + noise)
    diag = compute_residual_diagnostics(records, fit, eta_base)
    assert abs(diag["autocorr_lag1"]) < 0.4
    assert diag["overfit_warning"] is False


def test_residuals_trigger_warning_for_structured_error():
    loss_db = np.linspace(20.0, 40.0, 20)
    eta_base = 0.2
    fit = FitResult(eta_scale=1.0, p_bg=1e-4, flip_prob=0.01, rmse=0.0, residual_std=0.0)
    pred = np.array([predict_qber(l, fit.eta_scale, fit.p_bg, fit.flip_prob, eta_base) for l in loss_db])
    trend = np.linspace(0.0, 0.01, loss_db.size)
    records = _make_records(loss_db, pred + trend)
    diag = compute_residual_diagnostics(records, fit, eta_base)
    assert diag["overfit_warning"] is True
