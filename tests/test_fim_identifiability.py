import numpy as np

from sat_qkd_lab.calibration_fit import predict_qber
from sat_qkd_lab.fim_identifiability import compute_fim_identifiability, propagate_uncertainty
from sat_qkd_lab.telemetry import TelemetryRecord


def test_fim_identifiability_well_conditioned():
    eta_base = 0.2
    params = {"eta_scale": 0.9, "p_bg": 2e-4, "flip_prob": 0.01}
    losses = [10.0, 20.0, 35.0, 50.0]
    records = [
        TelemetryRecord(
            loss_db=loss,
            qber_mean=predict_qber(
                loss_db=loss,
                eta_scale=params["eta_scale"],
                p_bg=params["p_bg"],
                flip_prob=params["flip_prob"],
                eta_base=eta_base,
            ),
        )
        for loss in losses
    ]
    result = compute_fim_identifiability(records, eta_base=eta_base, params=params, sigma=0.002)
    assert result.is_degenerate is False
    assert result.condition_number < 1e14
    cov = np.array(result.covariance)
    assert cov.shape == (3, 3)
    assert np.all(np.diag(cov) >= 0.0)


def test_fim_identifiability_detects_degeneracy():
    eta_base = 0.2
    params = {"eta_scale": 1.0, "p_bg": 1e-4, "flip_prob": 0.01}
    records = [
        TelemetryRecord(loss_db=20.0, qber_mean=0.02),
        TelemetryRecord(loss_db=20.0, qber_mean=0.02),
        TelemetryRecord(loss_db=20.0, qber_mean=0.02),
    ]
    result = compute_fim_identifiability(records, eta_base=eta_base, params=params, sigma=0.01)
    assert result.is_degenerate is True


def test_uncertainty_propagation_returns_finite():
    params = {"eta_scale": 1.0, "p_bg": 1e-4, "flip_prob": 0.01}
    covariance = [
        [1e-4, 0.0, 0.0],
        [0.0, 1e-8, 0.0],
        [0.0, 0.0, 1e-5],
    ]

    mean, std = propagate_uncertainty(
        lambda p: p["eta_scale"] + p["p_bg"] * 1e3 + p["flip_prob"],
        params,
        covariance,
    )
    assert np.isfinite(mean)
    assert std > 0.0
