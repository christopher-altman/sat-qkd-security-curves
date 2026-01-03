import json
from pathlib import Path

import numpy as np

from sat_qkd_lab.calibration_fit import fit_telemetry_parameters, predict_qber, compute_fit_quality
from sat_qkd_lab.telemetry import load_telemetry, TelemetryRecord


def test_fit_recovers_parameters():
    eta_base = 0.2
    true_params = {
        "eta_scale": 0.8,
        "p_bg": 1e-4,
        "flip_prob": 0.01,
    }
    losses = [10, 20, 30, 40]
    records = [
        TelemetryRecord(
            loss_db=loss,
            qber_mean=predict_qber(
                loss_db=loss,
                eta_scale=true_params["eta_scale"],
                p_bg=true_params["p_bg"],
                flip_prob=true_params["flip_prob"],
                eta_base=eta_base,
            ),
        )
        for loss in losses
    ]

    fit = fit_telemetry_parameters(
        records=records,
        eta_base=eta_base,
        p_bg_grid=np.linspace(5e-5, 2e-4, 7),
        flip_grid=np.linspace(0.0, 0.02, 9),
        eta_scale_grid=np.linspace(0.6, 1.0, 9),
    )

    assert abs(fit.eta_scale - true_params["eta_scale"]) <= 0.1
    assert abs(fit.p_bg - true_params["p_bg"]) <= 1e-4
    assert abs(fit.flip_prob - true_params["flip_prob"]) <= 0.01


def test_load_telemetry_json(tmp_path: Path):
    payload = [
        {"loss_db": 20.0, "qber_mean": 0.03, "n_sent": 1000},
        {"loss_db": 30.0, "qber_mean": 0.04},
    ]
    path = tmp_path / "telemetry.json"
    path.write_text(json.dumps(payload))
    records = load_telemetry(str(path))
    assert len(records) == 2
    assert records[0].loss_db == 20.0
    assert records[0].qber_mean == 0.03
    assert records[0].n_sent == 1000


def test_fit_instrument_params_from_telemetry():
    eta_base = 0.2
    record = TelemetryRecord(
        loss_db=20.0,
        qber_mean=predict_qber(
            loss_db=20.0,
            eta_scale=0.9,
            p_bg=1e-4,
            flip_prob=0.01,
            eta_base=eta_base,
        ),
        coincidence_histogram=[0, 1, 6, 2, 1, 0, 0],
        coincidence_bin_seconds=1e-9,
        off_window_counts=120.0,
        off_window_seconds=12.0,
        transmittance_series=[0.09, 0.1, 0.11, 0.1],
    )

    fit = fit_telemetry_parameters(
        records=[record],
        eta_base=eta_base,
        p_bg_grid=np.array([1e-4]),
        flip_grid=np.array([0.01]),
        eta_scale_grid=np.array([0.9]),
    )

    assert fit.clock_offset_s is not None
    assert abs(fit.clock_offset_s - (-1e-9)) <= 1e-12
    assert fit.background_rate is not None
    assert abs(fit.background_rate - 10.0) <= 1e-9
    assert fit.pointing_jitter_sigma is not None
    expected_sigma = float(np.std(record.transmittance_series, ddof=1))
    assert abs(fit.pointing_jitter_sigma - expected_sigma) <= 1e-12


def test_fit_quality_metrics():
    eta_base = 0.2
    true_params = {
        "eta_scale": 0.85,
        "p_bg": 2e-4,
        "flip_prob": 0.02,
    }
    records = [
        TelemetryRecord(
            loss_db=loss,
            qber_mean=predict_qber(
                loss_db=loss,
                eta_scale=true_params["eta_scale"],
                p_bg=true_params["p_bg"],
                flip_prob=true_params["flip_prob"],
                eta_base=eta_base,
            ),
        )
        for loss in [10, 20, 30, 40]
    ]
    fit = fit_telemetry_parameters(
        records=records,
        eta_base=eta_base,
        p_bg_grid=np.array([true_params["p_bg"]]),
        flip_grid=np.array([true_params["flip_prob"]]),
        eta_scale_grid=np.array([true_params["eta_scale"]]),
    )
    quality = compute_fit_quality(records, fit, eta_base)
    assert quality["r2"] > 0.99
    assert "parameter_uncertainty" in quality
    assert quality["identifiable"] in (True, False)
