import json
from pathlib import Path

import numpy as np

from sat_qkd_lab.calibration_fit import fit_telemetry_parameters, predict_qber
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
