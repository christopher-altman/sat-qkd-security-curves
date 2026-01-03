import json
from pathlib import Path

import pytest

from sat_qkd_lab.clock_sync import SyncParams, write_sync_params
from sat_qkd_lab.run import build_parser, _run_sync_estimate, _run_experiment, _run_forecast_run


def test_sync_estimate_writes_params(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args([
        "sync-estimate",
        "--duration-s", "1.0",
        "--rate-hz", "1000",
        "--offset-s", "2e-9",
        "--drift-ppm", "3.0",
        "--jitter-ps", "0.0",
        "--seed", "7",
        "--outdir", str(tmp_path),
    ])
    _run_sync_estimate(args)
    report_path = tmp_path / "reports" / "latest_sync_params.json"
    payload = json.loads(report_path.read_text())
    assert payload["mode"] == "sync-params"
    assert payload["source"] == "beacon"
    assert payload["offset_s"] == pytest.approx(2e-9, rel=0, abs=1e-11)
    assert payload["drift_ppm"] == pytest.approx(3.0, rel=0, abs=1e-3)


def test_experiment_and_forecast_accept_sync_params(tmp_path: Path) -> None:
    sync_path = tmp_path / "sync_params.json"
    write_sync_params(str(sync_path), SyncParams(offset_s=1e-6, drift_ppm=2.0, source="beacon"))

    parser = build_parser()
    exp_args = parser.parse_args([
        "experiment-run",
        "--n-blocks", "2",
        "--block-seconds", "1.0",
        "--rep-rate-hz", "1000",
        "--pass-seconds", "2.0",
        "--sync-params", str(sync_path),
        "--outdir", str(tmp_path / "exp"),
    ])
    _run_experiment(exp_args)
    exp_report = json.loads((tmp_path / "exp" / "reports" / "latest_experiment.json").read_text())
    assert exp_report["inputs"]["sync_params"] == str(sync_path)

    forecast_path = tmp_path / "forecasts.json"
    forecast_path.write_text(json.dumps([
        {
            "forecast_id": "F001",
            "timestamp_utc": "2026-01-03T00:00:00Z",
            "window_id": "W000",
            "metric_name": "headroom",
            "operator": ">=",
            "value": 0.0,
        }
    ]))
    fc_args = parser.parse_args([
        "forecast-run",
        "--forecasts", str(forecast_path),
        "--n-blocks", "1",
        "--block-seconds", "1.0",
        "--rep-rate-hz", "1000",
        "--sync-params", str(sync_path),
        "--outdir", str(tmp_path / "fc"),
    ])
    _run_forecast_run(fc_args)
    fc_report = json.loads((tmp_path / "fc" / "reports" / "forecast_blinded.json").read_text())
    assert fc_report["inputs"]["sync_params"] == str(sync_path)
