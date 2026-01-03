import json
import pytest
from pathlib import Path

from sat_qkd_lab.forecast import Forecast, load_forecasts
from sat_qkd_lab.scoring import score_forecast
from sat_qkd_lab.windows import generate_windows, assign_groups_blinded
from sat_qkd_lab.forecast_harness import run_forecast_harness


def test_forecast_parsing_json(tmp_path: Path):
    payload = [
        {
            "forecast_id": "F001",
            "timestamp_utc": "2026-01-03T00:00:00Z",
            "window_id": "W000",
            "metric_name": "headroom",
            "operator": ">=",
            "value": 0.08,
        },
        {
            "forecast_id": "F002",
            "timestamp_utc": "2026-01-03T00:00:00Z",
            "window_id": "W001",
            "metric_name": "qber_mean",
            "operator": "==",
            "value": 0.03,
        },
        {
            "forecast_id": "F003",
            "timestamp_utc": "2026-01-03T00:00:00Z",
            "window_id": "W002",
            "metric_name": "total_secret_bits",
            "operator": "increases",
            "value": None,
        },
    ]
    path = tmp_path / "forecasts.json"
    path.write_text(json.dumps(payload))
    forecasts = load_forecasts(str(path))
    assert len(forecasts) == 3
    assert forecasts[0].operator == ">="
    assert forecasts[1].operator == "=="
    assert forecasts[2].operator == "increases"


def test_windows_deterministic():
    windows = generate_windows(3, 10.0, seed=7)
    assignments_a = assign_groups_blinded(windows, seed=7)
    assignments_b = assign_groups_blinded(windows, seed=7)
    assert assignments_a["labels_by_window"] == assignments_b["labels_by_window"]


def test_scoring_threshold_and_point():
    forecast_threshold = Forecast(
        forecast_id="F001",
        timestamp_utc=None,
        window_id="W000",
        metric_name="headroom",
        operator=">=",
        value=0.1,
    )
    forecast_point = Forecast(
        forecast_id="F002",
        timestamp_utc=None,
        window_id="W000",
        metric_name="qber_mean",
        operator="==",
        value=0.03,
    )
    assert score_forecast(forecast_threshold, outcome=0.2)["hit"] is True
    assert score_forecast(forecast_point, outcome=0.031)["error"] == pytest.approx(0.001)


def test_forecast_harness_blinding(tmp_path: Path):
    payload = [
        {
            "forecast_id": "F001",
            "timestamp_utc": "2026-01-03T00:00:00Z",
            "window_id": "W000",
            "metric_name": "headroom",
            "operator": ">=",
            "value": 0.0,
        }
    ]
    path = tmp_path / "forecasts.json"
    path.write_text(json.dumps(payload))

    run_forecast_harness(
        forecasts_path=str(path),
        outdir=tmp_path,
        seed=5,
        n_blocks=2,
        block_seconds=10.0,
        rep_rate_hz=1e6,
        unblind=False,
    )

    blinded_path = tmp_path / "reports" / "forecast_blinded.json"
    assert blinded_path.exists()
    text = blinded_path.read_text()
    assert "control" not in text
    assert "intervention" not in text

    unblinded_path = tmp_path / "reports" / "forecast_unblinded.json"
    assert not unblinded_path.exists()

    run_forecast_harness(
        forecasts_path=str(path),
        outdir=tmp_path,
        seed=5,
        n_blocks=2,
        block_seconds=10.0,
        rep_rate_hz=1e6,
        unblind=True,
    )
    assert unblinded_path.exists()
    output = json.loads(unblinded_path.read_text())
    assert output["analysis"]["group_labels_included"] is True


def test_forecast_harness_identifiability_outputs(tmp_path: Path):
    payload = [
        {
            "forecast_id": "F001",
            "timestamp_utc": "2026-01-03T00:00:00Z",
            "window_id": "W000",
            "metric_name": "headroom",
            "operator": ">=",
            "value": 0.0,
        }
    ]
    path = tmp_path / "forecasts.json"
    path.write_text(json.dumps(payload))

    output = run_forecast_harness(
        forecasts_path=str(path),
        outdir=tmp_path,
        seed=3,
        n_blocks=2,
        block_seconds=10.0,
        rep_rate_hz=1e6,
        unblind=False,
        estimate_identifiability=True,
    )

    assert "identifiability" in output
    assert "uncertainty" in output
    metrics = output["uncertainty"]["metrics"]
    assert "key_rate_bps" in metrics
    assert "headroom" in metrics


def test_forecast_harness_fdr_outputs(tmp_path: Path):
    payload = [
        {
            "forecast_id": "F001",
            "timestamp_utc": "2026-01-03T00:00:00Z",
            "window_id": "W000",
            "metric_name": "headroom",
            "operator": ">=",
            "value": 0.0,
        }
    ]
    path = tmp_path / "forecasts.json"
    path.write_text(json.dumps(payload))

    output = run_forecast_harness(
        forecasts_path=str(path),
        outdir=tmp_path,
        seed=4,
        n_blocks=2,
        block_seconds=10.0,
        rep_rate_hz=1e6,
        unblind=False,
        fdr_enabled=True,
        fdr_alpha=0.1,
    )
    assert output["fdr"]["enabled"] is True
    assert output["scores"][0]["q_value"] is not None
    assert output["scores"][0]["p_value"] is not None
