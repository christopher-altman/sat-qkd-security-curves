import json
from pathlib import Path

from sat_qkd_lab.run import build_parser, _run_pass_sweep
from sat_qkd_lab.experiment import ExperimentParams, run_experiment
from sat_qkd_lab.forecast_harness import run_forecast_harness


def _run_pass_sweep_minimal(tmp_path: Path) -> dict:
    parser = build_parser()
    args = parser.parse_args([
        "pass-sweep",
        "--max-elevation", "20",
        "--min-elevation", "10",
        "--pass-duration", "2",
        "--time-step", "1",
        "--pulses", "20",
        "--outdir", str(tmp_path),
        "--seed", "1",
    ])
    _run_pass_sweep(args)
    report_path = tmp_path / "reports" / "latest_pass.json"
    return json.loads(report_path.read_text())


def test_units_explicit_in_reports(tmp_path: Path):
    pass_report = _run_pass_sweep_minimal(tmp_path)
    units = pass_report["units"]
    assert units["key_rate_bps"] == "bits/s"
    assert units["rep_rate_hz"] == "Hz"
    assert units["pass_seconds"] == "s"
    assert units["secret_bits"] == "bits"


def test_blinding_outputs_no_labels(tmp_path: Path):
    exp_params = ExperimentParams(seed=1, n_blocks=2, block_seconds=5.0, rep_rate_hz=1e6, pass_seconds=10.0)
    output = run_experiment(
        params=exp_params,
        metrics=["qber_mean"],
        outdir=tmp_path,
        finite_key=None,
        unblind=False,
    )
    assert output["analysis"]["group_labels_included"] is False
    schedule_text = (tmp_path / "reports" / "schedule_blinded.json").read_text()
    assert "control" not in schedule_text
    assert "intervention" not in schedule_text


def test_forecast_schema_keys_stable(tmp_path: Path):
    forecasts_path = tmp_path / "forecasts.json"
    forecasts_path.write_text(json.dumps([{
        "forecast_id": "F1",
        "timestamp_utc": "2026-01-03T00:00:00Z",
        "window_id": "W000",
        "metric_name": "headroom",
        "operator": ">=",
        "value": 0.0,
    }]))
    output = run_forecast_harness(
        forecasts_path=str(forecasts_path),
        outdir=tmp_path,
        seed=2,
        n_blocks=2,
        block_seconds=10.0,
        rep_rate_hz=1e6,
        unblind=False,
    )
    for key in ("schema_version", "mode", "inputs", "blinding", "windows", "outcomes", "scores", "summary"):
        assert key in output


def test_ci_gated_when_disabled(tmp_path: Path):
    pass_report = _run_pass_sweep_minimal(tmp_path)
    time_series = pass_report["time_series"]
    assert "qber_ci_low" not in time_series
    assert "qber_ci_high" not in time_series
