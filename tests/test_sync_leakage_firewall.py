import json
from pathlib import Path

import pytest

from sat_qkd_lab.clock_sync import SyncParams, write_sync_params
from sat_qkd_lab.run import build_parser, _run_coincidence_sim


def _run_coincidence(tmp_path: Path, extra_args):
    parser = build_parser()
    args = parser.parse_args([
        "coincidence-sim",
        "--loss-min", "20",
        "--loss-max", "20",
        "--steps", "1",
        "--duration", "0.01",
        "--pair-rate-hz", "50",
        "--background-rate-hz", "0",
        "--outdir", str(tmp_path),
    ] + extra_args)
    _run_coincidence_sim(args)
    report_path = tmp_path / "reports" / "latest_coincidence.json"
    return json.loads(report_path.read_text())


def test_scoring_does_not_call_resync(monkeypatch, tmp_path: Path):
    def _boom(*args, **kwargs):
        raise AssertionError("Resync estimator should not be called")

    monkeypatch.setattr("sat_qkd_lab.timing.estimate_clock_offset", _boom)
    sync_path = tmp_path / "sync_params.json"
    write_sync_params(str(sync_path), SyncParams(offset_s=1e-6, drift_ppm=2.0, source="beacon"))
    output = _run_coincidence(tmp_path, ["--sync-params", str(sync_path), "--seed", "1"])
    assert output["sync"]["locked"] is True
    assert output["sync"]["source"] == "beacon"


def test_locked_sync_params_stable_across_data(tmp_path: Path):
    sync_path = tmp_path / "sync_params.json"
    write_sync_params(str(sync_path), SyncParams(offset_s=1e-6, drift_ppm=2.0, source="beacon"))
    output_a = _run_coincidence(tmp_path, ["--sync-params", str(sync_path), "--seed", "1"])
    output_b = _run_coincidence(tmp_path, ["--sync-params", str(sync_path), "--seed", "2"])
    assert output_a["sync"]["offset_s"] == pytest.approx(1e-6)
    assert output_b["sync"]["offset_s"] == pytest.approx(1e-6)
    assert output_a["timing_model"]["estimated_clock_offset_s"] is None
    assert output_b["timing_model"]["estimated_clock_offset_s"] is None


def test_resync_requires_explicit_allow(tmp_path: Path):
    parser = build_parser()
    args = parser.parse_args([
        "coincidence-sim",
        "--loss-min", "20",
        "--loss-max", "20",
        "--steps", "1",
        "--duration", "0.01",
        "--pair-rate-hz", "50",
        "--background-rate-hz", "0",
        "--outdir", str(tmp_path),
        "--estimate-offset",
    ])
    with pytest.raises(ValueError):
        _run_coincidence_sim(args)


def test_resync_allowed_marks_unlocked(tmp_path: Path):
    output = _run_coincidence(tmp_path, ["--estimate-offset", "--allow-resync", "--seed", "3"])
    assert output["sync"]["locked"] is False
    assert output["sync"]["source"] == "resync"
