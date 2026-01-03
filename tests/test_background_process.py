import json
from pathlib import Path

import numpy as np

from sat_qkd_lab.background_process import BackgroundProcessParams, simulate_background_scales
from sat_qkd_lab.run import build_parser, _run_coincidence_sim


def test_background_process_repeatable():
    time_s = np.linspace(0.0, 10.0, 6)
    params = BackgroundProcessParams(enabled=True, mean=1.0, sigma=0.1, tau_seconds=5.0, seed=42)
    series_a = simulate_background_scales(time_s, params)
    series_b = simulate_background_scales(time_s, params)
    assert np.allclose(series_a, series_b)
    series_c = simulate_background_scales(time_s, BackgroundProcessParams(enabled=True, mean=1.0, sigma=0.1, tau_seconds=5.0, seed=43))
    assert not np.allclose(series_a, series_c)


def _run_coincidence(tmp_path: Path, extra_args):
    parser = build_parser()
    args = parser.parse_args([
        "coincidence-sim",
        "--loss-min", "20",
        "--loss-max", "20",
        "--steps", "3",
        "--duration", "0.03",
        "--pair-rate-hz", "500",
        "--background-rate-hz", "50",
        "--tau-ps", "200",
        "--outdir", str(tmp_path),
        "--seed", "7",
    ] + extra_args)
    _run_coincidence_sim(args)
    report_path = tmp_path / "reports" / "latest_coincidence.json"
    return json.loads(report_path.read_text())


def test_background_process_degrades_car(tmp_path: Path):
    base = _run_coincidence(tmp_path / "base", [])
    scaled = _run_coincidence(
        tmp_path / "scaled",
        ["--background-process", "--bg-ou-mean", "5.0", "--bg-ou-sigma", "0.0"],
    )
    assert scaled["summary"]["mean_car"] <= base["summary"]["mean_car"]
