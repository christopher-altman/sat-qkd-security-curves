import json
from pathlib import Path

from sat_qkd_lab.run import build_parser, _run_pass_sweep as _run_pass_sweep_cmd


def _run_pass_sweep_report(tmp_path: Path, sigma: float, seed: int) -> dict:
    parser = build_parser()
    args = parser.parse_args([
        "pass-sweep",
        "--max-elevation", "20",
        "--min-elevation", "10",
        "--pass-duration", "4",
        "--time-step", "1",
        "--pulses", "50",
        "--seed", str(seed),
        "--fading-ou",
        "--fading-ou-mean", "0.7",
        "--fading-ou-sigma", str(sigma),
        "--fading-ou-tau-s", "2.0",
        "--fading-ou-seed", "5",
        "--fading-ou-outage-threshold", "0.6",
        "--outdir", str(tmp_path),
    ])
    _run_pass_sweep_cmd(args)
    report_path = tmp_path / "reports" / "latest_pass.json"
    return json.loads(report_path.read_text())


def test_ou_fading_reproducible(tmp_path: Path) -> None:
    report_a = _run_pass_sweep_report(tmp_path / "a", sigma=0.1, seed=1)
    report_b = _run_pass_sweep_report(tmp_path / "b", sigma=0.1, seed=1)
    assert report_a["summary"]["fading"]["mean"] == report_b["summary"]["fading"]["mean"]
    assert report_a["summary"]["outages"]["count"] == report_b["summary"]["outages"]["count"]


def test_ou_fading_sigma_increases_outages(tmp_path: Path) -> None:
    low = _run_pass_sweep_report(tmp_path / "low", sigma=0.05, seed=2)
    high = _run_pass_sweep_report(tmp_path / "high", sigma=0.25, seed=2)
    assert high["summary"]["outages"]["count"] >= low["summary"]["outages"]["count"]
