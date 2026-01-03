import json
from pathlib import Path

from sat_qkd_lab.run import build_parser, _run_calibration_fit


def test_calibration_card_has_model_fields(tmp_path: Path) -> None:
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(json.dumps([
        {"loss_db": 20.0, "qber_mean": 0.03},
    ]))

    parser = build_parser()
    args = parser.parse_args([
        "calibration-fit",
        "--telemetry", str(telemetry_path),
        "--grid-steps", "3",
        "--outdir", str(tmp_path),
    ])
    _run_calibration_fit(args)

    card_path = tmp_path / "reports" / "latest_calibration_card.json"
    card = json.loads(card_path.read_text())
    model = card["model_card"]
    assert "fit" in model
    assert "fim" in model
    assert "r2" in model["fit"]
    assert "param_uncertainty" in model["fit"]
    assert "cond" in model["fim"]
    assert "identifiable" in model["fim"]
    assert "identifiable" in model
    if model["identifiable"] is False:
        assert "warnings" in card
