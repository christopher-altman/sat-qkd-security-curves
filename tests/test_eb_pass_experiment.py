import json
from pathlib import Path

from sat_qkd_lab.eb_qkd import EBQKDParams, simulate_eb_qkd_expected
from sat_qkd_lab.finite_key import hoeffding_bound
from sat_qkd_lab.free_space_link import FreeSpaceLinkParams
from sat_qkd_lab.pass_model import elevation_to_loss_db, background_prob
from sat_qkd_lab.experiment import ExperimentParams, generate_schedule, run_experiment


def test_hoeffding_bound_monotonic():
    bound_small = hoeffding_bound(100, 0.02, 1e-6)
    bound_large = hoeffding_bound(1000, 0.02, 1e-6)
    assert bound_large <= bound_small


def test_eb_qkd_abort_on_high_qber():
    params = EBQKDParams(loss_db=20.0, flip_prob=0.2)
    result = simulate_eb_qkd_expected(n_pairs=50_000, params=params)
    assert result["aborted"] is True
    assert result["n_secret_est_finite"] == 0.0
    for key in ("qber_upper", "secret_fraction_finite", "n_secret_est_finite"):
        assert key in result


def test_pass_model_loss_and_background():
    link = FreeSpaceLinkParams()
    loss_low = elevation_to_loss_db(10.0, link)
    loss_high = elevation_to_loss_db(80.0, link)
    assert loss_high < loss_low

    p_bg = 1e-4
    night = background_prob(p_bg, link, "night")
    day = background_prob(p_bg, link, "day")
    assert day >= night


def test_experiment_blinding_schedule(tmp_path: Path):
    schedule_a = generate_schedule(10, seed=123)
    schedule_b = generate_schedule(10, seed=123)
    assert schedule_a == schedule_b

    params = ExperimentParams(seed=123, n_blocks=6, block_seconds=10.0, rep_rate_hz=1e6)
    output = run_experiment(
        params=params,
        metrics=["qber_mean", "headroom", "total_secret_bits"],
        outdir=tmp_path,
        finite_key=None,
        unblind=False,
    )
    assert output["analysis"]["group_labels_included"] is False
    for metric in output["analysis"]["metrics"].values():
        assert metric["control_mean"] is None
        assert metric["intervention_mean"] is None
        assert metric["delta_intervention_minus_control"] is None

    blinded_path = tmp_path / "reports" / "schedule_blinded.json"
    assert blinded_path.exists()
    assert not (tmp_path / "reports" / "schedule_unblinded.json").exists()

    blinded = json.loads(blinded_path.read_text())
    assert "label" not in json.dumps(blinded)
