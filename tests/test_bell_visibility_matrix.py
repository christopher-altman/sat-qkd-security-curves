from sat_qkd_lab.experiment import ExperimentParams, run_experiment
from sat_qkd_lab.eb_observables import correlation_from_counts


def _get_bell(output):
    return output["block_results"][0]["bell"]


def test_bell_mode_violates_chsh_for_perfect(tmp_path):
    params = ExperimentParams(
        seed=1,
        n_blocks=1,
        block_seconds=10.0,
        rep_rate_hz=1e6,
        pass_seconds=10.0,
        base_qber=0.0,
        qber_jitter=0.0,
    )
    output = run_experiment(
        params=params,
        metrics=["qber_mean"],
        outdir=tmp_path,
        finite_key=None,
        bell_mode=True,
        unblind=False,
    )
    bell = _get_bell(output)
    assert bell["chsh_s"] > 2.0
    matrix_z = bell["coincidence_matrices"]["Z"]
    assert matrix_z[0][1] == 0
    assert matrix_z[1][0] == 0


def test_bell_mode_noisy_does_not_violate(tmp_path):
    params = ExperimentParams(
        seed=2,
        n_blocks=1,
        block_seconds=10.0,
        rep_rate_hz=1e6,
        pass_seconds=10.0,
        base_qber=0.3,
        qber_jitter=0.0,
    )
    output = run_experiment(
        params=params,
        metrics=["qber_mean"],
        outdir=tmp_path,
        finite_key=None,
        bell_mode=True,
        unblind=False,
    )
    bell = _get_bell(output)
    assert bell["chsh_s"] < 2.0
    matrix_z = bell["coincidence_matrices"]["Z"]
    corr = correlation_from_counts(matrix_z)
    assert corr == bell["correlations"]["AB"]
