from sat_qkd_lab.eb_observables import compute_observables


def test_chsh_violation_for_perfect_correlations():
    counts = {
        "AB": [[50, 0], [0, 50]],
        "ABp": [[50, 0], [0, 50]],
        "ApB": [[50, 0], [0, 50]],
        "ApBp": [[0, 50], [50, 0]],
    }
    obs = compute_observables(counts, n_pairs=100)
    assert obs.chsh_s == 4.0
    assert obs.visibility == 1.0
    assert obs.aborted is False


def test_chsh_not_violated_for_noise():
    counts = {
        "AB": [[25, 25], [25, 25]],
        "ABp": [[25, 25], [25, 25]],
        "ApB": [[25, 25], [25, 25]],
        "ApBp": [[25, 25], [25, 25]],
    }
    obs = compute_observables(counts, n_pairs=100, min_chsh_s=2.1)
    assert obs.chsh_s == 0.0
    assert obs.visibility == 0.0
    assert obs.aborted is True
