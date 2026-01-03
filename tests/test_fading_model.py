import numpy as np

from sat_qkd_lab.pass_model import FadingParams, PassModelParams, compute_pass_records, sample_fading_factors


def test_log_normal_fading_mean_close_to_one():
    rng = np.random.default_rng(7)
    samples = sample_fading_factors(20000, sigma_ln=0.4, rng=rng)
    mean = float(np.mean(samples))
    assert 0.97 <= mean <= 1.03


def test_pass_records_fading_ci():
    params = PassModelParams(pass_seconds=10.0, dt_seconds=1.0, rep_rate_hz=1e6)
    fading = FadingParams(enabled=True, sigma_ln=0.3, n_samples=30, seed=3)
    records, _ = compute_pass_records(params=params, fading=fading)
    first = records[0]
    assert "qber_ci_low" in first
    assert first["qber_ci_low"] <= first["qber_mean"] <= first["qber_ci_high"]
