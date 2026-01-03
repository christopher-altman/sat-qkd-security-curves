import numpy as np

from sat_qkd_lab.fading_samples import sample_fading_transmittance


def test_fading_samples_mean_variance():
    sigma = 0.4
    samples = sample_fading_transmittance(sigma_ln=sigma, n_samples=50000, seed=1)
    mean = float(np.mean(samples))
    var = float(np.var(samples, ddof=1))
    expected_var = np.exp(sigma ** 2) * (np.exp(sigma ** 2) - 1.0)
    assert abs(mean - 1.0) < 0.02
    assert abs(var - expected_var) / expected_var < 0.15
