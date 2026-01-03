import numpy as np

from sat_qkd_lab.ou_fading import simulate_ou_transmittance, compute_outage_clusters


def test_ou_fading_mean_reversion():
    times, values = simulate_ou_transmittance(
        duration_s=50.0,
        dt_s=0.1,
        mu=0.7,
        theta=0.4,
        sigma=0.02,
        seed=1,
        t0=0.2,
    )
    assert times.size == values.size
    assert abs(float(np.mean(values)) - 0.7) < 0.1


def test_outage_clusters_detected():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    values = np.array([0.5, 0.1, 0.1, 0.6, 0.2], dtype=float)
    stats = compute_outage_clusters(times, values, threshold=0.3)
    assert stats.count == 2
    assert stats.durations_s[0] == 2.0
