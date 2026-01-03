import numpy as np

from sat_qkd_lab.polarization_drift import (
    CompensationParams,
    PolarizationDriftParams,
    adjust_qber_for_angle,
    compensate_polarization_drift,
    simulate_polarization_drift,
)


def test_drift_increases_basis_errors():
    time_s = np.linspace(0.0, 10.0, 11)
    params = PolarizationDriftParams(enabled=True, sigma_deg=5.0, seed=1)
    angles = simulate_polarization_drift(time_s, params)
    base_qber = 0.02
    qbers = [adjust_qber_for_angle(base_qber, angle) for angle in angles]
    assert float(np.mean(qbers)) >= base_qber


def test_compensation_reduces_errors_with_lag():
    time_s = np.linspace(0.0, 10.0, 11)
    params = PolarizationDriftParams(enabled=True, sigma_deg=8.0, seed=2)
    angles = simulate_polarization_drift(time_s, params)
    comp = CompensationParams(enabled=True, lag_seconds=3.0)
    residual, _ = compensate_polarization_drift(time_s, angles, comp)
    base_qber = 0.02
    qbers_raw = [adjust_qber_for_angle(base_qber, angle) for angle in angles]
    qbers_comp = [adjust_qber_for_angle(base_qber, angle) for angle in residual]
    assert float(np.mean(qbers_comp)) <= float(np.mean(qbers_raw))
    assert float(np.mean(qbers_comp)) >= base_qber
