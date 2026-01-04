"""
Tests for EB-QKD coincidence Monte Carlo simulation.

These tests enforce numerical sanity bounds (Opus gate C0 contract).
"""
import numpy as np
import pytest
from sat_qkd_lab.eb_coincidence import (
    EBCoincidenceParams,
    simulate_eb_coincidence_monte_carlo,
    sweep_eb_coincidence_loss,
)
from sat_qkd_lab.detector import DetectorParams
from sat_qkd_lab.finite_key import FiniteKeyParams


# Contract constants (from gate C0)
QBER_MIN = 0.0
QBER_MAX = 0.5
SECRET_FRACTION_MIN = 0.0
SECRET_FRACTION_MAX = 1.0


def test_qber_in_valid_range():
    """Test QBER-equivalent is within [0, 0.5]."""
    params = EBCoincidenceParams(
        loss_db_alice=10.0,
        loss_db_bob=10.0,
        flip_prob=0.01,
    )
    detector = DetectorParams(eta=0.2, p_bg=1e-4)
    
    result = simulate_eb_coincidence_monte_carlo(
        n_pairs=100_000,
        params=params,
        detector_alice=detector,
        detector_bob=detector,
        seed=42,
    )
    
    qber = result["qber_mean"]
    assert QBER_MIN <= qber <= QBER_MAX, (
        f"QBER {qber} violates contract bounds [{QBER_MIN}, {QBER_MAX}]"
    )


def test_qber_at_high_loss():
    """Test QBER at very high loss (should approach 0.5 or be nan)."""
    params = EBCoincidenceParams(
        loss_db_alice=30.0,
        loss_db_bob=30.0,
        flip_prob=0.01,
    )
    detector = DetectorParams(eta=0.2, p_bg=1e-4)
    
    result = simulate_eb_coincidence_monte_carlo(
        n_pairs=100_000,
        params=params,
        detector_alice=detector,
        detector_bob=detector,
        seed=42,
    )
    
    qber = result["qber_mean"]
    # At very high loss, QBER is dominated by background -> ~0.5 or nan if no coincidences
    assert np.isnan(qber) or (0.4 <= qber <= QBER_MAX), (
        f"High-loss QBER {qber} outside expected range"
    )


def test_coincidence_rate_decreases_with_loss():
    """Test coincidence rate decreases monotonically with increasing loss."""
    loss_values = np.linspace(5.0, 20.0, 5)
    detector = DetectorParams(eta=0.2, p_bg=1e-4)
    
    results = sweep_eb_coincidence_loss(
        loss_db_values=loss_values,
        n_pairs=50_000,
        flip_prob=0.005,
        detector_alice=detector,
        detector_bob=detector,
        seed=123,
    )
    
    rates = [r["coincidence_rate"] for r in results]
    
    # Check monotonic decrease (allow small noise wiggles)
    # We'll check that overall trend is decreasing by comparing first and last
    assert rates[0] > rates[-1], (
        f"Coincidence rate not decreasing: first={rates[0]}, last={rates[-1]}"
    )
    
    # Check that on average, each step is not increasing by >10% (tolerance for noise)
    violations = 0
    for i in range(len(rates) - 1):
        if rates[i+1] > rates[i] * 1.1:  # Allow 10% increase for noise
            violations += 1
    
    # Allow at most 1 violation in monotonicity due to noise
    assert violations <= 1, (
        f"Too many violations in monotonic decrease: {violations}, rates={rates}"
    )


def test_secret_fraction_positive_at_low_loss():
    """Test secret fraction > 0 at low loss (toy estimate)."""
    params = EBCoincidenceParams(
        loss_db_alice=2.0,
        loss_db_bob=2.0,
        flip_prob=0.001,
    )
    detector = DetectorParams(eta=0.3, p_bg=1e-5)
    
    # Use larger n_pairs to ensure some secret bits
    result = simulate_eb_coincidence_monte_carlo(
        n_pairs=500_000,
        params=params,
        detector_alice=detector,
        detector_bob=detector,
        seed=42,
    )
    
    sf = result["secret_fraction_estimate"]
    sf_asym = result["secret_fraction_asymptotic"]
    
    # At least asymptotic should be positive at low loss
    assert sf_asym > 0.0, (
        f"Secret fraction asymptotic {sf_asym} should be > 0 at low loss"
    )


def test_secret_fraction_zero_at_high_loss():
    """Test secret fraction → 0 at very high loss."""
    params = EBCoincidenceParams(
        loss_db_alice=25.0,
        loss_db_bob=25.0,
        flip_prob=0.01,
    )
    detector = DetectorParams(eta=0.2, p_bg=1e-4)
    
    result = simulate_eb_coincidence_monte_carlo(
        n_pairs=100_000,
        params=params,
        detector_alice=detector,
        detector_bob=detector,
        seed=42,
    )
    
    sf = result["secret_fraction_estimate"]
    
    # At high loss, should abort or have zero secret fraction
    assert sf == 0.0 or result["aborted"], (
        f"Secret fraction {sf} should be 0 or aborted at high loss"
    )


def test_secret_fraction_in_valid_range():
    """Test secret fraction estimate is within [0, 1]."""
    params = EBCoincidenceParams(
        loss_db_alice=8.0,
        loss_db_bob=8.0,
        flip_prob=0.005,
    )
    detector = DetectorParams(eta=0.25, p_bg=1e-4)
    
    result = simulate_eb_coincidence_monte_carlo(
        n_pairs=200_000,
        params=params,
        detector_alice=detector,
        detector_bob=detector,
        seed=99,
    )
    
    sf = result["secret_fraction_estimate"]
    sf_asym = result["secret_fraction_asymptotic"]
    
    assert SECRET_FRACTION_MIN <= sf <= SECRET_FRACTION_MAX, (
        f"Secret fraction estimate {sf} violates contract bounds"
    )
    assert SECRET_FRACTION_MIN <= sf_asym <= SECRET_FRACTION_MAX, (
        f"Secret fraction asymptotic {sf_asym} violates contract bounds"
    )


def test_output_structure_compliance():
    """Test output structure matches expected schema."""
    params = EBCoincidenceParams(
        loss_db_alice=10.0,
        loss_db_bob=10.0,
        flip_prob=0.005,
    )
    detector = DetectorParams(eta=0.2, p_bg=1e-4)
    
    result = simulate_eb_coincidence_monte_carlo(
        n_pairs=50_000,
        params=params,
        detector_alice=detector,
        detector_bob=detector,
        seed=0,
    )
    
    # Check required fields
    required_fields = [
        "n_pairs",
        "n_coincidences",
        "n_signal_coincidence",
        "n_sifted",
        "n_errors",
        "coincidence_rate",
        "qber_mean",
        "qber_upper",
        "secret_fraction_estimate",
        "secret_fraction_asymptotic",
        "n_secret_est",
        "key_rate_per_pair",
        "aborted",
        "finite_key",
        "meta",
    ]
    
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    # Check types
    assert isinstance(result["n_pairs"], int)
    assert isinstance(result["n_coincidences"], int)
    assert isinstance(result["coincidence_rate"], float)
    assert isinstance(result["qber_mean"], float)
    assert isinstance(result["secret_fraction_estimate"], float)
    assert isinstance(result["aborted"], bool)
    assert isinstance(result["finite_key"], dict)
    assert isinstance(result["meta"], dict)


def test_finite_key_parameters():
    """Test finite-key analysis integration."""
    params = EBCoincidenceParams(
        loss_db_alice=10.0,
        loss_db_bob=10.0,
        flip_prob=0.005,
    )
    detector = DetectorParams(eta=0.2, p_bg=1e-4)
    fk_params = FiniteKeyParams(
        eps_pe=1e-10,
        eps_sec=1e-10,
        eps_cor=1e-15,
        ec_efficiency=1.16,
        pe_frac=0.5,
    )
    
    result = simulate_eb_coincidence_monte_carlo(
        n_pairs=100_000,
        params=params,
        detector_alice=detector,
        detector_bob=detector,
        finite_key=fk_params,
        seed=42,
    )
    
    # Check finite-key fields
    assert "finite_key" in result
    fk = result["finite_key"]
    assert "bound" in fk
    assert "status" in fk
    assert fk["status"] in ["secure", "insecure"]


def test_sweep_output_consistency():
    """Test sweep function returns consistent results."""
    loss_values = np.array([10.0, 15.0, 20.0])
    detector = DetectorParams(eta=0.2, p_bg=1e-4)
    
    results = sweep_eb_coincidence_loss(
        loss_db_values=loss_values,
        n_pairs=30_000,
        flip_prob=0.005,
        detector_alice=detector,
        detector_bob=detector,
        seed=42,
    )
    
    assert len(results) == len(loss_values), "Number of results != number of loss values"
    
    for i, (loss_db, result) in enumerate(zip(loss_values, results)):
        assert "loss_db" in result
        assert abs(result["loss_db"] - loss_db) < 1e-9
        assert result["meta"]["loss_db_total"] == loss_db


def test_background_dominance_at_high_loss():
    """Test that background dominates at high loss."""
    params = EBCoincidenceParams(
        loss_db_alice=30.0,
        loss_db_bob=30.0,
        flip_prob=0.01,
    )
    # High background
    detector = DetectorParams(eta=0.2, p_bg=1e-3)
    
    result = simulate_eb_coincidence_monte_carlo(
        n_pairs=100_000,
        params=params,
        detector_alice=detector,
        detector_bob=detector,
        seed=42,
    )
    
    # At high loss with high background, most coincidences should be accidental
    if result["n_coincidences"] > 0:
        signal_fraction = result["n_signal_coincidence"] / result["n_coincidences"]
        # Background should dominate
        assert signal_fraction < 0.1, (
            f"Signal fraction {signal_fraction} unexpectedly high at high loss"
        )


def test_zero_flip_prob():
    """Test with zero flip probability."""
    params = EBCoincidenceParams(
        loss_db_alice=5.0,
        loss_db_bob=5.0,
        flip_prob=0.0,
    )
    detector = DetectorParams(eta=0.3, p_bg=1e-5)
    
    result = simulate_eb_coincidence_monte_carlo(
        n_pairs=100_000,
        params=params,
        detector_alice=detector,
        detector_bob=detector,
        seed=42,
    )
    
    # With no intrinsic errors and low loss, QBER should be dominated by background
    # which is very small, so QBER should be very low
    qber = result["qber_mean"]
    if not np.isnan(qber):
        assert qber < 0.02, f"QBER {qber} too high with zero flip_prob"
