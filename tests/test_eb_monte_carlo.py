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


def test_negative_n_pairs_raises_error():
    """Test that n_pairs <= 0 raises ValueError."""
    params = EBCoincidenceParams(
        loss_db_alice=10.0,
        loss_db_bob=10.0,
        flip_prob=0.01,
    )
    detector = DetectorParams(eta=0.2, p_bg=1e-4)
    
    # Test n_pairs = 0
    with pytest.raises(ValueError, match="n_pairs must be positive"):
        simulate_eb_coincidence_monte_carlo(
            n_pairs=0,
            params=params,
            detector_alice=detector,
            detector_bob=detector,
            seed=42,
        )
    
    # Test n_pairs < 0
    with pytest.raises(ValueError, match="n_pairs must be positive"):
        simulate_eb_coincidence_monte_carlo(
            n_pairs=-100,
            params=params,
            detector_alice=detector,
            detector_bob=detector,
            seed=42,
        )


def test_increasing_flip_prob_increases_qber():
    """Test that increasing flip_prob leads to increased qber_mean."""
    detector = DetectorParams(eta=0.3, p_bg=1e-5)
    flip_probs = [0.001, 0.01, 0.05, 0.10]
    qber_values = []
    
    for flip_prob in flip_probs:
        params = EBCoincidenceParams(
            loss_db_alice=5.0,
            loss_db_bob=5.0,
            flip_prob=flip_prob,
        )
        result = simulate_eb_coincidence_monte_carlo(
            n_pairs=200_000,
            params=params,
            detector_bob=detector,
            seed=42,
        )
        qber_values.append(result["qber_mean"])
    
    # Check that QBER increases with flip_prob
    for i in range(len(qber_values) - 1):
        assert qber_values[i] < qber_values[i + 1], (
            f"QBER did not increase: flip_prob[{i}]={flip_probs[i]} -> qber={qber_values[i]}, "
            f"flip_prob[{i+1}]={flip_probs[i+1]} -> qber={qber_values[i+1]}"
        )


def test_detector_efficiency_affects_coincidence_rate():
    """Test that higher detector efficiency increases coincidence_rate."""
    params = EBCoincidenceParams(
        loss_db_alice=8.0,
        loss_db_bob=8.0,
        flip_prob=0.01,
    )
    
    eta_values = [0.1, 0.2, 0.4, 0.6]
    coincidence_rates = []
    
    for eta in eta_values:
        detector = DetectorParams(eta=eta, p_bg=1e-5)
        result = simulate_eb_coincidence_monte_carlo(
            n_pairs=100_000,
            params=params,
            detector_alice=detector,
            detector_bob=detector,
            seed=123,
        )
        coincidence_rates.append(result["coincidence_rate"])
    
    # Check that coincidence rate increases with detector efficiency
    for i in range(len(coincidence_rates) - 1):
        assert coincidence_rates[i] < coincidence_rates[i + 1], (
            f"Coincidence rate did not increase: eta[{i}]={eta_values[i]} -> rate={coincidence_rates[i]}, "
            f"eta[{i+1}]={eta_values[i+1]} -> rate={coincidence_rates[i+1]}"
        )


def test_background_probability_affects_qber_and_coincidence():
    """Test that increased background probability affects qber_mean and coincidence_rate."""
    params = EBCoincidenceParams(
        loss_db_alice=10.0,
        loss_db_bob=10.0,
        flip_prob=0.005,
    )
    
    p_bg_values = [1e-6, 1e-5, 1e-4, 1e-3]
    qber_values = []
    coincidence_rates = []
    
    for p_bg in p_bg_values:
        detector = DetectorParams(eta=0.25, p_bg=p_bg)
        result = simulate_eb_coincidence_monte_carlo(
            n_pairs=150_000,
            params=params,
            detector_alice=detector,
            detector_bob=detector,
            seed=456,
        )
        qber_values.append(result["qber_mean"])
        coincidence_rates.append(result["coincidence_rate"])
    
    # QBER should increase with background (more background-induced errors)
    assert qber_values[-1] > qber_values[0], (
        f"QBER did not increase with background: p_bg={p_bg_values[0]} -> qber={qber_values[0]}, "
        f"p_bg={p_bg_values[-1]} -> qber={qber_values[-1]}"
    )
    
    # Coincidence rate should increase with background (more accidental coincidences)
    assert coincidence_rates[-1] > coincidence_rates[0], (
        f"Coincidence rate did not increase with background: "
        f"p_bg={p_bg_values[0]} -> rate={coincidence_rates[0]}, "
        f"p_bg={p_bg_values[-1]} -> rate={coincidence_rates[-1]}"
    )


def test_qber_abort_threshold_aborts_protocol():
    """Test that qber_abort_threshold correctly aborts protocol and sets n_secret_est to 0."""
    detector = DetectorParams(eta=0.2, p_bg=5e-4)
    
    # Set high flip_prob to ensure QBER exceeds threshold
    params_abort = EBCoincidenceParams(
        loss_db_alice=15.0,
        loss_db_bob=15.0,
        flip_prob=0.15,
        qber_abort_threshold=0.08,  # Low threshold to trigger abort
    )
    
    result = simulate_eb_coincidence_monte_carlo(
        n_pairs=100_000,
        params=params_abort,
        detector_alice=detector,
        detector_bob=detector,
        seed=789,
    )
    
    # Check that protocol is aborted when QBER exceeds threshold
    if not np.isnan(result["qber_mean"]):
        if result["qber_mean"] > params_abort.qber_abort_threshold:
            assert result["aborted"], (
                f"Protocol should abort when QBER {result['qber_mean']} > threshold {params_abort.qber_abort_threshold}"
            )
            assert result["n_secret_est"] == 0.0, (
                f"n_secret_est should be 0 when aborted, got {result['n_secret_est']}"
            )
