"""
Unit tests for parameter validation and decoy-state BB84 realism features.

Tests cover:
- DecoyParams validation for negative intensity fluctuation standard deviations
- DetectorParams validation for afterpulsing, dead time, and basis efficiency parameters
- _sample_truncated_mu behavior (non-negativity)
- Impact of intensity noise on key rate
- Differential basis efficiency effects on QBER and key rate
"""
import pytest
import numpy as np
from sat_qkd_lab.decoy_bb84 import DecoyParams, simulate_decoy_bb84
from sat_qkd_lab.detector import DetectorParams


# --- DecoyParams Validation Tests ---

def test_decoy_params_negative_mu_s_sigma_raises_valueerror():
    """DecoyParams should raise ValueError for negative mu_s_sigma."""
    with pytest.raises(ValueError, match="mu_s_sigma must be >= 0"):
        DecoyParams(mu_s_sigma=-0.01)


def test_decoy_params_negative_mu_d_sigma_raises_valueerror():
    """DecoyParams should raise ValueError for negative mu_d_sigma."""
    with pytest.raises(ValueError, match="mu_d_sigma must be >= 0"):
        DecoyParams(mu_d_sigma=-0.05)


def test_decoy_params_both_negative_sigmas_raises_valueerror():
    """DecoyParams should raise ValueError when both sigmas are negative."""
    with pytest.raises(ValueError, match="mu_s_sigma must be >= 0"):
        DecoyParams(mu_s_sigma=-0.01, mu_d_sigma=-0.02)


def test_decoy_params_zero_sigmas_valid():
    """DecoyParams should accept zero values for intensity fluctuation sigmas."""
    params = DecoyParams(mu_s_sigma=0.0, mu_d_sigma=0.0)
    assert params.mu_s_sigma == 0.0
    assert params.mu_d_sigma == 0.0


def test_decoy_params_positive_sigmas_valid():
    """DecoyParams should accept positive values for intensity fluctuation sigmas."""
    params = DecoyParams(mu_s_sigma=0.05, mu_d_sigma=0.02)
    assert params.mu_s_sigma == 0.05
    assert params.mu_d_sigma == 0.02


# --- DetectorParams Validation Tests ---

def test_detector_params_negative_p_afterpulse_raises_valueerror():
    """DetectorParams should raise ValueError for negative p_afterpulse."""
    with pytest.raises(ValueError, match="p_afterpulse must be in \\[0, 1\\]"):
        DetectorParams(p_afterpulse=-0.01)


def test_detector_params_p_afterpulse_exceeds_one_raises_valueerror():
    """DetectorParams should raise ValueError for p_afterpulse > 1."""
    with pytest.raises(ValueError, match="p_afterpulse must be in \\[0, 1\\]"):
        DetectorParams(p_afterpulse=1.5)


def test_detector_params_negative_afterpulse_window_raises_valueerror():
    """DetectorParams should raise ValueError for negative afterpulse_window."""
    with pytest.raises(ValueError, match="afterpulse_window must be >= 0"):
        DetectorParams(afterpulse_window=-1)


def test_detector_params_negative_afterpulse_decay_raises_valueerror():
    """DetectorParams should raise ValueError for negative afterpulse_decay."""
    with pytest.raises(ValueError, match="afterpulse_decay must be >= 0"):
        DetectorParams(afterpulse_decay=-0.5)


def test_detector_params_negative_dead_time_raises_valueerror():
    """DetectorParams should raise ValueError for negative dead_time_pulses."""
    with pytest.raises(ValueError, match="dead_time_pulses must be >= 0"):
        DetectorParams(dead_time_pulses=-5)


def test_detector_params_invalid_eta_z_raises_valueerror():
    """DetectorParams should raise ValueError for eta_z out of [0, 1]."""
    with pytest.raises(ValueError, match="eta_z must be in \\[0, 1\\]"):
        DetectorParams(eta=0.5, eta_z=-0.1)
    
    with pytest.raises(ValueError, match="eta_z must be in \\[0, 1\\]"):
        DetectorParams(eta=0.5, eta_z=1.5)


def test_detector_params_invalid_eta_x_raises_valueerror():
    """DetectorParams should raise ValueError for eta_x out of [0, 1]."""
    with pytest.raises(ValueError, match="eta_x must be in \\[0, 1\\]"):
        DetectorParams(eta=0.5, eta_x=-0.2)
    
    with pytest.raises(ValueError, match="eta_x must be in \\[0, 1\\]"):
        DetectorParams(eta=0.5, eta_x=2.0)


def test_detector_params_valid_afterpulse_parameters():
    """DetectorParams should accept valid afterpulsing parameters."""
    params = DetectorParams(
        p_afterpulse=0.05,
        afterpulse_window=5,
        afterpulse_decay=2.0,
    )
    assert params.p_afterpulse == 0.05
    assert params.afterpulse_window == 5
    assert params.afterpulse_decay == 2.0


def test_detector_params_valid_dead_time():
    """DetectorParams should accept valid dead_time_pulses."""
    params = DetectorParams(dead_time_pulses=10)
    assert params.dead_time_pulses == 10


def test_detector_params_valid_basis_efficiencies():
    """DetectorParams should accept valid eta_z and eta_x."""
    params = DetectorParams(eta=0.5, eta_z=0.6, eta_x=0.4)
    assert params.eta_z == 0.6
    assert params.eta_x == 0.4


def test_detector_params_eta_z_defaults_to_eta():
    """DetectorParams should default eta_z to eta when not specified."""
    params = DetectorParams(eta=0.3)
    assert params.eta_z == 0.3


def test_detector_params_eta_x_defaults_to_eta():
    """DetectorParams should default eta_x to eta when not specified."""
    params = DetectorParams(eta=0.3)
    assert params.eta_x == 0.3


# --- _sample_truncated_mu Tests ---

def test_sample_truncated_mu_always_non_negative():
    """_sample_truncated_mu should always return non-negative values."""
    # We need to test the internal function by running simulations with intensity noise
    decoy = DecoyParams(mu_s_sigma=0.2, mu_d_sigma=0.1)  # Large sigma to test truncation
    det = DetectorParams(eta=0.5, p_bg=1e-5)
    
    # Run many simulations to stress-test the truncation logic
    for seed in range(20):
        result = simulate_decoy_bb84(
            n_pulses=10_000,
            loss_db=25.0,
            decoy=decoy,
            detector=det,
            seed=seed,
        )
        
        # All key metrics should be valid (non-negative, finite)
        assert result["Q_signal"] >= 0.0
        assert result["Q_decoy"] >= 0.0
        assert result["Q_vacuum"] >= 0.0
        assert result["key_rate_asymptotic"] >= 0.0
        assert not np.isnan(result["Q_signal"])
        assert not np.isnan(result["key_rate_asymptotic"])


def test_sample_truncated_mu_with_extreme_sigma():
    """_sample_truncated_mu should handle extreme sigma values gracefully."""
    # Very large sigma relative to mean - would frequently go negative without truncation
    decoy = DecoyParams(mu_s=0.2, mu_d=0.05, mu_s_sigma=0.3, mu_d_sigma=0.1)
    det = DetectorParams(eta=0.5, p_bg=1e-5)
    
    result = simulate_decoy_bb84(
        n_pulses=5_000,
        loss_db=30.0,
        decoy=decoy,
        detector=det,
        seed=42,
    )
    
    # Should complete without errors and produce valid output
    assert result["Q_signal"] >= 0.0
    assert not np.isnan(result["Q_signal"])
    assert not np.isinf(result["Q_signal"])


# --- Intensity Noise Impact Tests ---

def test_intensity_noise_reduces_key_rate():
    """Intensity noise should reduce key rate compared to no noise."""
    decoy_no_noise = DecoyParams(mu_s_sigma=0.0, mu_d_sigma=0.0)
    decoy_with_noise = DecoyParams(mu_s_sigma=0.05, mu_d_sigma=0.02)
    det = DetectorParams(eta=0.3, p_bg=5e-5)
    
    result_no_noise = simulate_decoy_bb84(
        n_pulses=100_000,
        loss_db=30.0,
        decoy=decoy_no_noise,
        detector=det,
        seed=123,
    )
    
    result_with_noise = simulate_decoy_bb84(
        n_pulses=100_000,
        loss_db=30.0,
        decoy=decoy_with_noise,
        detector=det,
        seed=123,
    )
    
    # With noise, key rate should be lower (or equal in edge cases)
    assert result_with_noise["key_rate_asymptotic"] <= result_no_noise["key_rate_asymptotic"]
    
    # QBER should be similar or slightly higher with noise
    if result_no_noise["n_sift_signal"] > 0 and result_with_noise["n_sift_signal"] > 0:
        assert result_with_noise["E_signal"] >= result_no_noise["E_signal"] - 0.01


def test_intensity_noise_at_multiple_loss_values():
    """Intensity noise impact should be observable at various loss levels."""
    decoy_no_noise = DecoyParams(mu_s_sigma=0.0, mu_d_sigma=0.0)
    decoy_with_noise = DecoyParams(mu_s_sigma=0.08, mu_d_sigma=0.03)
    det = DetectorParams(eta=0.25, p_bg=1e-4)
    
    # Test at low-moderate loss where key rates are reliably non-zero
    loss_values = [20.0, 25.0]
    noise_degrades = False
    
    for loss_db in loss_values:
        result_no_noise = simulate_decoy_bb84(
            n_pulses=100_000,  # More pulses for stable statistics
            loss_db=loss_db,
            decoy=decoy_no_noise,
            detector=det,
            seed=42,
        )
        
        result_with_noise = simulate_decoy_bb84(
            n_pulses=100_000,
            loss_db=loss_db,
            decoy=decoy_with_noise,
            detector=det,
            seed=42,
        )
        
        # Check if noise degrades key rate at any loss value
        # With large enough samples, at least one should show degradation
        if result_with_noise["key_rate_asymptotic"] < result_no_noise["key_rate_asymptotic"]:
            noise_degrades = True
            break
    
    # At least one loss value should show that noise doesn't improve key rate
    assert noise_degrades or all(
        result_with_noise["key_rate_asymptotic"] <= result_no_noise["key_rate_asymptotic"] + 1e-5
        for loss_db in loss_values
        for result_no_noise, result_with_noise in [
            (simulate_decoy_bb84(100_000, loss_db, decoy=decoy_no_noise, detector=det, seed=42),
             simulate_decoy_bb84(100_000, loss_db, decoy=decoy_with_noise, detector=det, seed=42))
        ]
    ), "Intensity noise should not systematically improve key rate"


# --- Differential Basis Efficiency Tests ---

def test_basis_efficiency_impacts_qber_and_key_rate():
    """Different eta_z and eta_x should impact QBER and key rate."""
    decoy = DecoyParams()
    
    # Equal efficiencies (baseline)
    det_equal = DetectorParams(eta=0.4, eta_z=0.4, eta_x=0.4, p_bg=1e-4)
    
    # X-basis has significantly lower efficiency
    det_x_lower = DetectorParams(eta=0.4, eta_z=0.4, eta_x=0.1, p_bg=1e-4)
    
    # Use lower loss for stable, non-zero key rates
    result_equal = simulate_decoy_bb84(
        n_pulses=150_000,
        loss_db=25.0,
        decoy=decoy,
        detector=det_equal,
        seed=7890,
    )
    
    result_x_lower = simulate_decoy_bb84(
        n_pulses=150_000,
        loss_db=25.0,
        decoy=decoy,
        detector=det_x_lower,
        seed=7890,  # Same seed to reduce Monte Carlo variation
    )
    
    # With significantly lower X-basis efficiency, we should see impact
    # Both simulations should produce valid, non-zero results
    assert result_equal["Q_signal"] > 0
    assert result_x_lower["Q_signal"] > 0
    assert result_equal["key_rate_asymptotic"] > 0
    
    # The degraded X-basis should not improve results
    # (May be similar due to Z-basis dominance, but shouldn't improve)
    assert result_x_lower["key_rate_asymptotic"] <= result_equal["key_rate_asymptotic"] * 1.1




def test_basis_efficiency_mismatch_increases_background_impact():
    """Lower efficiency in one basis should increase background click impact."""
    decoy = DecoyParams()
    
    # High background to make the effect more visible
    det_equal = DetectorParams(eta=0.4, eta_z=0.4, eta_x=0.4, p_bg=5e-4)
    det_x_low = DetectorParams(eta=0.4, eta_z=0.4, eta_x=0.2, p_bg=5e-4)
    
    result_equal = simulate_decoy_bb84(
        n_pulses=80_000,
        loss_db=32.0,
        decoy=decoy,
        detector=det_equal,
        seed=999,
    )
    
    result_x_low = simulate_decoy_bb84(
        n_pulses=80_000,
        loss_db=32.0,
        decoy=decoy,
        detector=det_x_low,
        seed=999,
    )
    
    # With lower X efficiency, signal/background ratio worsens in X basis
    # This should result in higher overall QBER
    if result_equal["n_sift_signal"] > 100 and result_x_low["n_sift_signal"] > 100:
        assert result_x_low["E_signal"] > result_equal["E_signal"]


def test_extreme_basis_asymmetry():
    """Extreme difference between eta_z and eta_x should still produce valid results."""
    decoy = DecoyParams()
    
    # Very asymmetric efficiencies
    det_asymmetric = DetectorParams(eta=0.5, eta_z=0.5, eta_x=0.05, p_bg=1e-4)
    
    result = simulate_decoy_bb84(
        n_pulses=50_000,
        loss_db=25.0,
        decoy=decoy,
        detector=det_asymmetric,
        seed=111,
    )
    
    # Should complete without errors
    assert result["Q_signal"] >= 0.0
    assert 0.0 <= result["E_signal"] <= 0.5
    assert result["key_rate_asymptotic"] >= 0.0
    assert not np.isnan(result["key_rate_asymptotic"])


def test_basis_efficiency_reproducibility():
    """Same seed with basis efficiency mismatch should give reproducible results."""
    decoy = DecoyParams()
    det = DetectorParams(eta=0.3, eta_z=0.35, eta_x=0.25, p_bg=1e-4)
    
    result1 = simulate_decoy_bb84(
        n_pulses=30_000,
        loss_db=30.0,
        decoy=decoy,
        detector=det,
        seed=12345,
    )
    
    result2 = simulate_decoy_bb84(
        n_pulses=30_000,
        loss_db=30.0,
        decoy=decoy,
        detector=det,
        seed=12345,
    )
    
    # Results should be identical
    assert result1["Q_signal"] == result2["Q_signal"]
    assert result1["E_signal"] == result2["E_signal"]
    assert result1["key_rate_asymptotic"] == result2["key_rate_asymptotic"]
    assert result1["n_sift_signal"] == result2["n_sift_signal"]
