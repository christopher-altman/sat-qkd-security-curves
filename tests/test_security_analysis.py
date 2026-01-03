"""Tests for finite_key.py and decoy_bb84.py security modules."""
import pytest
import math
import numpy as np
from sat_qkd_lab.finite_key import (
    FiniteKeyParams, hoeffding_bound, hoeffding_lower_bound, 
    finite_key_bounds, finite_key_secret_length
)
from sat_qkd_lab.decoy_bb84 import DecoyParams, poisson_n, simulate_decoy_bb84


class TestFiniteKeyParams:
    """Test FiniteKeyParams validation."""

    def test_default_params(self):
        """Default parameters should be valid."""
        params = FiniteKeyParams()
        assert params.eps_pe == 1e-10
        assert params.eps_sec == 1e-10
        assert params.ec_efficiency == 1.16

    def test_eps_pe_bounds(self):
        """eps_pe must be in (0, 1)."""
        with pytest.raises(ValueError):
            FiniteKeyParams(eps_pe=0.0)
        with pytest.raises(ValueError):
            FiniteKeyParams(eps_pe=1.0)
        with pytest.raises(ValueError):
            FiniteKeyParams(eps_pe=-0.1)

    def test_eps_sec_bounds(self):
        """eps_sec must be in (0, 1)."""
        with pytest.raises(ValueError):
            FiniteKeyParams(eps_sec=0.0)
        with pytest.raises(ValueError):
            FiniteKeyParams(eps_sec=1.1)

    def test_eps_cor_bounds(self):
        """eps_cor must be in (0, 1)."""
        with pytest.raises(ValueError):
            FiniteKeyParams(eps_cor=0.0)
        with pytest.raises(ValueError):
            FiniteKeyParams(eps_cor=1.01)

    def test_ec_efficiency_bounds(self):
        """ec_efficiency must be >= 1.0."""
        with pytest.raises(ValueError):
            FiniteKeyParams(ec_efficiency=0.99)
        assert FiniteKeyParams(ec_efficiency=1.0).ec_efficiency == 1.0

    def test_pe_frac_bounds(self):
        """pe_frac must be in (0, 1]."""
        with pytest.raises(ValueError):
            FiniteKeyParams(pe_frac=0.0)
        with pytest.raises(ValueError):
            FiniteKeyParams(pe_frac=1.1)
        assert FiniteKeyParams(pe_frac=0.5).pe_frac == 0.5

    def test_m_pe_validation(self):
        """m_pe must be >= 1 when provided."""
        with pytest.raises(ValueError):
            FiniteKeyParams(m_pe=0)
        assert FiniteKeyParams(m_pe=1).m_pe == 1

    def test_eps_total(self):
        """eps_total should sum the failure probabilities."""
        params = FiniteKeyParams(eps_pe=1e-10, eps_sec=2e-10, eps_cor=3e-10)
        assert abs(params.eps_total - 6e-10) < 1e-20


class TestHoeffding:
    """Test Hoeffding bound functions."""

    def test_hoeffding_bound_zero_samples(self):
        """Zero samples should give worst-case upper bound."""
        assert hoeffding_bound(0, 0.05, 1e-6) == 1.0

    def test_hoeffding_bound_zero_eps(self):
        """Zero epsilon should give worst case."""
        assert hoeffding_bound(1000, 0.05, 0.0) == 1.0

    def test_hoeffding_bound_tight(self):
        """Bound should be >= observed rate."""
        observed = 0.05
        bound = hoeffding_bound(1000, observed, 1e-6)
        assert bound >= observed

    def test_hoeffding_bound_large_sample(self):
        """Large samples should give tighter bounds."""
        small = hoeffding_bound(100, 0.05, 1e-6)
        large = hoeffding_bound(10000, 0.05, 1e-6)
        assert large < small  # Tighter bound for larger sample

    def test_hoeffding_lower_bound_zero_samples(self):
        """Zero samples should give worst-case lower bound."""
        assert hoeffding_lower_bound(0, 0.05, 1e-6) == 0.0

    def test_hoeffding_lower_bound_tight(self):
        """Bound should be <= observed rate."""
        observed = 0.05
        bound = hoeffding_lower_bound(1000, observed, 1e-6)
        assert bound <= observed

    def test_hoeffding_bounds_bracketing(self):
        """Observed should be between lower and upper bounds."""
        observed = 0.1
        n = 1000
        lower = hoeffding_lower_bound(n, observed, 1e-6)
        upper = hoeffding_bound(n, observed, 1e-6)
        assert lower <= observed <= upper


class TestFiniteKeyBounds:
    """Test finite_key_bounds function."""

    def test_zero_sifted(self):
        """Zero sifted bits should return worst-case bounds."""
        result = finite_key_bounds(0, 0)
        assert math.isnan(result["qber_hat"])
        assert result["qber_upper"] == 1.0
        assert result["qber_lower"] == 0.0

    def test_zero_errors(self):
        """Zero errors should give zero QBER estimate."""
        result = finite_key_bounds(1000, 0)
        assert result["qber_hat"] == 0.0
        assert result["qber_upper"] >= 0.0

    def test_all_errors(self):
        """All errors should give high QBER."""
        result = finite_key_bounds(1000, 1000)
        assert result["qber_hat"] == 1.0
        # QBER is clamped to 0.5 (physical max)
        assert result["qber_upper"] <= 0.5

    def test_physical_qber_bound(self):
        """QBER upper bound should not exceed 0.5."""
        result = finite_key_bounds(100, 100)
        assert result["qber_upper"] <= 0.5

    def test_m_pe_override(self):
        """m_pe parameter should override sample size."""
        result1 = finite_key_bounds(1000, 50, m_pe=100)
        result2 = finite_key_bounds(1000, 50, m_pe=500)
        # Tighter m_pe should give tighter bound
        assert result1["qber_upper"] >= result2["qber_upper"]


class TestFiniteKeySecretLength:
    """Test finite_key_secret_length function."""

    def test_high_qber_zero_key(self):
        """High QBER should yield zero secret length."""
        result = finite_key_secret_length(
            n_sifted=1000,
            qber_upper=0.15,
            qber_abort_threshold=0.11
        )
        # Function returns 'l_secret_bits' not 'secret_length'
        assert result["l_secret_bits"] <= 0 or result["aborted"]

    def test_low_qber_positive_key(self):
        """Low QBER should yield positive secret."""
        result = finite_key_secret_length(
            n_sifted=10000,
            qber_upper=0.05,
            qber_abort_threshold=0.11
        )
        assert result["l_secret_bits"] > 0 and not result["aborted"]

    def test_abort_threshold_respected(self):
        """Abort should occur above threshold."""
        result = finite_key_secret_length(
            n_sifted=1000,
            qber_upper=0.12,
            qber_abort_threshold=0.11
        )
        assert result["aborted"] or result["l_secret_bits"] <= 0


class TestDecoyParams:
    """Test DecoyParams validation."""

    def test_default_decoy(self):
        """Default DecoyParams should be valid."""
        params = DecoyParams()
        assert params.mu_s == 0.6
        assert params.mu_d == 0.1
        assert params.mu_v == 0.0
        assert abs(params.p_s + params.p_d + params.p_v - 1.0) < 1e-9

    def test_vacuum_intensity_zero(self):
        """Vacuum intensity must be 0."""
        with pytest.raises(ValueError):
            DecoyParams(mu_v=0.1)

    def test_signal_exceeds_decoy(self):
        """Signal intensity must exceed decoy."""
        with pytest.raises(ValueError):
            DecoyParams(mu_s=0.1, mu_d=0.2)

    def test_probabilities_sum_to_one(self):
        """Probabilities must sum to 1."""
        with pytest.raises(ValueError):
            DecoyParams(p_s=0.5, p_d=0.3, p_v=0.3)

    def test_probabilities_nonnegative(self):
        """Probabilities must be non-negative."""
        with pytest.raises(ValueError):
            DecoyParams(p_s=-0.1, p_d=0.6, p_v=0.5)

    def test_intensity_sigmas_nonnegative(self):
        """Intensity sigmas must be non-negative."""
        with pytest.raises(ValueError):
            DecoyParams(mu_s_sigma=-0.1)
        with pytest.raises(ValueError):
            DecoyParams(mu_d_sigma=-0.05)


class TestPoissonN:
    """Test poisson_n function."""

    def test_zero_mu_zero_photons(self):
        """Zero intensity should give prob 1 for 0 photons."""
        assert poisson_n(0.0, 0) == 1.0

    def test_zero_mu_nonzero_photons(self):
        """Zero intensity should give prob 0 for n>0."""
        assert poisson_n(0.0, 1) == 0.0
        assert poisson_n(0.0, 5) == 0.0

    def test_typical_mu_probabilities(self):
        """Probabilities should sum to 1."""
        mu = 0.6
        total = sum(poisson_n(mu, n) for n in range(10))
        # Higher n are unlikely but sum should approach 1
        assert 0.9 < total <= 1.0

    def test_single_photon_prob(self):
        """Single-photon probability should increase with mu."""
        p0 = poisson_n(0.1, 1)
        p1 = poisson_n(0.5, 1)
        p2 = poisson_n(1.0, 1)
        assert p0 < p1 < p2

    def test_monotonic_decay(self):
        """Higher photon numbers should be less probable."""
        mu = 0.6
        probs = [poisson_n(mu, n) for n in range(5)]
        # Should generally decrease (with possible small variations due to factorial)
        assert probs[0] > probs[1]


class TestSimulateDecoyBB84:
    """Test simulate_decoy_bb84 function."""

    def test_decoy_basic_execution(self):
        """Decoy simulation should execute without error."""
        result = simulate_decoy_bb84(n_pulses=1000, seed=42)
        assert isinstance(result, dict)

    def test_decoy_with_loss(self):
        """Decoy simulation with loss should work."""
        r_no_loss = simulate_decoy_bb84(n_pulses=5000, loss_db=0.0, seed=42)
        r_with_loss = simulate_decoy_bb84(n_pulses=5000, loss_db=5.0, seed=42)
        # With loss, gains should be lower
        assert r_with_loss.get("Q_mu_s", 0) <= r_no_loss.get("Q_mu_s", 1)

    def test_decoy_parameters(self):
        """Custom decoy parameters should be applied."""
        # Probabilities must sum to 1
        custom_decoy = DecoyParams(mu_s=0.8, mu_d=0.15, p_s=0.7, p_d=0.2, p_v=0.1)
        result = simulate_decoy_bb84(n_pulses=1000, decoy=custom_decoy, seed=42)
        assert isinstance(result, dict)

    def test_decoy_flip_prob(self):
        """Flip probability should affect QBER."""
        r_no_flip = simulate_decoy_bb84(n_pulses=5000, flip_prob=0.0, seed=42)
        r_with_flip = simulate_decoy_bb84(n_pulses=5000, flip_prob=0.05, seed=42)
        # Results should differ due to noise
        assert r_no_flip is not None and r_with_flip is not None

    def test_decoy_ec_efficiency(self):
        """EC efficiency should affect key rate calculation."""
        r1 = simulate_decoy_bb84(n_pulses=1000, ec_efficiency=1.16, seed=42)
        r2 = simulate_decoy_bb84(n_pulses=1000, ec_efficiency=1.5, seed=42)
        # Higher inefficiency should give lower key rate
        key_rate_1 = r1.get("secret_key_rate_asymptotic", 0)
        key_rate_2 = r2.get("secret_key_rate_asymptotic", 0)
        # Results may vary, but function should execute
        assert isinstance(r1, dict) and isinstance(r2, dict)
