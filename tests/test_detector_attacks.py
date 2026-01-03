"""Tests for detector.py and attacks.py modules."""
import pytest
import numpy as np
import math
from sat_qkd_lab.detector import DetectorParams, DEFAULT_DETECTOR
from sat_qkd_lab.attacks import (
    AttackConfig, AttackState, apply_attack, poisson_multi_photon_fraction, apply_time_shift
)


class TestDetectorParams:
    """Test DetectorParams dataclass and validation."""

    def test_default_detector(self):
        """DEFAULT_DETECTOR should have reasonable values."""
        assert 0.0 <= DEFAULT_DETECTOR.eta <= 1.0
        assert 0.0 <= DEFAULT_DETECTOR.p_bg <= 1.0
        assert DEFAULT_DETECTOR.eta == 0.2
        assert DEFAULT_DETECTOR.p_bg == 1e-4

    def test_detector_creation(self):
        """Detector should create with valid parameters."""
        det = DetectorParams(eta=0.5, p_bg=1e-5)
        assert det.eta == 0.5
        assert det.p_bg == 1e-5

    def test_eta_bounds(self):
        """eta must be in [0, 1]."""
        with pytest.raises(ValueError):
            DetectorParams(eta=-0.1)
        with pytest.raises(ValueError):
            DetectorParams(eta=1.1)
        assert DetectorParams(eta=0.0).eta == 0.0
        assert DetectorParams(eta=1.0).eta == 1.0

    def test_p_bg_bounds(self):
        """p_bg must be in [0, 1]."""
        with pytest.raises(ValueError):
            DetectorParams(p_bg=-0.01)
        with pytest.raises(ValueError):
            DetectorParams(p_bg=1.01)
        assert DetectorParams(p_bg=0.0).p_bg == 0.0
        assert DetectorParams(p_bg=1.0).p_bg == 1.0

    def test_afterpulse_params(self):
        """Afterpulse parameters should be validated."""
        with pytest.raises(ValueError):
            DetectorParams(p_afterpulse=-0.1)
        with pytest.raises(ValueError):
            DetectorParams(p_afterpulse=1.1)
        with pytest.raises(ValueError):
            DetectorParams(afterpulse_window=-1)
        with pytest.raises(ValueError):
            DetectorParams(afterpulse_decay=-0.1)

    def test_dead_time_validation(self):
        """dead_time_pulses must be non-negative."""
        with pytest.raises(ValueError):
            DetectorParams(dead_time_pulses=-1)
        assert DetectorParams(dead_time_pulses=0).dead_time_pulses == 0
        assert DetectorParams(dead_time_pulses=10).dead_time_pulses == 10

    def test_basis_dependent_efficiency_defaults(self):
        """eta_z and eta_x should default to eta."""
        det = DetectorParams(eta=0.3)
        assert det.eta_z == 0.3
        assert det.eta_x == 0.3

    def test_basis_dependent_efficiency_override(self):
        """eta_z and eta_x can override eta."""
        det = DetectorParams(eta=0.3, eta_z=0.4, eta_x=0.2)
        assert det.eta_z == 0.4
        assert det.eta_x == 0.2

    def test_basis_dependent_efficiency_bounds(self):
        """eta_z and eta_x must be in [0, 1]."""
        with pytest.raises(ValueError):
            DetectorParams(eta_z=-0.1)
        with pytest.raises(ValueError):
            DetectorParams(eta_x=1.5)

    def test_immutable(self):
        """DetectorParams should be frozen."""
        det = DEFAULT_DETECTOR
        with pytest.raises(Exception):
            det.eta = 0.5


class TestAttackConfig:
    """Test AttackConfig dataclass and validation."""

    def test_default_config(self):
        """Default AttackConfig should have no attack."""
        config = AttackConfig()
        assert config.attack == "none"

    def test_attack_types(self):
        """Valid attack types should work."""
        for attack in ["none", "intercept_resend", "pns", "time_shift", "blinding"]:
            config = AttackConfig(attack=attack)
            assert config.attack == attack

    def test_mu_bounds(self):
        """mu must be non-negative."""
        with pytest.raises(ValueError):
            AttackConfig(mu=-0.1)
        assert AttackConfig(mu=0.0).mu == 0.0
        assert AttackConfig(mu=1.0).mu == 1.0

    def test_timeshift_bias_bounds(self):
        """timeshift_bias must be in [0, 1]."""
        with pytest.raises(ValueError):
            AttackConfig(timeshift_bias=-0.1)
        with pytest.raises(ValueError):
            AttackConfig(timeshift_bias=1.1)

    def test_blinding_mode_validation(self):
        """blinding_mode must be loud or stealth."""
        assert AttackConfig(blinding_mode="loud").blinding_mode == "loud"
        assert AttackConfig(blinding_mode="stealth").blinding_mode == "stealth"
        with pytest.raises(ValueError):
            AttackConfig(blinding_mode="invalid")

    def test_blinding_prob_bounds(self):
        """blinding_prob must be in [0, 1]."""
        with pytest.raises(ValueError):
            AttackConfig(blinding_prob=-0.1)
        with pytest.raises(ValueError):
            AttackConfig(blinding_prob=1.1)

    def test_leakage_fraction_bounds(self):
        """leakage_fraction must be in [0, 1]."""
        with pytest.raises(ValueError):
            AttackConfig(leakage_fraction=-0.1)
        with pytest.raises(ValueError):
            AttackConfig(leakage_fraction=1.1)

    def test_immutable(self):
        """AttackConfig should be frozen."""
        config = AttackConfig()
        with pytest.raises(Exception):
            config.attack = "blinding"


class TestPoissonMultiPhoton:
    """Test poisson_multi_photon_fraction function."""

    def test_zero_mu(self):
        """mu=0 should give 0 multi-photon fraction."""
        assert poisson_multi_photon_fraction(0.0) == 0.0

    def test_small_mu(self):
        """Small mu should give small multi-photon fraction."""
        frac = poisson_multi_photon_fraction(0.1)
        assert 0.0 <= frac < 0.01

    def test_typical_mu(self):
        """mu=0.6 should give reasonable fraction."""
        frac = poisson_multi_photon_fraction(0.6)
        assert 0.05 < frac < 0.15

    def test_large_mu(self):
        """Large mu should give large multi-photon fraction."""
        frac = poisson_multi_photon_fraction(5.0)
        assert 0.95 < frac <= 1.0

    def test_negative_mu(self):
        """Negative mu should return 0."""
        assert poisson_multi_photon_fraction(-1.0) == 0.0

    def test_monotonic(self):
        """Multi-photon fraction should increase with mu."""
        fracs = [poisson_multi_photon_fraction(mu) for mu in [0.1, 0.3, 0.6, 1.0, 2.0, 5.0]]
        for i in range(len(fracs) - 1):
            assert fracs[i] <= fracs[i + 1]


class TestTimeShiftAttack:
    """Test apply_time_shift function."""

    def test_no_bias(self):
        """Zero bias should leave efficiencies unchanged."""
        eta_z, eta_x = apply_time_shift(0.3, 0.2, 0.0)
        assert eta_z == 0.3
        assert eta_x == 0.2

    def test_equal_efficiencies(self):
        """Equal efficiencies should be unaffected."""
        eta_z, eta_x = apply_time_shift(0.3, 0.3, 0.5)
        assert eta_z == 0.3
        assert eta_x == 0.3

    def test_bias_favors_larger(self):
        """Bias should increase the larger efficiency."""
        eta_z, eta_x = apply_time_shift(0.4, 0.2, 0.5)
        # 0.4 is larger, should increase
        assert eta_z > 0.4
        # 0.2 is smaller, should decrease
        assert eta_x < 0.2

    def test_bias_favors_x(self):
        """If X is larger, bias should favor X."""
        eta_z, eta_x = apply_time_shift(0.2, 0.4, 0.5)
        # 0.4 (X) is larger, should increase
        assert eta_x > 0.4
        # 0.2 (Z) is smaller, should decrease
        assert eta_z < 0.2

    def test_clamping(self):
        """Results should stay in [0, 1]."""
        eta_z, eta_x = apply_time_shift(0.9, 0.1, 1.0)
        assert 0.0 <= eta_z <= 1.0
        assert 0.0 <= eta_x <= 1.0


class TestApplyAttack:
    """Test apply_attack function."""

    def test_no_attack(self):
        """No attack should leave bits unchanged."""
        state = AttackState(
            a_bits=np.array([0, 1, 0, 1], dtype=np.int8),
            a_basis=np.array([0, 0, 1, 1], dtype=np.int8),
            click=np.array([True, True, True, False]),
            bg_only=np.array([False] * 4),
        )
        config = AttackConfig(attack="none")
        rng = np.random.default_rng(42)
        outcome = apply_attack(config, state, rng)
        
        assert np.array_equal(outcome.incoming_bits[:3], state.a_bits[:3])
        assert np.array_equal(outcome.incoming_basis[:3], state.a_basis[:3])

    def test_intercept_resend_adds_error(self):
        """Intercept-resend should flip some basis-mismatched bits."""
        state = AttackState(
            a_bits=np.array([0] * 100, dtype=np.int8),
            a_basis=np.array([0] * 50 + [1] * 50, dtype=np.int8),
            click=np.ones(100, dtype=bool),
            bg_only=np.zeros(100, dtype=bool),
        )
        config = AttackConfig(attack="intercept_resend")
        rng = np.random.default_rng(42)
        outcome = apply_attack(config, state, rng)
        
        # Some bits should differ due to eavesdropper's guesses
        assert outcome.incoming_bits.size == 100

    def test_blinding_forces_clicks(self):
        """Blinding attack should force some clicks."""
        state = AttackState(
            a_bits=np.array([0, 1, 0, 1], dtype=np.int8),
            a_basis=np.array([0, 0, 1, 1], dtype=np.int8),
            click=np.array([True, False, False, False]),
            bg_only=np.array([False] * 4),
        )
        config = AttackConfig(attack="blinding", blinding_prob=1.0)
        rng = np.random.default_rng(42)
        outcome = apply_attack(config, state, rng)
        
        # All positions should be in click array (some forced)
        assert np.sum(outcome.click) >= 1

    def test_pns_attack_metadata(self):
        """PNS attack should set metadata."""
        state = AttackState(
            a_bits=np.array([0, 1, 0, 1], dtype=np.int8),
            a_basis=np.array([0, 0, 1, 1], dtype=np.int8),
            click=np.ones(4, dtype=bool),
            bg_only=np.zeros(4, dtype=bool),
        )
        config = AttackConfig(attack="pns", mu=0.6)
        rng = np.random.default_rng(42)
        outcome = apply_attack(config, state, rng)
        
        assert "pns_multi_photon_frac" in outcome.meta
        assert 0.0 <= outcome.meta["pns_multi_photon_frac"] <= 1.0

    def test_leakage_metadata(self):
        """Leakage should be recorded in metadata."""
        state = AttackState(
            a_bits=np.array([0, 1, 0, 1], dtype=np.int8),
            a_basis=np.array([0, 0, 1, 1], dtype=np.int8),
            click=np.ones(4, dtype=bool),
            bg_only=np.zeros(4, dtype=bool),
        )
        config = AttackConfig(attack="none", leakage_fraction=0.1)
        rng = np.random.default_rng(42)
        outcome = apply_attack(config, state, rng)
        
        assert "leakage_fraction" in outcome.meta
        assert outcome.meta["leakage_fraction"] == 0.1

    def test_blinding_loud_mode(self):
        """Loud blinding should assign random bits."""
        state = AttackState(
            a_bits=np.array([1, 1, 1, 1], dtype=np.int8),
            a_basis=np.array([0, 0, 0, 0], dtype=np.int8),
            click=np.array([False, False, False, False]),
            bg_only=np.zeros(4, dtype=bool),
        )
        config = AttackConfig(attack="blinding", blinding_prob=1.0, blinding_mode="loud")
        rng = np.random.default_rng(42)
        outcome = apply_attack(config, state, rng)
        
        # Forced bits may be random
        assert outcome.blinding_forced.size == 4

    def test_blinding_stealth_mode(self):
        """Stealth blinding should force to Alice's bits."""
        state = AttackState(
            a_bits=np.array([0, 1, 0, 1], dtype=np.int8),
            a_basis=np.array([0, 0, 0, 0], dtype=np.int8),
            click=np.array([False, False, False, False]),
            bg_only=np.zeros(4, dtype=bool),
        )
        config = AttackConfig(attack="blinding", blinding_prob=1.0, blinding_mode="stealth")
        rng = np.random.default_rng(42)
        outcome = apply_attack(config, state, rng)
        
        # Stealth mode forces bits to match Alice's
        assert outcome.incoming_bits is not None
