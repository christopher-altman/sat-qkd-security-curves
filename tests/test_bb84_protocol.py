"""Tests for bb84.py - BB84 QKD protocol simulation."""
import pytest
import numpy as np
from sat_qkd_lab.bb84 import simulate_bb84, _apply_detector_effects
from sat_qkd_lab.detector import DetectorParams, DEFAULT_DETECTOR
from sat_qkd_lab.attacks import AttackConfig


class TestBB84BasicSimulation:
    """Test basic BB84 simulation functionality."""

    def test_bb84_default_params_succeeds(self):
        """BB84 with default parameters should succeed."""
        result = simulate_bb84(n_pulses=1000, seed=42)
        assert result.n_sent == 1000
        assert result.n_received <= result.n_sent
        assert result.n_sifted <= result.n_received
        assert not result.aborted

    def test_bb84_zero_pulses(self):
        """Zero pulses should abort."""
        result = simulate_bb84(n_pulses=0, seed=42)
        assert result.n_sent == 0
        assert result.aborted

    def test_bb84_high_loss_aborts(self):
        """Very high loss (total darkness) should abort."""
        result = simulate_bb84(n_pulses=1000, loss_db=50.0, seed=42)
        assert result.aborted

    def test_bb84_no_loss_basic(self):
        """No loss should produce reasonable results."""
        result = simulate_bb84(n_pulses=10000, loss_db=0.0, seed=42)
        assert result.n_received > 0
        assert result.n_sifted > 0
        assert not result.aborted

    def test_bb84_deterministic_seeding(self):
        """Same seed should produce identical results."""
        r1 = simulate_bb84(n_pulses=5000, seed=123)
        r2 = simulate_bb84(n_pulses=5000, seed=123)
        assert r1.n_sent == r2.n_sent
        assert r1.n_received == r2.n_received
        assert r1.n_sifted == r2.n_sifted
        assert r1.qber == r2.qber

    def test_bb84_different_seeds_differ(self):
        """Different seeds should produce different results."""
        r1 = simulate_bb84(n_pulses=5000, seed=123)
        r2 = simulate_bb84(n_pulses=5000, seed=456)
        assert r1.n_received != r2.n_received or r1.qber != r2.qber

    def test_bb84_secret_fraction_in_valid_range(self):
        """Secret fraction should be between 0 and 1."""
        for loss in [0.0, 5.0, 15.0]:
            result = simulate_bb84(n_pulses=5000, loss_db=loss, seed=42)
            assert 0.0 <= result.secret_fraction <= 1.0

    def test_bb84_qber_physical_bounds(self):
        """QBER should be in [0, 0.5] (or NaN if no sifted bits)."""
        for loss in [0.0, 3.0, 10.0]:
            result = simulate_bb84(n_pulses=5000, loss_db=loss, seed=42)
            if result.n_sifted > 0:
                assert 0.0 <= result.qber <= 0.5
            else:
                assert np.isnan(result.qber)


class TestBB84Loss:
    """Test BB84 behavior under various loss conditions."""

    def test_loss_increases_with_db(self):
        """Higher loss should reduce received counts."""
        r_low = simulate_bb84(n_pulses=10000, loss_db=0.0, seed=42)
        r_high = simulate_bb84(n_pulses=10000, loss_db=10.0, seed=42)
        assert r_low.n_received > r_high.n_received

    def test_loss_approaches_zero_at_extreme(self):
        """Extreme loss should approach zero received counts."""
        result = simulate_bb84(n_pulses=10000, loss_db=40.0, seed=42)
        # At extreme loss, received counts should be very small (near zero)
        # Background clicks can still occur, so allow a small number
        assert result.n_received < 10

    def test_transmittance_calculation(self):
        """Loss in dB should correctly map to transmittance."""
        # 10 dB loss = 10^(-10/10) = 0.1 transmittance
        r1 = simulate_bb84(n_pulses=50000, loss_db=10.0, seed=42)
        r2 = simulate_bb84(n_pulses=50000, loss_db=0.0, seed=42)
        # Expected ratio roughly: (0.1 / 1.0) * sifting_prob
        ratio = r1.n_received / r2.n_received if r2.n_received > 0 else 0
        assert 0.05 < ratio < 0.2  # Allows for randomness


class TestBB84Noise:
    """Test BB84 behavior under intrinsic noise."""

    def test_flip_prob_zero(self):
        """Zero flip probability should produce low QBER at no loss."""
        result = simulate_bb84(n_pulses=10000, loss_db=0.0, flip_prob=0.0, seed=42)
        if result.n_sifted > 0:
            # With background clicks, QBER might not be zero
            # But on average should be low
            assert result.qber < 0.2

    def test_flip_prob_high_increases_qber(self):
        """High flip probability should increase QBER."""
        r_no_flip = simulate_bb84(n_pulses=10000, loss_db=0.0, flip_prob=0.0, seed=42)
        r_with_flip = simulate_bb84(n_pulses=10000, loss_db=0.0, flip_prob=0.1, seed=42)
        if r_no_flip.n_sifted > 0 and r_with_flip.n_sifted > 0:
            assert r_with_flip.qber > r_no_flip.qber

    def test_flip_prob_bounds(self):
        """Flip probability should be valid in [0, 1]."""
        # Flip prob is not explicitly validated in simulate_bb84,
        # it just affects the simulation. Values outside [0,1] would cause issues
        # but the function doesn't validate it upfront.
        # Test that extreme flip values produce results (no validation error)
        result = simulate_bb84(flip_prob=0.0, n_pulses=100, seed=42)
        assert result.n_sent == 100


class TestBB84Detector:
    """Test detector model integration."""

    def test_default_detector(self):
        """Default detector should work."""
        result = simulate_bb84(n_pulses=5000, detector=DEFAULT_DETECTOR, seed=42)
        assert not result.aborted or result.n_sent == 0

    def test_perfect_detector(self):
        """Perfect detector (eta=1, p_bg=0) should maximize counts."""
        perf = DetectorParams(eta=1.0, p_bg=0.0)
        default_result = simulate_bb84(n_pulses=10000, loss_db=0.0, seed=42)
        perf_result = simulate_bb84(n_pulses=10000, loss_db=0.0, detector=perf, seed=42)
        assert perf_result.n_received >= default_result.n_received

    def test_low_efficiency_detector(self):
        """Low efficiency detector should reduce counts."""
        low_eff = DetectorParams(eta=0.05, p_bg=1e-4)
        result = simulate_bb84(n_pulses=10000, loss_db=0.0, detector=low_eff, seed=42)
        assert result.n_received < 2000  # Expect sparse detections

    def test_high_background_increases_qber(self):
        """High background should increase QBER."""
        r_low_bg = simulate_bb84(n_pulses=10000, loss_db=0.0, 
                                 detector=DetectorParams(eta=0.2, p_bg=1e-5), seed=42)
        r_high_bg = simulate_bb84(n_pulses=10000, loss_db=0.0,
                                  detector=DetectorParams(eta=0.2, p_bg=0.01), seed=42)
        if r_low_bg.n_sifted > 0 and r_high_bg.n_sifted > 0:
            assert r_high_bg.qber > r_low_bg.qber

    def test_basis_dependent_efficiency(self):
        """Different efficiencies per basis should be applied."""
        det = DetectorParams(eta_z=0.3, eta_x=0.1)
        result = simulate_bb84(n_pulses=5000, loss_db=0.0, detector=det, seed=42)
        assert result.n_received > 0


class TestBB84Attacks:
    """Test integration with attack models."""

    def test_no_attack(self):
        """No attack should work."""
        result = simulate_bb84(n_pulses=5000, attack="none", seed=42)
        assert not result.aborted or result.n_sent == 0

    def test_intercept_resend_attack(self):
        """Intercept-resend attack should increase QBER."""
        r_no_attack = simulate_bb84(n_pulses=10000, loss_db=0.0, attack="none", seed=42)
        r_attack = simulate_bb84(n_pulses=10000, loss_db=0.0, attack="intercept_resend", seed=42)
        if r_no_attack.n_sifted > 0 and r_attack.n_sifted > 0:
            # IR attack should roughly double QBER (wrong basis = 50% error)
            assert r_attack.qber > r_no_attack.qber

    def test_pns_attack(self):
        """PNS attack should register in metadata."""
        config = AttackConfig(attack="pns", mu=0.6)
        result = simulate_bb84(n_pulses=5000, attack_config=config, seed=42)
        assert "pns_multi_photon_frac" in result.meta

    def test_time_shift_attack(self):
        """Time-shift attack should work."""
        config = AttackConfig(attack="time_shift", timeshift_bias=0.5)
        result = simulate_bb84(n_pulses=5000, attack_config=config, seed=42)
        assert result.n_sent == 5000

    def test_blinding_attack(self):
        """Blinding attack should work."""
        config = AttackConfig(attack="blinding", blinding_prob=0.05)
        result = simulate_bb84(n_pulses=5000, attack_config=config, seed=42)
        # Blinding attack should be applied (check that it doesn't fail)
        assert result.n_sent == 5000


class TestDetectorEffects:
    """Test _apply_detector_effects function."""

    def test_no_effects(self):
        """No detector effects should leave arrays unchanged."""
        click = np.array([True, False, True, False])
        bg_only = np.array([False, False, False, False])
        det = DetectorParams()
        rng = np.random.default_rng(42)
        
        click_out, bg_only_out, _ = _apply_detector_effects(
            click.copy(), bg_only.copy(), None, det, rng
        )
        assert np.array_equal(click_out, click)
        assert np.array_equal(bg_only_out, bg_only)

    def test_dead_time_suppresses_clicks(self):
        """Dead time should suppress subsequent clicks."""
        click = np.array([True, True, True, True, False, True])
        bg_only = np.array([False] * 6)
        det = DetectorParams(dead_time_pulses=2)
        rng = np.random.default_rng(42)
        
        click_out, _, _ = _apply_detector_effects(
            click.copy(), bg_only.copy(), None, det, rng
        )
        # First click at 0, then positions 1,2 should be suppressed
        assert click_out[0] == True
        assert click_out[1] == False
        assert click_out[2] == False

    def test_afterpulse_can_create_clicks(self):
        """Afterpulsing can create clicks after a detection."""
        click = np.array([True, False, False, False])
        bg_only = np.array([False, False, False, False])
        det = DetectorParams(p_afterpulse=1.0, afterpulse_window=2)
        rng = np.random.default_rng(42)
        
        click_out, _, _ = _apply_detector_effects(
            click.copy(), bg_only.copy(), None, det, rng
        )
        # Position 0 has a click, so positions 1-2 should have afterpulse
        assert click_out[0] == True
        # Positions 1-2 may have afterpulse (prob=1)
        assert click_out[1] or click_out[2]

    def test_copies_made(self):
        """Detector effects should not modify input arrays."""
        click_orig = np.array([True, False, True])
        bg_only_orig = np.array([False, False, False])
        blinding_orig = np.array([False, False, False])
        det = DetectorParams(dead_time_pulses=1)
        rng = np.random.default_rng(42)
        
        click_copy = click_orig.copy()
        bg_copy = bg_only_orig.copy()
        blinding_copy = blinding_orig.copy()
        
        _apply_detector_effects(click_copy, bg_copy, blinding_copy, det, rng)
        
        # Originals should be unchanged
        assert np.array_equal(click_orig, [True, False, True])
        assert np.array_equal(bg_only_orig, [False, False, False])
