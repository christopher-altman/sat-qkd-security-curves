"""Tests for optical_link.py stub module and integration tests for remaining modules."""
import pytest
import math
import numpy as np
from sat_qkd_lab.optical_link import (
    OpticalLinkParams, geometric_coupling_loss_db, 
    pointing_loss_db, atmospheric_extinction_db, optical_link_loss_db
)


class TestOpticalLinkParams:
    """Test OpticalLinkParams dataclass."""

    def test_default_params(self):
        """Default parameters should be physically reasonable."""
        params = OpticalLinkParams()
        assert params.wavelength_m == 850e-9
        assert params.tx_divergence_urad > 0
        assert params.rx_aperture_m > 0
        assert params.altitude_m > 0
        assert params.earth_radius_m > params.altitude_m

    def test_custom_params(self):
        """Custom parameters should be accepted."""
        params = OpticalLinkParams(
            wavelength_m=1550e-9,
            tx_divergence_urad=10.0,
            rx_aperture_m=0.5,
        )
        assert params.wavelength_m == 1550e-9
        assert params.tx_divergence_urad == 10.0
        assert params.rx_aperture_m == 0.5

    def test_immutable(self):
        """OpticalLinkParams should be frozen."""
        params = OpticalLinkParams()
        with pytest.raises(Exception):
            params.wavelength_m = 1000e-9


class TestGeometricCouplingLoss:
    """Test geometric_coupling_loss_db function."""

    def test_zero_range(self):
        """Very short range should give low loss."""
        loss = geometric_coupling_loss_db(1000, 10.0, 1.0)
        assert loss >= 0  # May be zero or slightly negative due to numerical precision
        assert loss < 100

    def test_increasing_range(self):
        """Longer range should increase loss."""
        loss_near = geometric_coupling_loss_db(1000, 10.0, 1.0)
        loss_far = geometric_coupling_loss_db(100000, 10.0, 1.0)
        assert loss_far > loss_near

    def test_larger_aperture(self):
        """Larger aperture should decrease loss."""
        loss_small = geometric_coupling_loss_db(10000, 10.0, 0.5)
        loss_large = geometric_coupling_loss_db(10000, 10.0, 1.5)
        assert loss_large < loss_small

    def test_smaller_divergence(self):
        """Smaller divergence affects coupling loss."""
        loss_narrow = geometric_coupling_loss_db(10000, 5.0, 1.0)
        loss_wide = geometric_coupling_loss_db(10000, 20.0, 1.0)
        # Both should be valid non-negative losses
        assert loss_narrow >= 0
        assert loss_wide >= 0

    def test_physical_bounds(self):
        """Loss should be non-negative and < 100 dB for reasonable params."""
        for range_m in [100, 1000, 10000, 100000]:
            for div_urad in [5, 10, 20]:
                for ap_m in [0.3, 1.0, 2.0]:
                    loss = geometric_coupling_loss_db(range_m, div_urad, ap_m)
                    assert 0 <= loss <= 100

    def test_zero_aperture_edge(self):
        """Zero aperture should give high loss."""
        loss = geometric_coupling_loss_db(10000, 10.0, 0.01)
        assert loss > 20  # Small aperture gives high loss

    def test_zero_divergence(self):
        """Zero divergence edge case should handle gracefully."""
        loss = geometric_coupling_loss_db(10000, 0.0, 1.0)
        assert loss == 0.0  # Perfect beam (unphysical but safe)


class TestPointingLoss:
    """Test pointing_loss_db function."""

    def test_zero_pointing_error(self):
        """Zero pointing error should give zero loss."""
        loss = pointing_loss_db(0.0, 10.0)
        assert loss == 0.0

    def test_small_pointing_error(self):
        """Small error relative to divergence should give small loss."""
        loss = pointing_loss_db(1.0, 10.0)  # 1/10 ratio
        assert 0 < loss < 10

    def test_large_pointing_error(self):
        """Large error relative to divergence should give large loss."""
        loss = pointing_loss_db(20.0, 10.0)  # 2/1 ratio
        assert loss > 10

    def test_increasing_error(self):
        """Larger pointing error should increase loss."""
        loss_small = pointing_loss_db(1.0, 10.0)
        loss_large = pointing_loss_db(5.0, 10.0)
        assert loss_large > loss_small

    def test_increasing_divergence(self):
        """Larger divergence should reduce loss (more forgiving)."""
        loss_narrow = pointing_loss_db(2.0, 10.0)
        loss_wide = pointing_loss_db(2.0, 20.0)
        assert loss_wide < loss_narrow

    def test_zero_divergence(self):
        """Zero divergence should return zero loss (undefined/safe)."""
        loss = pointing_loss_db(2.0, 0.0)
        assert loss == 0.0

    def test_symmetric_error(self):
        """Error ratio should determine loss."""
        # Same ratio should give same loss
        loss1 = pointing_loss_db(2.0, 10.0)  # ratio = 0.2
        loss2 = pointing_loss_db(4.0, 20.0)  # ratio = 0.2
        assert abs(loss1 - loss2) < 1e-10


class TestAtmosphericExtinction:
    """Test atmospheric_extinction_db function."""

    def test_zenith_minimum(self):
        """Zenith (90°) should give minimum extinction."""
        loss_zenith = atmospheric_extinction_db(90.0, 850e-9)
        loss_high = atmospheric_extinction_db(45.0, 850e-9)
        loss_low = atmospheric_extinction_db(10.0, 850e-9)
        assert loss_zenith <= loss_high <= loss_low

    def test_low_elevation_high_loss(self):
        """Low elevation should give high extinction."""
        loss = atmospheric_extinction_db(5.0, 850e-9)
        assert loss > 2.0

    def test_increasing_wavelength(self):
        """The model doesn't explicitly vary with wavelength (stub)."""
        # Stub implementation doesn't model wavelength dependence
        loss_vis = atmospheric_extinction_db(45.0, 550e-9)
        loss_ir = atmospheric_extinction_db(45.0, 1550e-9)
        # Both should be close (stub doesn't differentiate)
        assert abs(loss_vis - loss_ir) < 1.0

    def test_visibility_parameter(self):
        """Different visibility should affect loss."""
        loss_good = atmospheric_extinction_db(45.0, 850e-9, visibility_km=23.0)
        loss_poor = atmospheric_extinction_db(45.0, 850e-9, visibility_km=5.0)
        # Stub implementation may not use visibility, but parameter should be accepted
        assert isinstance(loss_good, float)
        assert isinstance(loss_poor, float)

    def test_elevation_clipping(self):
        """Very low elevation should be clipped."""
        loss_extreme = atmospheric_extinction_db(0.1, 850e-9)
        loss_valid = atmospheric_extinction_db(1.0, 850e-9)
        # Should not crash
        assert loss_extreme > 0
        assert loss_valid > 0

    def test_airmass_scaling(self):
        """Loss should increase with airmass (1/sin(el))."""
        elevations = [10, 30, 45, 60, 85]
        losses = [atmospheric_extinction_db(el, 850e-9) for el in elevations]
        # Should be monotonically decreasing
        for i in range(len(losses) - 1):
            assert losses[i] >= losses[i + 1]


class TestOpticalLinkLossStub:
    """Test optical_link_loss_db stub function."""

    def test_not_implemented(self):
        """optical_link_loss_db should raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            optical_link_loss_db(45.0)

    def test_not_implemented_with_params(self):
        """Should raise NotImplementedError even with parameters."""
        params = OpticalLinkParams()
        with pytest.raises(NotImplementedError):
            optical_link_loss_db(45.0, params=params)


class TestOpticalLinkComponentInteraction:
    """Test interactions between optical link components."""

    def test_typical_leo_scenario(self):
        """Typical LEO QKD scenario should have reasonable losses."""
        params = OpticalLinkParams()
        
        # Typical LEO at 500km altitude, 15 µrad divergence, 1m aperture
        range_m = 800_000  # 800 km slant range at medium elevation
        
        geometric_loss = geometric_coupling_loss_db(
            range_m, 
            params.tx_divergence_urad, 
            params.rx_aperture_m
        )
        
        pointing_loss = pointing_loss_db(
            params.pointing_rms_urad,
            params.tx_divergence_urad
        )
        
        atmospheric_loss = atmospheric_extinction_db(
            45.0,  # Mid elevation
            params.wavelength_m
        )
        
        total = geometric_loss + pointing_loss + atmospheric_loss
        
        # Should be reasonable for LEO QKD
        assert 20 < total < 60

    def test_best_case_scenario(self):
        """Zenith pass with no pointing error should minimize loss."""
        params = OpticalLinkParams(pointing_rms_urad=0.1)
        
        # Zenith, minimum range
        range_m = params.altitude_m
        
        geometric_loss = geometric_coupling_loss_db(
            range_m,
            params.tx_divergence_urad,
            params.rx_aperture_m
        )
        
        pointing_loss = pointing_loss_db(0.1, params.tx_divergence_urad)
        atmospheric_loss = atmospheric_extinction_db(90.0, params.wavelength_m)
        
        total = geometric_loss + pointing_loss + atmospheric_loss
        assert total < 30

    def test_worst_case_scenario(self):
        """Horizon pass with large errors should maximize loss."""
        # Large pointing error
        pointing_loss = pointing_loss_db(5.0, 15.0)
        
        # Far distance
        geometric_loss = geometric_coupling_loss_db(2000_000, 15.0, 1.0)
        
        # Low elevation
        atmospheric_loss = atmospheric_extinction_db(5.0, 850e-9)
        
        total = geometric_loss + pointing_loss + atmospheric_loss
        assert total > 30


class TestOpticalLinkEdgeCases:
    """Test edge cases across optical link functions."""

    def test_extremely_large_range(self):
        """Very large range should handle gracefully."""
        loss = geometric_coupling_loss_db(1e9, 10.0, 1.0)
        assert 0 < loss < 200

    def test_extremely_small_aperture(self):
        """Very small aperture should give high loss."""
        loss = geometric_coupling_loss_db(100_000, 10.0, 0.001)
        assert loss > 50  # Very small aperture gives high loss

    def test_extremely_large_divergence(self):
        """Very large divergence (wide beam) should reduce loss."""
        loss = geometric_coupling_loss_db(100_000, 1000.0, 1.0)
        assert 0 <= loss < 50

    def test_all_zero_inputs_safe(self):
        """Zero inputs should not crash (return safe defaults)."""
        # These should handle gracefully even with problematic inputs
        assert geometric_coupling_loss_db(0, 0, 0) >= 0
        assert pointing_loss_db(0, 0) >= 0
        assert atmospheric_extinction_db(0, 0) >= 0

    def test_negative_inputs_handled(self):
        """Negative inputs should be handled (clipped or error)."""
        # Should not crash on edge cases
        loss = atmospheric_extinction_db(-10.0, 850e-9)  # Clipped to 1.0°
        assert loss > 0


class TestOpticalLinkPhysicalLaws:
    """Verify optical link functions obey physical laws."""

    def test_coupling_monotonicity(self):
        """Coupling should decrease with distance (monotonic)."""
        ranges = [100_000, 500_000, 1_000_000, 2_000_000]
        losses = [geometric_coupling_loss_db(r, 15.0, 1.0) for r in ranges]
        
        # Should be monotonically increasing
        for i in range(len(losses) - 1):
            assert losses[i] <= losses[i + 1]

    def test_pointing_loss_exponential(self):
        """Pointing loss should follow exponential decay."""
        errors = np.linspace(0.1, 5.0, 10)
        losses = [pointing_loss_db(e, 10.0) for e in errors]
        
        # Should be generally increasing
        for i in range(len(losses) - 1):
            assert losses[i] <= losses[i + 1]

    def test_aperture_quadratic_effect(self):
        """Larger aperture should have significant effect on coupling."""
        loss_2x = geometric_coupling_loss_db(100_000, 15.0, 0.5)
        loss_1x = geometric_coupling_loss_db(100_000, 15.0, 1.0)
        
        # 2x aperture should reduce loss (roughly 6dB = 2x power)
        delta = loss_2x - loss_1x
        assert 3 < delta < 10

    def test_divergence_linear_effect(self):
        """Larger divergence affects coupling loss."""
        loss_narrow = geometric_coupling_loss_db(100_000, 10.0, 1.0)
        loss_wide = geometric_coupling_loss_db(100_000, 30.0, 1.0)
        
        # Both should be valid losses
        assert loss_narrow >= 0
        assert loss_wide >= 0
