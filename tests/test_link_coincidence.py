"""Tests for coincidence.py, link_budget.py, and free_space_link.py with edge cases."""
import pytest
import math
import numpy as np
from sat_qkd_lab.coincidence import match_coincidences, CoincidenceResult
from sat_qkd_lab.timing import TimingModel
from sat_qkd_lab.timetags import TimeTags
from sat_qkd_lab.link_budget import SatLinkParams, slant_range_m, fspl_db, atmospheric_loss_db, total_channel_loss_db


class TestMatchCoincidences:
    """Test match_coincidences function with edge cases."""

    def test_empty_tags(self):
        """Empty tags should give zero coincidences."""
        tags_a = TimeTags(
            times=np.array([], dtype=float),
            is_signal=np.ones(0, dtype=bool),
            basis=np.zeros(0, dtype=np.int8),
            bit=np.zeros(0, dtype=np.int8),
        )
        tags_b = TimeTags(
            times=np.array([], dtype=float),
            is_signal=np.ones(0, dtype=bool),
            basis=np.zeros(0, dtype=np.int8),
            bit=np.zeros(0, dtype=np.int8),
        )
        result = match_coincidences(tags_a, tags_b, tau_seconds=1e-6)
        assert result.coincidences == 0
        assert result.accidentals == 0
        assert result.car == 0.0

    def test_invalid_tau(self):
        """Invalid tau should raise error."""
        tags_a = TimeTags(
            times=np.array([1.0], dtype=float),
            is_signal=np.ones(1, dtype=bool),
            basis=np.zeros(1, dtype=np.int8),
            bit=np.zeros(1, dtype=np.int8),
        )
        tags_b = TimeTags(
            times=np.array([1.0], dtype=float),
            is_signal=np.ones(1, dtype=bool),
            basis=np.zeros(1, dtype=np.int8),
            bit=np.zeros(1, dtype=np.int8),
        )
        with pytest.raises(ValueError):
            match_coincidences(tags_a, tags_b, tau_seconds=0.0)
        with pytest.raises(ValueError):
            match_coincidences(tags_a, tags_b, tau_seconds=-1e-6)

    def test_perfect_coincidences(self):
        """Perfectly aligned events should coincide."""
        tags_a = TimeTags(
            times=np.array([1.0, 2.0, 3.0], dtype=float),
            is_signal=np.ones(3, dtype=bool),
            basis=np.zeros(3, dtype=np.int8),
            bit=np.zeros(3, dtype=np.int8),
        )
        tags_b = TimeTags(
            times=np.array([1.0, 2.0, 3.0], dtype=float),
            is_signal=np.ones(3, dtype=bool),
            basis=np.zeros(3, dtype=np.int8),
            bit=np.zeros(3, dtype=np.int8),
        )
        result = match_coincidences(tags_a, tags_b, tau_seconds=1e-6)
        assert result.coincidences == 3
        assert result.accidentals == 0
        assert math.isinf(result.car)  # No accidentals

    def test_accidental_coincidences(self):
        """Background-only coincidences should count as accidentals."""
        tags_a = TimeTags(
            times=np.array([1.0, 2.0], dtype=float),
            is_signal=np.array([True, False]),
            basis=np.zeros(2, dtype=np.int8),
            bit=np.zeros(2, dtype=np.int8),
        )
        tags_b = TimeTags(
            times=np.array([1.0, 2.0], dtype=float),
            is_signal=np.array([True, False]),
            basis=np.zeros(2, dtype=np.int8),
            bit=np.zeros(2, dtype=np.int8),
        )
        result = match_coincidences(tags_a, tags_b, tau_seconds=1e-6)
        assert result.coincidences == 1
        assert result.accidentals == 1
        assert result.car == 1.0

    def test_car_computation(self):
        """CAR should be coincidences/accidentals."""
        tags_a = TimeTags(
            times=np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
            is_signal=np.array([True, True, False, False]),
            basis=np.zeros(4, dtype=np.int8),
            bit=np.zeros(4, dtype=np.int8),
        )
        tags_b = TimeTags(
            times=np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
            is_signal=np.array([True, True, False, False]),
            basis=np.zeros(4, dtype=np.int8),
            bit=np.zeros(4, dtype=np.int8),
        )
        result = match_coincidences(tags_a, tags_b, tau_seconds=1e-6)
        assert result.coincidences == 2  # Both signal events
        assert result.accidentals == 2  # Both background events
        assert abs(result.car - 1.0) < 1e-10

    def test_basis_matching(self):
        """Matched basis should count in matrices."""
        tags_a = TimeTags(
            times=np.array([1.0], dtype=float),
            is_signal=np.ones(1, dtype=bool),
            basis=np.array([0], dtype=np.int8),  # Z
            bit=np.array([0], dtype=np.int8),
        )
        tags_b = TimeTags(
            times=np.array([1.0], dtype=float),
            is_signal=np.ones(1, dtype=bool),
            basis=np.array([0], dtype=np.int8),  # Z
            bit=np.array([0], dtype=np.int8),
        )
        result = match_coincidences(tags_a, tags_b, tau_seconds=1e-6)
        assert result.matrices["Z"][0][0] == 1  # Both measured 0 in Z
        assert result.matrices["X"][0][0] == 0

    def test_timing_model_integration(self):
        """Timing model should be applied."""
        tags_a = TimeTags(
            times=np.array([1.0], dtype=float),
            is_signal=np.ones(1, dtype=bool),
            basis=np.zeros(1, dtype=np.int8),
            bit=np.zeros(1, dtype=np.int8),
        )
        tags_b = TimeTags(
            times=np.array([2.0], dtype=float),
            is_signal=np.ones(1, dtype=bool),
            basis=np.zeros(1, dtype=np.int8),
            bit=np.zeros(1, dtype=np.int8),
        )
        model = TimingModel(delta_t=-1.0)  # Shift B back by 1.0
        result = match_coincidences(tags_a, tags_b, tau_seconds=1e-6, timing_model=model)
        # After shift, tags_b.times = 1.0, should coincide
        assert result.coincidences == 1

    def test_car_infinite_no_accidentals(self):
        """CAR should be infinite when no accidentals."""
        tags_a = TimeTags(
            times=np.array([1.0, 2.0, 3.0], dtype=float),
            is_signal=np.ones(3, dtype=bool),
            basis=np.zeros(3, dtype=np.int8),
            bit=np.zeros(3, dtype=np.int8),
        )
        tags_b = TimeTags(
            times=np.array([1.0, 2.0, 3.0], dtype=float),
            is_signal=np.ones(3, dtype=bool),
            basis=np.zeros(3, dtype=np.int8),
            bit=np.zeros(3, dtype=np.int8),
        )
        result = match_coincidences(tags_a, tags_b, tau_seconds=1e-6)
        assert math.isinf(result.car) and result.car > 0


class TestSatLinkParams:
    """Test SatLinkParams dataclass."""

    def test_default_params(self):
        """Default parameters should be reasonable."""
        params = SatLinkParams()
        assert params.wavelength_m > 0
        assert params.altitude_m > 0
        assert params.earth_radius_m > params.altitude_m

    def test_custom_params(self):
        """Custom parameters should be accepted."""
        params = SatLinkParams(
            wavelength_m=1550e-9,
            altitude_m=400e3,
            atmospheric_loss_db_zenith=1.0,
        )
        assert params.wavelength_m == 1550e-9
        assert params.altitude_m == 400e3


class TestSlantRange:
    """Test slant_range_m function."""

    def test_zenith_elevation(self):
        """90 degree elevation should give minimum range."""
        params = SatLinkParams()
        range_zenith = slant_range_m(90.0, params)
        range_high = slant_range_m(45.0, params)
        assert range_zenith < range_high

    def test_horizon_elevation(self):
        """Low elevation should give larger range."""
        params = SatLinkParams()
        range_low = slant_range_m(5.0, params)
        range_mid = slant_range_m(30.0, params)
        assert range_low > range_mid

    def test_clipping(self):
        """Extreme angles should be clipped."""
        params = SatLinkParams()
        # Very low angle should not fail
        result = slant_range_m(0.1, params)
        assert result > params.altitude_m

    def test_monotonic_decrease(self):
        """Range should decrease as elevation increases."""
        params = SatLinkParams()
        ranges = [slant_range_m(el, params) for el in [10, 30, 60, 89]]
        for i in range(len(ranges) - 1):
            assert ranges[i] > ranges[i + 1]


class TestFSPL:
    """Test fspl_db function."""

    def test_zero_range(self):
        """Very small range should give low loss."""
        params = SatLinkParams()
        loss_near = fspl_db(1000, params.wavelength_m)
        loss_far = fspl_db(1e6, params.wavelength_m)
        assert loss_near < loss_far

    def test_scaling(self):
        """Doubling range should increase loss by 6 dB."""
        params = SatLinkParams()
        loss_r = fspl_db(1000, params.wavelength_m)
        loss_2r = fspl_db(2000, params.wavelength_m)
        # 20*log10(2) ≈ 6.02 dB
        assert abs((loss_2r - loss_r) - 6.02) < 0.1

    def test_always_positive(self):
        """FSPL should always be positive."""
        params = SatLinkParams()
        for range_m in [1e3, 1e4, 1e5, 1e6]:
            loss = fspl_db(range_m, params.wavelength_m)
            assert loss > 0


class TestAtmosphericLoss:
    """Test atmospheric_loss_db function."""

    def test_zenith_minimum(self):
        """Zenith should give minimum atmospheric loss."""
        params = SatLinkParams()
        loss_zenith = atmospheric_loss_db(90.0, params)
        loss_high = atmospheric_loss_db(45.0, params)
        loss_low = atmospheric_loss_db(10.0, params)
        assert loss_zenith <= loss_high <= loss_low

    def test_monotonic(self):
        """Loss should increase monotonically toward horizon."""
        params = SatLinkParams()
        losses = [atmospheric_loss_db(el, params) for el in [89, 60, 30, 10, 5]]
        for i in range(len(losses) - 1):
            assert losses[i] <= losses[i + 1]

    def test_bounds(self):
        """Loss should be between zenith and horizon values."""
        params = SatLinkParams()
        for el in [5, 15, 30, 45, 60, 75]:
            loss = atmospheric_loss_db(el, params)
            assert params.atmospheric_loss_db_zenith <= loss <= params.atmospheric_loss_db_horizon

    def test_clipping(self):
        """Extreme angles should be clipped safely."""
        params = SatLinkParams()
        loss_extreme = atmospheric_loss_db(-45.0, params)
        loss_valid = atmospheric_loss_db(90.0, params)
        assert loss_extreme >= loss_valid


class TestTotalChannelLoss:
    """Test total_channel_loss_db function."""

    def test_zenith_lower_than_horizon(self):
        """Zenith should have lower total loss."""
        params = SatLinkParams()
        loss_zenith = total_channel_loss_db(90.0, params)
        loss_horizon = total_channel_loss_db(10.0, params)
        assert loss_zenith < loss_horizon

    def test_positive_loss(self):
        """Loss should always be positive."""
        params = SatLinkParams()
        for el in [5, 15, 30, 45, 60, 90]:
            loss = total_channel_loss_db(el, params)
            assert loss > 0

    def test_includes_all_components(self):
        """Total loss should be sum of components."""
        params = SatLinkParams()
        loss = total_channel_loss_db(45.0, params)
        # Loss should be roughly >= pointing + system (other components positive)
        assert loss >= params.pointing_loss_db + params.system_margin_db

    def test_transmittance_calculation(self):
        """Loss should translate to reasonable transmittance."""
        params = SatLinkParams()
        loss_10db = total_channel_loss_db(60.0, params)
        trans = 10 ** (-loss_10db / 10.0)
        assert 0.0 < trans <= 1.0

    def test_extreme_loss_at_horizon(self):
        """Very low elevation should give high loss."""
        params = SatLinkParams()
        loss = total_channel_loss_db(1.0, params)
        # Should be very high, probably > 30 dB
        assert loss > 30.0


class TestLinkBudgetSequence:
    """Test realistic sequences of link budget operations."""

    def test_typical_leo_pass(self):
        """Typical LEO pass should show realistic loss variation."""
        params = SatLinkParams()
        # Typical pass: 10° rise to 60° zenith to 10° set
        elevations = [10, 20, 30, 45, 60, 45, 30, 20, 10]
        losses = [total_channel_loss_db(el, params) for el in elevations]
        
        # Should be symmetric around zenith
        peak_idx = losses.index(min(losses))
        assert peak_idx == 4  # Middle element

    def test_day_vs_night(self):
        """Different conditions could affect loss."""
        params_day = SatLinkParams(atmospheric_loss_db_zenith=3.0)
        params_night = SatLinkParams(atmospheric_loss_db_zenith=0.5)
        
        loss_day = total_channel_loss_db(45.0, params_day)
        loss_night = total_channel_loss_db(45.0, params_night)
        assert loss_day > loss_night

    def test_parametric_sweep(self):
        """Sweeping parameters should show monotonic behavior."""
        base_params = SatLinkParams()
        
        # Sweep elevation
        elevations = np.linspace(5, 89, 10)
        losses = [total_channel_loss_db(el, base_params) for el in elevations]
        
        # Should be monotonically decreasing
        for i in range(len(losses) - 1):
            assert losses[i] >= losses[i + 1]
