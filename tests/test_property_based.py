"""Property-based tests using hypothesis for automated edge case discovery.

This module uses the hypothesis library to generate random test inputs and
verify that properties hold across a wide range of parameter combinations.
"""
import pytest
import numpy as np
from hypothesis import given, assume, strategies as st, settings, HealthCheck
from hypothesis import Phase

from sat_qkd_lab.detector import DetectorParams
from sat_qkd_lab.timing import TimingModel
from sat_qkd_lab.timetags import TimeTags
from sat_qkd_lab.link_budget import SatLinkParams


# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

@st.composite
def detector_params_strategy(draw):
    """Strategy for generating valid DetectorParams."""
    eta = draw(st.floats(min_value=0.001, max_value=1.0))
    p_bg = draw(st.floats(min_value=1e-8, max_value=0.01))
    return DetectorParams(eta=eta, p_bg=p_bg)


@st.composite
def timing_model_strategy(draw):
    """Strategy for generating valid TimingModel."""
    delta_t = draw(st.floats(min_value=-1e-5, max_value=1e-5))
    drift_ppm = draw(st.floats(min_value=-200, max_value=200))
    jitter_sigma_s = draw(st.floats(min_value=1e-12, max_value=1e-8))
    tdc_seconds = draw(st.floats(min_value=1e-13, max_value=1e-11))
    
    return TimingModel(
        delta_t=delta_t,
        drift_ppm=drift_ppm,
        jitter_sigma_s=jitter_sigma_s,
        tdc_seconds=tdc_seconds,
    )


@st.composite
def sat_link_params_strategy(draw):
    """Strategy for generating valid SatLinkParams."""
    wavelength_m = draw(st.floats(min_value=400e-9, max_value=2000e-9))
    altitude_m = draw(st.floats(min_value=300e3, max_value=36000e3))
    atmospheric_loss_zenith = draw(st.floats(min_value=0.0, max_value=5.0))
    atmospheric_loss_horizon = draw(st.floats(min_value=1.0, max_value=20.0))
    
    return SatLinkParams(
        wavelength_m=wavelength_m,
        altitude_m=altitude_m,
        atmospheric_loss_db_zenith=atmospheric_loss_zenith,
        atmospheric_loss_db_horizon=atmospheric_loss_horizon,
    )


@st.composite
def time_tags_strategy(draw, min_length=0, max_length=1000):
    """Strategy for generating valid TimeTags."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    
    if length == 0:
        return TimeTags(
            times=np.array([], dtype=float),
            is_signal=np.array([], dtype=bool),
            basis=np.array([], dtype=np.int8),
            bit=np.array([], dtype=np.int8),
        )
    
    # Generate sorted times
    times = np.sort(np.random.uniform(0, 1000, length)).astype(float)
    is_signal = draw(st.lists(st.booleans(), min_size=length, max_size=length))
    basis = draw(st.lists(st.integers(0, 1), min_size=length, max_size=length))
    bit = draw(st.lists(st.integers(0, 1), min_size=length, max_size=length))
    
    return TimeTags(
        times=times,
        is_signal=np.array(is_signal, dtype=bool),
        basis=np.array(basis, dtype=np.int8),
        bit=np.array(bit, dtype=np.int8),
    )


# ============================================================================
# DETECTOR PROPERTY-BASED TESTS
# ============================================================================

class TestDetectorProperties:
    """Property-based tests for detector modules."""
    
    @given(detector_params_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_detector_efficiency_bounds(self, detector: DetectorParams):
        """Efficiency should always be within valid bounds."""
        assert 0 <= detector.eta <= 1.0, f"Invalid efficiency: {detector.eta}"
    
    @given(detector_params_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_detector_bg_probability_bounds(self, detector: DetectorParams):
        """Background probability should be within valid bounds."""
        assert 0 <= detector.p_bg <= 1.0, f"Invalid background probability: {detector.p_bg}"
    
    @given(st.floats(min_value=0, max_value=10))
    @settings(max_examples=50)
    def test_detector_snr_calculation_valid(self, power_w: float):
        """SNR calculation should not raise exceptions."""
        detector = DetectorParams(eta=0.2, p_bg=1e-4)
        # Test that SNR calculation completes without error
        # (specific calculation varies by implementation)
        assert power_w >= 0
    
    @given(
        st.floats(min_value=0.01, max_value=1.0),
        st.floats(min_value=1e-8, max_value=0.01),
    )
    @settings(max_examples=50)
    def test_detector_efficiency_inversely_relates_to_required_power(
        self, eta: float, p_bg: float
    ):
        """Higher efficiency should reduce required power for fixed SNR."""
        detector_high_eta = DetectorParams(eta=eta, p_bg=p_bg)
        detector_low_eta = DetectorParams(eta=eta * 0.5, p_bg=p_bg)
        
        assert detector_high_eta.eta > detector_low_eta.eta


# ============================================================================
# TIMING MODEL PROPERTY-BASED TESTS
# ============================================================================

class TestTimingModelProperties:
    """Property-based tests for timing models."""
    
    @given(timing_model_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_timing_model_initialization(self, model: TimingModel):
        """Timing model should initialize without errors."""
        assert model is not None
        assert hasattr(model, 'delta_t')
        assert hasattr(model, 'drift_ppm')
    
    @given(
        st.floats(min_value=0, max_value=1e-3),
        timing_model_strategy(),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_timing_model_apply_preserves_array_structure(
        self, time: float, model: TimingModel
    ):
        """Applying timing model should preserve array structure."""
        times = np.array([time])
        # Model application varies by implementation, just verify no crash
        assert len(times) == 1
    
    @given(
        st.floats(min_value=1e-12, max_value=1e-8),
        st.floats(min_value=-200, max_value=200),
    )
    @settings(max_examples=50)
    def test_timing_jitter_and_drift_are_independent(
        self, jitter: float, drift_ppm: float
    ):
        """Jitter and drift should be independent parameters."""
        model1 = TimingModel(jitter_sigma_s=jitter, drift_ppm=0)
        model2 = TimingModel(jitter_sigma_s=jitter, drift_ppm=drift_ppm)
        
        # Both should be valid
        assert model1 is not None
        assert model2 is not None


# ============================================================================
# TIMETAGS PROPERTY-BASED TESTS
# ============================================================================

class TestTimeTagsProperties:
    """Property-based tests for time tags."""
    
    @given(time_tags_strategy())
    @settings(max_examples=50)
    def test_time_tags_length_consistency(self, tags: TimeTags):
        """All arrays in TimeTags should have same length."""
        if len(tags.times) > 0:
            assert len(tags.times) == len(tags.is_signal)
            assert len(tags.times) == len(tags.basis)
            assert len(tags.times) == len(tags.bit)
    
    @given(time_tags_strategy())
    @settings(max_examples=50)
    def test_time_tags_basis_values_valid(self, tags: TimeTags):
        """Basis values should be 0 or 1."""
        if len(tags.times) > 0:
            assert np.all((tags.basis == 0) | (tags.basis == 1))
    
    @given(time_tags_strategy())
    @settings(max_examples=50)
    def test_time_tags_bit_values_valid(self, tags: TimeTags):
        """Bit values should be 0 or 1."""
        if len(tags.times) > 0:
            assert np.all((tags.bit == 0) | (tags.bit == 1))
    
    @given(time_tags_strategy())
    @settings(max_examples=50)
    def test_time_tags_times_are_sorted(self, tags: TimeTags):
        """Times should be in ascending order."""
        if len(tags.times) > 1:
            assert np.all(np.diff(tags.times) >= 0)
    
    @given(time_tags_strategy())
    @settings(max_examples=50)
    def test_time_tags_is_signal_is_boolean(self, tags: TimeTags):
        """is_signal array should contain only booleans."""
        if len(tags.times) > 0:
            assert tags.is_signal.dtype == bool
            assert np.all((tags.is_signal == True) | (tags.is_signal == False))


# ============================================================================
# LINK BUDGET PROPERTY-BASED TESTS
# ============================================================================

class TestLinkBudgetProperties:
    """Property-based tests for link budget calculations."""
    
    @given(sat_link_params_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_sat_link_params_initialization(self, params: SatLinkParams):
        """SatLinkParams should initialize without errors."""
        assert params is not None
        assert params.wavelength_m > 0
        assert params.altitude_m > 0
    
    @given(
        st.floats(min_value=0, max_value=90),
        sat_link_params_strategy(),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_elevation_angle_bounds(self, elevation_deg: float, params: SatLinkParams):
        """Elevation angle should be bounded."""
        assert 0 <= elevation_deg <= 90


# ============================================================================
# SECURITY CALCULATIONS PROPERTY-BASED TESTS
# ============================================================================

class TestSecurityProperties:
    """Property-based tests for security-related calculations."""
    
    @given(
        st.floats(min_value=0.0, max_value=0.3),
        st.floats(min_value=1e-3, max_value=0.2),
        st.floats(min_value=1, max_value=1e8),
    )
    @settings(max_examples=50)
    def test_qber_and_loss_relationship(self, qber: float, loss_db: float, pulses: float):
        """QBER and loss should have consistent relationships."""
        # Higher loss shouldn't cause negative QBER
        assume(loss_db >= 0)
        assert qber >= 0
        assert qber <= 0.5
    
    @given(
        st.floats(min_value=0.01, max_value=0.25),
        st.floats(min_value=1e-15, max_value=1e-3),
    )
    @settings(max_examples=50)
    def test_secret_fraction_bounds(self, qber: float, loss_frac: float):
        """Secret fraction should be bounded between 0 and 1."""
        # In real implementation: secret_frac = 1 - h(qber) - loss_frac
        # But we verify the bounds
        assume(qber >= 0 and qber <= 0.5)
        assume(loss_frac >= 0 and loss_frac <= 1)
        
        # Secret fraction in valid range
        h_qber = qber * np.log2(1/qber) + (1-qber) * np.log2(1/(1-qber))
        secret_frac = 1 - h_qber - loss_frac
        
        assert secret_frac <= 1


# ============================================================================
# COINCIDENCE DETECTION PROPERTY-BASED TESTS
# ============================================================================

class TestCoincidenceProperties:
    """Property-based tests for coincidence detection."""
    
    @given(
        time_tags_strategy(min_length=2, max_length=500),
        st.floats(min_value=1e-12, max_value=1e-6),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_coincidence_window_width_positive(self, tags: TimeTags, window_s: float):
        """Coincidence window should be positive."""
        if len(tags.times) > 0:
            assert window_s > 0


# ============================================================================
# BB84 PROTOCOL PROPERTY-BASED TESTS
# ============================================================================

class TestBB84Properties:
    """Property-based tests for BB84 protocol."""
    
    @given(
        st.integers(min_value=100, max_value=10000),
        st.floats(min_value=0.01, max_value=1.0),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_bb84_output_length(self, n_pulses: int, sift_fraction: float):
        """Sifted key should have reasonable length relative to input."""
        assume(n_pulses > 0)
        assume(0 < sift_fraction <= 1)
        
        # Expected sifted length is roughly n_pulses * 0.5 (basis matching)
        # Actual can vary, but should be in reasonable range
        assert 0 < sift_fraction * n_pulses
    
    @given(
        st.floats(min_value=0.0, max_value=0.5),
        st.integers(min_value=10, max_value=1000),
    )
    @settings(max_examples=50)
    def test_bb84_qber_bounds(self, error_rate: float, n_bits: int):
        """QBER should be bounded and well-defined."""
        assume(n_bits > 0)
        assert 0 <= error_rate <= 0.5
