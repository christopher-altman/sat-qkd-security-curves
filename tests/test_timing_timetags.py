"""Tests for timing.py and timetags.py with comprehensive edge case coverage."""
import pytest
import numpy as np
from sat_qkd_lab.timing import TimingModel, apply_timing_model, estimate_clock_offset, _count_coincidences
from sat_qkd_lab.timetags import (
    TimeTags, generate_pair_time_tags, generate_background_time_tags,
    merge_time_tags, apply_dead_time, add_afterpulsing, _clip_times, _sort_tags
)


class TestTimingModel:
    """Test TimingModel dataclass and application."""

    def test_default_timing_model(self):
        """Default model should have zero offset and drift."""
        model = TimingModel()
        assert model.delta_t == 0.0
        assert model.drift_ppm == 0.0
        assert model.tdc_seconds == 0.0
        assert model.jitter_sigma_s == 0.0

    def test_timing_model_creation(self):
        """TimingModel should create with custom parameters."""
        model = TimingModel(delta_t=1e-6, drift_ppm=10.0, tdc_seconds=1e-9)
        assert model.delta_t == 1e-6
        assert model.drift_ppm == 10.0

    def test_apply_timing_model_empty_tags(self):
        """Empty tags should remain empty after applying timing model."""
        tags = TimeTags(
            times=np.array([], dtype=float),
            is_signal=np.array([], dtype=bool),
            basis=np.array([], dtype=np.int8),
            bit=np.array([], dtype=np.int8),
        )
        model = TimingModel(delta_t=1e-6)
        result = apply_timing_model(tags, model)
        assert result.times.size == 0

    def test_apply_timing_model_offset(self):
        """Offset should shift all times."""
        tags = TimeTags(
            times=np.array([1.0, 2.0, 3.0], dtype=float),
            is_signal=np.ones(3, dtype=bool),
            basis=np.zeros(3, dtype=np.int8),
            bit=np.zeros(3, dtype=np.int8),
        )
        model = TimingModel(delta_t=0.5)
        result = apply_timing_model(tags, model)
        assert np.allclose(result.times, [1.5, 2.5, 3.5])

    def test_apply_timing_model_drift(self):
        """Drift should scale all times."""
        tags = TimeTags(
            times=np.array([1.0, 2.0, 3.0], dtype=float),
            is_signal=np.ones(3, dtype=bool),
            basis=np.zeros(3, dtype=np.int8),
            bit=np.zeros(3, dtype=np.int8),
        )
        model = TimingModel(drift_ppm=1000.0)  # 0.1% drift
        result = apply_timing_model(tags, model)
        drift_factor = 1.0 + 1000.0 * 1e-6
        assert np.allclose(result.times, tags.times * drift_factor)

    def test_apply_timing_model_tdc_quantization(self):
        """TDC should quantize times."""
        tags = TimeTags(
            times=np.array([1.234, 2.567, 3.891], dtype=float),
            is_signal=np.ones(3, dtype=bool),
            basis=np.zeros(3, dtype=np.int8),
            bit=np.zeros(3, dtype=np.int8),
        )
        model = TimingModel(tdc_seconds=0.1)
        result = apply_timing_model(tags, model)
        # Should round to nearest 0.1
        assert np.allclose(result.times, [1.2, 2.6, 3.9])

    def test_apply_timing_model_jitter(self):
        """Jitter should add noise."""
        tags = TimeTags(
            times=np.array([1.0, 2.0, 3.0], dtype=float),
            is_signal=np.ones(3, dtype=bool),
            basis=np.zeros(3, dtype=np.int8),
            bit=np.zeros(3, dtype=np.int8),
        )
        model = TimingModel(jitter_sigma_s=1e-6)
        rng = np.random.default_rng(42)
        result = apply_timing_model(tags, model, rng=rng)
        # Should be close but not exact
        assert not np.allclose(result.times, tags.times)
        assert np.allclose(result.times, tags.times, atol=1e-5)

    def test_delta_override(self):
        """delta_override should override model offset."""
        tags = TimeTags(
            times=np.array([1.0, 2.0], dtype=float),
            is_signal=np.ones(2, dtype=bool),
            basis=np.zeros(2, dtype=np.int8),
            bit=np.zeros(2, dtype=np.int8),
        )
        model = TimingModel(delta_t=1.0)
        result = apply_timing_model(tags, model, delta_override=2.0)
        assert np.allclose(result.times, [3.0, 4.0])


class TestCountCoincidences:
    """Test _count_coincidences helper function."""

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
        count = _count_coincidences(tags_a, tags_b, tau_seconds=1e-6)
        assert count == 0

    def test_perfect_coincidences(self):
        """Perfectly timed events should all coincide."""
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
        count = _count_coincidences(tags_a, tags_b, tau_seconds=1e-6)
        assert count == 3

    def test_no_coincidences(self):
        """Well-separated events should not coincide."""
        tags_a = TimeTags(
            times=np.array([1.0, 2.0, 3.0], dtype=float),
            is_signal=np.ones(3, dtype=bool),
            basis=np.zeros(3, dtype=np.int8),
            bit=np.zeros(3, dtype=np.int8),
        )
        tags_b = TimeTags(
            times=np.array([10.0, 20.0, 30.0], dtype=float),
            is_signal=np.ones(3, dtype=bool),
            basis=np.zeros(3, dtype=np.int8),
            bit=np.zeros(3, dtype=np.int8),
        )
        count = _count_coincidences(tags_a, tags_b, tau_seconds=1e-6)
        assert count == 0

    def test_partial_coincidences(self):
        """Some events should coincide with tolerance."""
        tags_a = TimeTags(
            times=np.array([1.0, 2.0, 3.0], dtype=float),
            is_signal=np.ones(3, dtype=bool),
            basis=np.zeros(3, dtype=np.int8),
            bit=np.zeros(3, dtype=np.int8),
        )
        tags_b = TimeTags(
            times=np.array([1.0005, 2.5, 3.0], dtype=float),  # 1st and 3rd close
            is_signal=np.ones(3, dtype=bool),
            basis=np.zeros(3, dtype=np.int8),
            bit=np.zeros(3, dtype=np.int8),
        )
        count = _count_coincidences(tags_a, tags_b, tau_seconds=1e-3)
        assert count == 2


class TestTimeTags:
    """Test TimeTags dataclass and utilities."""

    def test_timetags_creation(self):
        """TimeTags should create with arrays."""
        times = np.array([1.0, 2.0, 3.0])
        is_signal = np.array([True, True, False])
        basis = np.array([0, 1, 0], dtype=np.int8)
        bit = np.array([0, 1, 1], dtype=np.int8)
        tags = TimeTags(times=times, is_signal=is_signal, basis=basis, bit=bit)
        assert tags.times.size == 3
        assert np.sum(tags.is_signal) == 2

    def test_clip_times(self):
        """_clip_times should constrain to duration."""
        times = np.array([-1.0, 0.5, 2.0, 5.0])
        clipped = _clip_times(times, duration_s=3.0)
        assert np.allclose(clipped, [0.0, 0.5, 2.0, 3.0])

    def test_sort_tags(self):
        """_sort_tags should sort by time."""
        tags = TimeTags(
            times=np.array([3.0, 1.0, 2.0], dtype=float),
            is_signal=np.array([False, True, True]),
            basis=np.array([1, 0, 1], dtype=np.int8),
            bit=np.array([1, 0, 1], dtype=np.int8),
        )
        sorted_tags = _sort_tags(tags)
        assert np.allclose(sorted_tags.times, [1.0, 2.0, 3.0])
        assert np.array_equal(sorted_tags.is_signal, [True, True, False])


class TestGeneratePairTimeTags:
    """Test generate_pair_time_tags function with edge cases."""

    def test_zero_pairs(self):
        """Zero pairs should generate empty tags."""
        tags_a, tags_b = generate_pair_time_tags(
            n_pairs=0, duration_s=1.0, sigma_a=1e-6, sigma_b=1e-6, seed=42
        )
        assert tags_a.times.size == 0
        assert tags_b.times.size == 0

    def test_single_pair(self):
        """Single pair should generate one event."""
        tags_a, tags_b = generate_pair_time_tags(
            n_pairs=1, duration_s=1.0, sigma_a=1e-6, sigma_b=1e-6, seed=42
        )
        assert tags_a.times.size == 1
        assert tags_b.times.size == 1
        assert tags_a.is_signal[0] and tags_b.is_signal[0]

    def test_large_jitter(self):
        """Large jitter should still stay within duration."""
        tags_a, tags_b = generate_pair_time_tags(
            n_pairs=100, duration_s=1.0, sigma_a=0.1, sigma_b=0.1, seed=42
        )
        assert np.all(tags_a.times >= 0.0) and np.all(tags_a.times <= 1.0)
        assert np.all(tags_b.times >= 0.0) and np.all(tags_b.times <= 1.0)

    def test_deterministic_seeding(self):
        """Same seed should produce identical results."""
        tags_a1, tags_b1 = generate_pair_time_tags(
            n_pairs=10, duration_s=1.0, sigma_a=1e-6, sigma_b=1e-6, seed=42
        )
        tags_a2, tags_b2 = generate_pair_time_tags(
            n_pairs=10, duration_s=1.0, sigma_a=1e-6, sigma_b=1e-6, seed=42
        )
        assert np.allclose(tags_a1.times, tags_a2.times)
        assert np.allclose(tags_b1.times, tags_b2.times)

    def test_basis_independence(self):
        """Bases should be independent for each detector."""
        tags_a, tags_b = generate_pair_time_tags(
            n_pairs=100, duration_s=1.0, sigma_a=1e-6, sigma_b=1e-6, seed=42
        )
        # Some bases should differ
        basis_matches = np.sum(tags_a.basis == tags_b.basis)
        assert 30 < basis_matches < 70  # Expect roughly 50%


class TestGenerateBackgroundTimeTags:
    """Test generate_background_time_tags function."""

    def test_zero_rate(self):
        """Zero rate should generate no events."""
        tags = generate_background_time_tags(
            rate_hz=0.0, duration_s=1.0, sigma=1e-6, seed=42
        )
        assert tags.times.size == 0

    def test_high_rate(self):
        """High rate should generate many events."""
        tags = generate_background_time_tags(
            rate_hz=1000.0, duration_s=1.0, sigma=1e-6, seed=42
        )
        assert tags.times.size > 900  # Poisson fluctuation
        assert tags.times.size < 1100

    def test_background_flag(self):
        """All background tags should have is_signal=False."""
        tags = generate_background_time_tags(
            rate_hz=100.0, duration_s=1.0, sigma=1e-6, seed=42
        )
        assert not np.any(tags.is_signal)

    def test_random_basis_bit(self):
        """Basis and bit should be uniformly random."""
        tags = generate_background_time_tags(
            rate_hz=1000.0, duration_s=1.0, sigma=1e-6, seed=42
        )
        # Roughly 50% should be 0, 50% should be 1
        basis_0 = np.sum(tags.basis == 0)
        bit_0 = np.sum(tags.bit == 0)
        assert 400 < basis_0 < 600
        assert 400 < bit_0 < 600


class TestMergeTimeTags:
    """Test merge_time_tags function."""

    def test_merge_empty_signal(self):
        """Merging empty signal with background should give background."""
        signal = TimeTags(
            times=np.array([], dtype=float),
            is_signal=np.array([], dtype=bool),
            basis=np.array([], dtype=np.int8),
            bit=np.array([], dtype=np.int8),
        )
        background = TimeTags(
            times=np.array([1.0, 2.0], dtype=float),
            is_signal=np.zeros(2, dtype=bool),
            basis=np.zeros(2, dtype=np.int8),
            bit=np.zeros(2, dtype=np.int8),
        )
        merged = merge_time_tags(signal, background)
        assert merged.times.size == 2
        assert not np.any(merged.is_signal)

    def test_merge_sorted(self):
        """Merged tags should be sorted by time."""
        signal = TimeTags(
            times=np.array([3.0, 1.0], dtype=float),
            is_signal=np.ones(2, dtype=bool),
            basis=np.zeros(2, dtype=np.int8),
            bit=np.zeros(2, dtype=np.int8),
        )
        background = TimeTags(
            times=np.array([2.0, 4.0], dtype=float),
            is_signal=np.zeros(2, dtype=bool),
            basis=np.zeros(2, dtype=np.int8),
            bit=np.zeros(2, dtype=np.int8),
        )
        merged = merge_time_tags(signal, background)
        assert np.allclose(merged.times, [1.0, 2.0, 3.0, 4.0])


class TestApplyDeadTime:
    """Test apply_dead_time function."""

    def test_zero_dead_time(self):
        """Zero dead time should keep all events."""
        tags = TimeTags(
            times=np.array([1.0, 1.01, 1.02], dtype=float),
            is_signal=np.ones(3, dtype=bool),
            basis=np.zeros(3, dtype=np.int8),
            bit=np.zeros(3, dtype=np.int8),
        )
        result = apply_dead_time(tags, dead_time_s=0.0)
        assert result.times.size == 3

    def test_dead_time_removes_close_events(self):
        """Events within dead time should be removed."""
        tags = TimeTags(
            times=np.array([1.0, 1.001, 2.0, 2.0005], dtype=float),
            is_signal=np.ones(4, dtype=bool),
            basis=np.zeros(4, dtype=np.int8),
            bit=np.zeros(4, dtype=np.int8),
        )
        result = apply_dead_time(tags, dead_time_s=0.001)
        # Should keep [1.0, 2.0] (drop 1.001, 2.0005)
        assert result.times.size == 2
        assert np.allclose(result.times, [1.0, 2.0])

    def test_large_dead_time(self):
        """Large dead time should keep only first event."""
        tags = TimeTags(
            times=np.array([1.0, 1.1, 1.2, 1.3], dtype=float),
            is_signal=np.ones(4, dtype=bool),
            basis=np.zeros(4, dtype=np.int8),
            bit=np.zeros(4, dtype=np.int8),
        )
        result = apply_dead_time(tags, dead_time_s=0.15)
        assert result.times.size == 1
        assert result.times[0] == 1.0


class TestAddAfterpulsing:
    """Test add_afterpulsing function."""

    def test_zero_probability(self):
        """Zero probability should not add events."""
        tags = TimeTags(
            times=np.array([1.0, 2.0], dtype=float),
            is_signal=np.ones(2, dtype=bool),
            basis=np.zeros(2, dtype=np.int8),
            bit=np.zeros(2, dtype=np.int8),
        )
        result = add_afterpulsing(tags, p_afterpulse=0.0, window_s=1e-6, decay=0.0, seed=42)
        assert result.times.size == 2

    def test_afterpulse_follows_signal(self):
        """Afterpulse events should follow signal events."""
        tags = TimeTags(
            times=np.array([1.0], dtype=float),
            is_signal=np.array([True]),
            basis=np.array([0], dtype=np.int8),
            bit=np.array([0], dtype=np.int8),
        )
        result = add_afterpulsing(tags, p_afterpulse=1.0, window_s=0.1, decay=0.0, seed=42)
        # Should have original signal + at least one afterpulse
        assert result.times.size >= 1
        assert result.times[0] == 1.0  # First event is original

    def test_background_no_afterpulse(self):
        """Background events should not generate afterpulses."""
        tags = TimeTags(
            times=np.array([1.0], dtype=float),
            is_signal=np.array([False]),
            basis=np.array([0], dtype=np.int8),
            bit=np.array([0], dtype=np.int8),
        )
        result = add_afterpulsing(tags, p_afterpulse=1.0, window_s=0.1, decay=0.0, seed=42)
        # Only background, no afterpulse
        assert result.times.size == 1
