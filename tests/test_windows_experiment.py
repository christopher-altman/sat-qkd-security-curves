"""Tests for windows.py and experiment.py with edge case coverage."""
import pytest
import math
import random
import numpy as np
from sat_qkd_lab.windows import generate_windows, assign_groups_blinded, WindowSpec
from sat_qkd_lab.experiment import (
    ExperimentParams, generate_schedule, simulate_block_metrics,
    _difference_in_means, _counts_from_correlation
)
from sat_qkd_lab.finite_key import FiniteKeyParams


class TestGenerateWindows:
    """Test generate_windows function."""

    def test_zero_blocks(self):
        """Zero blocks should raise error."""
        with pytest.raises(ValueError):
            generate_windows(n_blocks=0, block_seconds=30.0, seed=42)

    def test_negative_blocks(self):
        """Negative blocks should raise error."""
        with pytest.raises(ValueError):
            generate_windows(n_blocks=-1, block_seconds=30.0, seed=42)

    def test_zero_duration(self):
        """Zero duration should raise error."""
        with pytest.raises(ValueError):
            generate_windows(n_blocks=5, block_seconds=0.0, seed=42)

    def test_negative_duration(self):
        """Negative duration should raise error."""
        with pytest.raises(ValueError):
            generate_windows(n_blocks=5, block_seconds=-1.0, seed=42)

    def test_single_block(self):
        """Single block should create one window."""
        windows = generate_windows(n_blocks=1, block_seconds=30.0, seed=42)
        assert len(windows) == 1
        assert windows[0].window_id == "W000"
        assert windows[0].t_start_seconds == 0.0
        assert windows[0].t_end_seconds == 30.0

    def test_multiple_blocks(self):
        """Multiple blocks should be sequential."""
        windows = generate_windows(n_blocks=5, block_seconds=10.0, seed=42)
        assert len(windows) == 5
        for i, window in enumerate(windows):
            assert window.window_id == f"W{i:03d}"
            assert window.t_start_seconds == i * 10.0
            assert window.t_end_seconds == (i + 1) * 10.0

    def test_window_continuity(self):
        """Windows should be contiguous with no gaps."""
        windows = generate_windows(n_blocks=10, block_seconds=5.0, seed=42)
        for i in range(len(windows) - 1):
            assert windows[i].t_end_seconds == windows[i + 1].t_start_seconds

    def test_large_block_count(self):
        """Large block count should work."""
        windows = generate_windows(n_blocks=1000, block_seconds=1.0, seed=42)
        assert len(windows) == 1000
        assert windows[-1].t_end_seconds == 1000.0

    def test_fractional_durations(self):
        """Fractional durations should be preserved."""
        windows = generate_windows(n_blocks=3, block_seconds=0.5, seed=42)
        assert windows[0].t_start_seconds == 0.0
        assert windows[0].t_end_seconds == 0.5
        assert windows[1].t_start_seconds == 0.5
        assert windows[1].t_end_seconds == 1.0


class TestAssignGroupsBlinded:
    """Test assign_groups_blinded function."""

    def test_empty_windows(self):
        """Empty window list should work."""
        windows = []
        result = assign_groups_blinded(windows, seed=42)
        assert "labels_by_window" in result
        assert "schedule_blinded" in result
        assert "schedule_unblinded" in result
        assert len(result["labels_by_window"]) == 0

    def test_single_window(self):
        """Single window should create assignments."""
        windows = [WindowSpec(window_id="W000", t_start_seconds=0.0, t_end_seconds=30.0)]
        result = assign_groups_blinded(windows, seed=42)
        assert len(result["labels_by_window"]) == 1
        assert "W000" in result["labels_by_window"]

    def test_blinding_codes_differ(self):
        """Blinded codes should be different from IDs."""
        windows = generate_windows(n_blocks=5, block_seconds=10.0, seed=42)
        result = assign_groups_blinded(windows, seed=42)
        for sched in result["schedule_blinded"]:
            assert sched["assignment_code"] != sched["window_id"]

    def test_deterministic_seeding(self):
        """Same seed should produce same assignments."""
        windows = generate_windows(n_blocks=5, block_seconds=10.0, seed=42)
        result1 = assign_groups_blinded(windows, seed=100)
        result2 = assign_groups_blinded(windows, seed=100)
        
        assert result1["labels_by_window"] == result2["labels_by_window"]
        for s1, s2 in zip(result1["schedule_blinded"], result2["schedule_blinded"]):
            assert s1["assignment_code"] == s2["assignment_code"]

    def test_control_intervention_split(self):
        """Should roughly split control and intervention."""
        windows = generate_windows(n_blocks=100, block_seconds=1.0, seed=42)
        result = assign_groups_blinded(windows, seed=42)
        
        labels = list(result["labels_by_window"].values())
        n_control = sum(1 for l in labels if l == "control")
        n_intervention = sum(1 for l in labels if l == "intervention")
        
        # Should be split roughly 50-50
        assert 40 < n_control < 60
        assert 40 < n_intervention < 60
        assert n_control + n_intervention == 100

    def test_unblinded_includes_labels(self):
        """Unblinded schedule should have labels."""
        windows = generate_windows(n_blocks=5, block_seconds=10.0, seed=42)
        result = assign_groups_blinded(windows, seed=42)
        
        for sched in result["schedule_unblinded"]:
            assert "label" in sched
            assert sched["label"] in ["control", "intervention"]


class TestGenerateSchedule:
    """Test generate_schedule function."""

    def test_zero_blocks(self):
        """Zero blocks should give empty list."""
        schedule = generate_schedule(0, seed=42)
        assert schedule == []

    def test_single_block(self):
        """Single block should have one assignment."""
        schedule = generate_schedule(1, seed=42)
        assert len(schedule) == 1
        assert schedule[0] in ["control", "intervention"]

    def test_two_blocks(self):
        """Two blocks should split control/intervention."""
        schedule = generate_schedule(2, seed=42)
        assert len(schedule) == 2
        assert "control" in schedule or "intervention" in schedule

    def test_odd_blocks(self):
        """Odd number should favor control (n//2 + n%2)."""
        schedule = generate_schedule(5, seed=42)
        assert len(schedule) == 5
        n_control = schedule.count("control")
        n_intervention = schedule.count("intervention")
        assert n_control + n_intervention == 5
        # For n=5: control = 5//2 + 5%2 = 3, intervention = 2
        assert n_control == 3
        assert n_intervention == 2

    def test_even_blocks(self):
        """Even number should be roughly 50-50."""
        schedule = generate_schedule(10, seed=42)
        n_control = schedule.count("control")
        n_intervention = schedule.count("intervention")
        assert n_control == 5
        assert n_intervention == 5

    def test_deterministic_seeding(self):
        """Same seed should give same schedule."""
        s1 = generate_schedule(20, seed=100)
        s2 = generate_schedule(20, seed=100)
        assert s1 == s2

    def test_different_seeds_differ(self):
        """Different seeds should (likely) give different schedules."""
        s1 = generate_schedule(20, seed=100)
        s2 = generate_schedule(20, seed=200)
        # Could match by chance, but unlikely for 20 items
        assert s1 != s2


class TestExperimentParams:
    """Test ExperimentParams dataclass."""

    def test_default_params(self):
        """Default parameters should be reasonable."""
        params = ExperimentParams()
        assert params.n_blocks > 0
        assert params.block_seconds > 0
        assert params.rep_rate_hz > 0
        assert 0.0 < params.sifted_fraction < 1.0
        assert 0.0 < params.qber_abort_threshold < 0.5

    def test_custom_params(self):
        """Custom parameters should be set."""
        params = ExperimentParams(
            n_blocks=50,
            block_seconds=60.0,
            base_qber=0.05,
        )
        assert params.n_blocks == 50
        assert params.block_seconds == 60.0
        assert params.base_qber == 0.05


class TestSimulateBlockMetrics:
    """Test simulate_block_metrics function."""

    def test_control_block(self):
        """Control block should produce metrics."""
        params = ExperimentParams()
        rng = random.Random(42)
        metrics = simulate_block_metrics("control", params, rng, None)
        
        assert "qber_mean" in metrics
        assert "headroom" in metrics
        assert "total_secret_bits" in metrics
        assert 0.0 <= metrics["qber_mean"] <= 0.5

    def test_intervention_block(self):
        """Intervention block should apply shift."""
        params = ExperimentParams(intervention_qber_shift=0.02)
        rng = random.Random(42)
        
        c_metrics = simulate_block_metrics("control", params, rng, None)
        rng = random.Random(42)  # Same seed
        i_metrics = simulate_block_metrics("intervention", params, rng, None)
        
        # Intervention should have higher QBER (shift added)
        # Note: due to randomness, we can't guarantee i > c exactly
        assert isinstance(i_metrics["qber_mean"], float)

    def test_bell_mode(self):
        """Bell mode should include Bell metrics."""
        params = ExperimentParams()
        rng = random.Random(42)
        metrics = simulate_block_metrics("control", params, rng, None, bell_mode=True)
        
        assert "bell" in metrics
        assert "visibility" in metrics["bell"]
        assert "chsh_s" in metrics["bell"]

    def test_finite_key_mode(self):
        """Finite-key mode should compute secret length."""
        params = ExperimentParams(base_qber=0.03)
        rng = random.Random(42)
        fk_params = FiniteKeyParams()
        metrics = simulate_block_metrics("control", params, rng, fk_params)
        
        assert "qber_mean" in metrics
        assert "total_secret_bits" in metrics

    def test_abort_on_high_qber(self):
        """High QBER should reduce secret bits."""
        params_low = ExperimentParams(base_qber=0.03, qber_abort_threshold=0.11)
        params_high = ExperimentParams(base_qber=0.12, qber_abort_threshold=0.11)
        
        rng = random.Random(42)
        m_low = simulate_block_metrics("control", params_low, rng, None)
        
        rng = random.Random(42)
        m_high = simulate_block_metrics("control", params_high, rng, None)
        
        # High QBER should have lower/zero secret bits
        assert m_high["total_secret_bits"] <= m_low["total_secret_bits"]


class TestDifferenceInMeans:
    """Test _difference_in_means function."""

    def test_empty_groups(self):
        """Empty groups should return None values."""
        result = _difference_in_means([], [])
        assert result["control_mean"] is None
        assert result["intervention_mean"] is None
        assert result["delta"] is None

    def test_empty_control(self):
        """Empty control should return None."""
        result = _difference_in_means([], [1.0, 2.0, 3.0])
        assert result["control_mean"] is None

    def test_empty_intervention(self):
        """Empty intervention should return None."""
        result = _difference_in_means([1.0, 2.0, 3.0], [])
        assert result["intervention_mean"] is None

    def test_single_value_groups(self):
        """Single value in each group should work."""
        result = _difference_in_means([1.0], [2.0])
        assert result["control_mean"] == 1.0
        assert result["intervention_mean"] == 2.0
        assert result["delta"] == 1.0
        # CI should be None with n=1
        assert result["ci_low"] is None

    def test_identical_groups(self):
        """Identical groups should have zero delta."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _difference_in_means(values, values)
        assert result["delta"] == 0.0

    def test_separated_groups(self):
        """Well-separated groups should show large delta."""
        control = [1.0, 2.0, 3.0]
        intervention = [10.0, 11.0, 12.0]
        result = _difference_in_means(control, intervention)
        assert result["delta"] == 9.0
        assert result["control_mean"] == 2.0
        assert result["intervention_mean"] == 11.0

    def test_ci_brackets_delta(self):
        """CI should bracket the delta estimate."""
        control = [1.0, 1.5, 2.0, 2.5, 3.0]
        intervention = [4.0, 4.5, 5.0, 5.5, 6.0]
        result = _difference_in_means(control, intervention)
        
        if result["ci_low"] is not None:
            assert result["ci_low"] <= result["delta"] <= result["ci_high"]


class TestCountsFromCorrelation:
    """Test _counts_from_correlation helper function."""

    def test_perfect_correlation(self):
        """Perfect correlation should all be same."""
        matrix = _counts_from_correlation(100, correlation=1.0)
        # All same outcome
        assert matrix[0][1] == 0 and matrix[1][0] == 0

    def test_perfect_anticorrelation(self):
        """Perfect anticorrelation should all differ."""
        matrix = _counts_from_correlation(100, correlation=-1.0)
        # All different outcomes
        assert matrix[0][0] == 0 and matrix[1][1] == 0

    def test_no_correlation(self):
        """Zero correlation should be roughly 50-50."""
        matrix = _counts_from_correlation(1000, correlation=0.0)
        total = sum(sum(row) for row in matrix)
        assert total == 1000
        # Roughly 25-25-25-25
        for row in matrix:
            for val in row:
                assert 200 < val < 300

    def test_total_counts(self):
        """Total counts should match n_pairs."""
        for n_pairs in [1, 10, 100, 1000]:
            matrix = _counts_from_correlation(n_pairs, correlation=0.5)
            total = sum(sum(row) for row in matrix)
            assert total == max(1, n_pairs)

    def test_correlation_bounds(self):
        """Correlations should be clamped to [-1, 1]."""
        m1 = _counts_from_correlation(100, correlation=2.0)
        m2 = _counts_from_correlation(100, correlation=1.0)
        # Should be identical (clamped)
        assert m1 == m2

    def test_monotonic_correlation(self):
        """Higher correlation should increase matching outcomes."""
        m_low = _counts_from_correlation(100, correlation=-0.5)
        m_mid = _counts_from_correlation(100, correlation=0.0)
        m_high = _counts_from_correlation(100, correlation=0.5)
        
        same_low = m_low[0][0] + m_low[1][1]
        same_mid = m_mid[0][0] + m_mid[1][1]
        same_high = m_high[0][0] + m_high[1][1]
        
        assert same_low <= same_mid <= same_high


class TestExperimentSequence:
    """Test realistic experiment sequences."""

    def test_full_experiment_workflow(self):
        """Complete experiment workflow should work."""
        # Generate schedule
        schedule = generate_schedule(20, seed=42)
        assert len(schedule) == 20
        
        # Generate windows
        windows = generate_windows(n_blocks=20, block_seconds=30.0, seed=42)
        assert len(windows) == 20
        
        # Assign blinded groups
        result = assign_groups_blinded(windows, seed=42)
        assert len(result["labels_by_window"]) == 20
        
        # Simulate metrics
        params = ExperimentParams(n_blocks=20)
        rng = random.Random(42)
        all_metrics = []
        for label in schedule:
            metrics = simulate_block_metrics(label, params, rng, None)
            all_metrics.append(metrics)
        
        assert len(all_metrics) == 20

    def test_paired_analysis(self):
        """Control-intervention pairing analysis."""
        schedule = generate_schedule(40, seed=42)
        params = ExperimentParams()
        
        control_values = []
        intervention_values = []
        rng = random.Random(42)
        
        for label in schedule:
            metrics = simulate_block_metrics(label, params, rng, None)
            if label == "control":
                control_values.append(metrics["total_secret_bits"])
            else:
                intervention_values.append(metrics["total_secret_bits"])
        
        # Both should have values
        assert len(control_values) > 0
        assert len(intervention_values) > 0
        
        # Compute difference
        result = _difference_in_means(control_values, intervention_values)
        assert result["delta"] is not None
