"""Tests for helpers.py - Input validation and utility functions."""
import pytest
import math
from sat_qkd_lab.helpers import (
    validate_int, validate_float, validate_seed, h2, RunSummary
)


class TestValidateInt:
    """Test validate_int function."""

    def test_valid_int(self):
        """Valid integers should pass."""
        assert validate_int("test", 5) == 5
        assert validate_int("test", -10) == -10
        assert validate_int("test", 0) == 0

    def test_float_as_int(self):
        """Floats representing integers should convert."""
        assert validate_int("test", 5.0) == 5
        assert validate_int("test", 0.0) == 0

    def test_float_not_as_int(self):
        """Floats not representing integers should raise."""
        with pytest.raises(ValueError):
            validate_int("test", 5.5)
        with pytest.raises(ValueError):
            validate_int("test", 5.1)

    def test_min_bound(self):
        """Min bound should be enforced."""
        assert validate_int("test", 5, min_value=3) == 5
        with pytest.raises(ValueError):
            validate_int("test", 2, min_value=3)
        assert validate_int("test", 3, min_value=3) == 3

    def test_max_bound(self):
        """Max bound should be enforced."""
        assert validate_int("test", 5, max_value=10) == 5
        with pytest.raises(ValueError):
            validate_int("test", 11, max_value=10)
        assert validate_int("test", 10, max_value=10) == 10

    def test_both_bounds(self):
        """Both bounds should work together."""
        assert validate_int("test", 5, min_value=3, max_value=10) == 5
        with pytest.raises(ValueError):
            validate_int("test", 2, min_value=3, max_value=10)
        with pytest.raises(ValueError):
            validate_int("test", 11, min_value=3, max_value=10)

    def test_bool_rejection(self):
        """Booleans should be rejected even though they're int-like."""
        with pytest.raises(ValueError):
            validate_int("test", True)
        with pytest.raises(ValueError):
            validate_int("test", False)

    def test_string_rejection(self):
        """Strings should raise."""
        with pytest.raises(ValueError):
            validate_int("test", "5")

    def test_none_rejection(self):
        """None should raise."""
        with pytest.raises(ValueError):
            validate_int("test", None)


class TestValidateFloat:
    """Test validate_float function."""

    def test_valid_float(self):
        """Valid floats should pass."""
        assert validate_float("test", 5.5) == 5.5
        assert validate_float("test", -10.1) == -10.1
        assert validate_float("test", 0.0) == 0.0

    def test_int_as_float(self):
        """Ints should convert to float."""
        assert validate_float("test", 5) == 5.0
        assert validate_float("test", 0) == 0.0

    def test_min_bound(self):
        """Min bound should be enforced."""
        assert validate_float("test", 5.5, min_value=3.0) == 5.5
        with pytest.raises(ValueError):
            validate_float("test", 2.5, min_value=3.0)
        assert validate_float("test", 3.0, min_value=3.0) == 3.0

    def test_max_bound(self):
        """Max bound should be enforced."""
        assert validate_float("test", 5.5, max_value=10.0) == 5.5
        with pytest.raises(ValueError):
            validate_float("test", 10.1, max_value=10.0)
        assert validate_float("test", 10.0, max_value=10.0) == 10.0

    def test_nan_default_rejection(self):
        """NaN should be rejected by default."""
        with pytest.raises(ValueError):
            validate_float("test", float("nan"))

    def test_nan_allowed(self):
        """NaN should be allowed with flag."""
        result = validate_float("test", float("nan"), allow_nan=True)
        assert math.isnan(result)

    def test_inf_default_rejection(self):
        """Infinity should be rejected by default."""
        with pytest.raises(ValueError):
            validate_float("test", float("inf"))
        with pytest.raises(ValueError):
            validate_float("test", float("-inf"))

    def test_inf_allowed(self):
        """Infinity should be allowed with flag."""
        result = validate_float("test", float("inf"), allow_inf=True)
        assert math.isinf(result)

    def test_bool_rejection(self):
        """Booleans should be rejected."""
        with pytest.raises(ValueError):
            validate_float("test", True)

    def test_string_rejection(self):
        """Strings should raise."""
        with pytest.raises(ValueError):
            validate_float("test", "5.5")

    def test_none_rejection(self):
        """None should raise."""
        with pytest.raises(ValueError):
            validate_float("test", None)


class TestValidateSeed:
    """Test validate_seed function."""

    def test_none_seed(self):
        """None should pass."""
        assert validate_seed(None) is None

    def test_valid_seed(self):
        """Valid non-negative integers should pass."""
        assert validate_seed(0) == 0
        assert validate_seed(42) == 42
        assert validate_seed(2**31) == 2**31

    def test_negative_seed(self):
        """Negative seeds should raise."""
        with pytest.raises(ValueError):
            validate_seed(-1)

    def test_float_seed(self):
        """Float seeds should raise."""
        with pytest.raises(ValueError):
            validate_seed(42.0)

    def test_bool_seed(self):
        """Boolean seeds should raise."""
        with pytest.raises(ValueError):
            validate_seed(True)

    def test_string_seed(self):
        """String seeds should raise."""
        with pytest.raises(ValueError):
            validate_seed("42")


class TestBinaryEntropy:
    """Test h2 (binary entropy) function."""

    def test_h2_zero(self):
        """h2(0) should be 0."""
        assert abs(h2(0.0)) < 1e-10

    def test_h2_one(self):
        """h2(1) should be 0."""
        assert abs(h2(1.0)) < 1e-10

    def test_h2_half(self):
        """h2(0.5) should be 1."""
        assert abs(h2(0.5) - 1.0) < 1e-10

    def test_h2_symmetry(self):
        """h2(p) should equal h2(1-p)."""
        for p in [0.1, 0.3, 0.4, 0.6, 0.7, 0.9]:
            assert abs(h2(p) - h2(1.0 - p)) < 1e-10

    def test_h2_bounds(self):
        """h2(p) should be in [0, 1]."""
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            assert 0.0 <= h2(p) <= 1.0

    def test_h2_monotonic(self):
        """h2 should increase from 0 to 0.5 then decrease."""
        vals = [h2(p) for p in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]]
        # Should increase to peak at 0.5
        assert vals[3] >= vals[0]  # h2(0.5) >= h2(0)
        assert vals[3] >= vals[1]  # h2(0.5) >= h2(0.1)
        assert vals[3] >= vals[2]  # h2(0.5) >= h2(0.3)

    def test_h2_extreme_values(self):
        """h2 should handle values near 0 and 1."""
        assert abs(h2(1e-12)) < 0.1
        assert abs(h2(1.0 - 1e-12)) < 0.1
        assert abs(h2(0.5) - 1.0) < 1e-10


class TestRunSummary:
    """Test RunSummary dataclass."""

    def test_run_summary_creation(self):
        """RunSummary should create with required fields."""
        summary = RunSummary(
            n_sent=1000,
            n_received=500,
            n_sifted=250,
            qber=0.05,
            secret_fraction=0.4,
            n_secret_est=100,
            aborted=False,
            meta={"test": "value"},
        )
        assert summary.n_sent == 1000
        assert summary.n_received == 500
        assert summary.qber == 0.05

    def test_run_summary_key_rate_per_pulse(self):
        """key_rate_per_pulse should combine detection, sifting, and secret."""
        summary = RunSummary(
            n_sent=1000,
            n_received=500,
            n_sifted=250,
            qber=0.05,
            secret_fraction=0.5,
            n_secret_est=125,
            aborted=False,
            meta={},
        )
        # (250 / 1000) * 0.5 = 0.125
        expected = (250.0 / 1000.0) * 0.5
        assert abs(summary.key_rate_per_pulse - expected) < 1e-10

    def test_run_summary_aborted_zero_rate(self):
        """Aborted run should have zero key rate."""
        summary = RunSummary(
            n_sent=1000,
            n_received=0,
            n_sifted=0,
            qber=float("nan"),
            secret_fraction=0.0,
            n_secret_est=0,
            aborted=True,
            meta={},
        )
        assert summary.key_rate_per_pulse == 0.0

    def test_run_summary_no_pulses_zero_rate(self):
        """Zero pulses should give zero rate."""
        summary = RunSummary(
            n_sent=0,
            n_received=0,
            n_sifted=0,
            qber=float("nan"),
            secret_fraction=0.0,
            n_secret_est=0,
            aborted=False,
            meta={},
        )
        assert summary.key_rate_per_pulse == 0.0

    def test_run_summary_immutable(self):
        """RunSummary should be frozen (immutable)."""
        summary = RunSummary(
            n_sent=1000,
            n_received=500,
            n_sifted=250,
            qber=0.05,
            secret_fraction=0.4,
            n_secret_est=100,
            aborted=False,
            meta={"test": "value"},
        )
        with pytest.raises(Exception):  # Frozen dataclass raises FrozenInstanceError
            summary.n_sent = 2000
