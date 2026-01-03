"""
Tests for engineering output calculations (bps, total bits, required rep rate, headroom).
"""

import math
import pytest
from sat_qkd_lab.sweep import compute_engineering_outputs, compute_headroom


class TestEngineeringOutputs:
    """Test engineering output computations."""

    def test_key_rate_bps_basic(self):
        """Test basic bps calculation."""
        result = compute_engineering_outputs(
            key_rate_per_pulse=0.001,
            rep_rate_hz=1e8,
        )
        assert "key_rate_bps" in result
        assert result["key_rate_bps"] == pytest.approx(1e5)

    def test_total_secret_bits(self):
        """Test total secret bits calculation."""
        result = compute_engineering_outputs(
            key_rate_per_pulse=0.001,
            rep_rate_hz=1e8,
            pass_seconds=300.0,
        )
        assert "key_rate_bps" in result
        assert "total_secret_bits" in result
        assert result["key_rate_bps"] == pytest.approx(1e5)
        assert result["total_secret_bits"] == pytest.approx(3e7)

    def test_required_rep_rate(self):
        """Test required rep rate calculation."""
        result = compute_engineering_outputs(
            key_rate_per_pulse=0.001,
            pass_seconds=300.0,
            target_bits=1_000_000,
        )
        assert "required_rep_rate_hz" in result
        # required = target / (key_rate_per_pulse * pass_seconds)
        #          = 1e6 / (0.001 * 300) = 1e6 / 0.3 = 3.333...e6
        assert result["required_rep_rate_hz"] == pytest.approx(1e6 / 0.3)

    def test_required_rep_rate_zero_key_rate(self):
        """Test required rep rate with zero key rate."""
        result = compute_engineering_outputs(
            key_rate_per_pulse=0.0,
            pass_seconds=300.0,
            target_bits=1_000_000,
        )
        assert "required_rep_rate_hz" in result
        assert result["required_rep_rate_hz"] == "inf"
        assert "required_rep_rate_note" in result

    def test_required_rep_rate_negative_key_rate(self):
        """Test required rep rate with negative key rate (edge case)."""
        result = compute_engineering_outputs(
            key_rate_per_pulse=-0.001,
            pass_seconds=300.0,
            target_bits=1_000_000,
        )
        assert result["required_rep_rate_hz"] == "inf"
        assert "required_rep_rate_note" in result

    def test_no_outputs_when_no_params(self):
        """Test that no outputs are generated when parameters are missing."""
        result = compute_engineering_outputs(key_rate_per_pulse=0.001)
        assert result == {}

    def test_bps_only_without_pass_seconds(self):
        """Test that bps is computed but total bits is not when pass_seconds is missing."""
        result = compute_engineering_outputs(
            key_rate_per_pulse=0.001,
            rep_rate_hz=1e8,
        )
        assert "key_rate_bps" in result
        assert "total_secret_bits" not in result


class TestHeadroomCalculations:
    """Test security headroom calculations."""

    def test_basic_headroom(self):
        """Test basic headroom calculation."""
        result = compute_headroom(qber_mean=0.05, qber_abort=0.11)
        assert "headroom" in result
        assert result["headroom"] == pytest.approx(0.06)

    def test_headroom_with_ci(self):
        """Test headroom with confidence intervals."""
        result = compute_headroom(
            qber_mean=0.05,
            qber_abort=0.11,
            qber_ci_low=0.04,
            qber_ci_high=0.06,
        )
        assert "headroom" in result
        assert "headroom_ci_low" in result
        assert "headroom_ci_high" in result
        # headroom = abort - mean = 0.11 - 0.05 = 0.06
        assert result["headroom"] == pytest.approx(0.06)
        # headroom_ci_low (conservative) = abort - ci_high = 0.11 - 0.06 = 0.05
        assert result["headroom_ci_low"] == pytest.approx(0.05)
        # headroom_ci_high (optimistic) = abort - ci_low = 0.11 - 0.04 = 0.07
        assert result["headroom_ci_high"] == pytest.approx(0.07)

    def test_negative_headroom(self):
        """Test headroom when QBER exceeds abort threshold."""
        result = compute_headroom(qber_mean=0.15, qber_abort=0.11)
        assert "headroom" in result
        # Headroom can be negative (means we've exceeded the abort threshold)
        assert result["headroom"] == pytest.approx(-0.04)

    def test_nan_qber(self):
        """Test headroom with NaN QBER (from aborted runs)."""
        result = compute_headroom(qber_mean=float("nan"), qber_abort=0.11)
        assert "headroom" in result
        assert math.isnan(result["headroom"])

    def test_headroom_default_abort(self):
        """Test headroom with default abort threshold."""
        result = compute_headroom(qber_mean=0.05)
        assert "headroom" in result
        # Default abort is 0.11
        assert result["headroom"] == pytest.approx(0.06)

    def test_ci_only_one_bound(self):
        """Test headroom CI with only one CI bound provided."""
        result = compute_headroom(
            qber_mean=0.05,
            qber_abort=0.11,
            qber_ci_low=0.04,
        )
        assert "headroom_ci_high" in result
        assert "headroom_ci_low" not in result

        result = compute_headroom(
            qber_mean=0.05,
            qber_abort=0.11,
            qber_ci_high=0.06,
        )
        assert "headroom_ci_low" in result
        assert "headroom_ci_high" not in result
