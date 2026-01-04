"""Tests for atmospheric attenuation models (scenario generation)."""

from __future__ import annotations

import pytest
import math

from sat_qkd_lab.atmosphere import (
    attenuation_db_km,
    compute_slant_path_km,
    compute_atmosphere_loss_db,
)


def test_attenuation_none_model():
    """Test that 'none' model returns zero attenuation."""
    atten = attenuation_db_km("none", 850.0, 23.0, 45.0)
    assert atten == 0.0


def test_attenuation_simple_clear_sky():
    """Test that 'simple_clear_sky' returns constant baseline."""
    atten = attenuation_db_km("simple_clear_sky", 850.0, 23.0, 45.0)
    assert atten == 0.2  # Constant baseline


def test_attenuation_kruse_positive():
    """Test that kruse model returns positive attenuation."""
    atten = attenuation_db_km("kruse", 850.0, 23.0, 45.0)
    assert atten >= 0.0


def test_attenuation_kruse_visibility_monotonic():
    """Test that lower visibility gives higher attenuation (kruse)."""
    vis_high = 40.0  # km, excellent visibility
    vis_low = 10.0   # km, moderate haze

    atten_high = attenuation_db_km("kruse", 850.0, vis_high, 45.0)
    atten_low = attenuation_db_km("kruse", 850.0, vis_low, 45.0)

    # Lower visibility should give higher attenuation
    assert atten_low > atten_high


def test_attenuation_kruse_invalid_visibility():
    """Test that kruse model raises error for invalid visibility."""
    with pytest.raises(ValueError, match="visibility_km must be positive"):
        attenuation_db_km("kruse", 850.0, -1.0, 45.0)

    with pytest.raises(ValueError, match="visibility_km must be positive"):
        attenuation_db_km("kruse", 850.0, 0.0, 45.0)


def test_attenuation_unknown_model():
    """Test that unknown model raises error."""
    with pytest.raises(ValueError, match="Unknown atmosphere model"):
        attenuation_db_km("bogus_model", 850.0, 23.0, 45.0)


def test_slant_path_zenith():
    """Test slant path at zenith (90 degrees)."""
    slant = compute_slant_path_km(90.0)
    assert abs(slant - 1.0) < 1e-6  # Should be ~1 km at zenith


def test_slant_path_45_degrees():
    """Test slant path at 45 degrees."""
    slant = compute_slant_path_km(45.0)
    expected = 1.0 / math.sin(45.0 * math.pi / 180.0)  # ~1.414 km
    assert abs(slant - expected) < 1e-6


def test_slant_path_low_elevation():
    """Test slant path at low elevation (clamped)."""
    # At 5 degrees, sin(5°) ~ 0.087, but we clamp to 0.1
    slant = compute_slant_path_km(5.0)
    assert slant == 10.0  # 1.0 / 0.1


def test_slant_path_invalid_elevation():
    """Test that invalid elevation raises error."""
    with pytest.raises(ValueError, match="elevation_deg must be in"):
        compute_slant_path_km(-10.0)

    with pytest.raises(ValueError, match="elevation_deg must be in"):
        compute_slant_path_km(100.0)


def test_compute_atmosphere_loss_none():
    """Test total atmosphere loss with 'none' model."""
    loss = compute_atmosphere_loss_db("none", 850.0, 23.0, 45.0)
    assert loss == 0.0


def test_compute_atmosphere_loss_simple_clear_sky():
    """Test total atmosphere loss with 'simple_clear_sky' model."""
    loss = compute_atmosphere_loss_db("simple_clear_sky", 850.0, 23.0, 45.0)
    # attenuation = 0.2 dB/km
    # slant_path at 45° ≈ 1.414 km
    # total loss ≈ 0.2 * 1.414 ≈ 0.283 dB
    assert loss > 0.0
    assert loss < 1.0  # Should be less than 1 dB for clear sky at 45°


def test_compute_atmosphere_loss_kruse():
    """Test total atmosphere loss with 'kruse' model."""
    loss = compute_atmosphere_loss_db("kruse", 850.0, 23.0, 45.0)
    assert loss >= 0.0


def test_atmosphere_loss_increases_with_lower_elevation():
    """Test that atmosphere loss increases at lower elevations (longer path)."""
    loss_high = compute_atmosphere_loss_db("simple_clear_sky", 850.0, 23.0, 70.0)
    loss_low = compute_atmosphere_loss_db("simple_clear_sky", 850.0, 23.0, 20.0)

    # Lower elevation = longer slant path = more atmospheric loss
    assert loss_low > loss_high


def test_atmosphere_loss_kruse_visibility_effect():
    """Test that kruse model shows visibility effect on total loss."""
    # Good visibility
    loss_good = compute_atmosphere_loss_db("kruse", 850.0, 40.0, 45.0)

    # Poor visibility (haze)
    loss_poor = compute_atmosphere_loss_db("kruse", 850.0, 10.0, 45.0)

    # Poor visibility should give higher atmospheric loss
    assert loss_poor > loss_good
