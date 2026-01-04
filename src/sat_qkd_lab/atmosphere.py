"""Atmospheric attenuation models for scenario generation.

IMPORTANT: This module provides SCENARIO GENERATORS, not forecasts or measurements.
These are parameterized models used to explore plausible atmospheric loss conditions
for QKD link planning. They are NOT:
- Meteorological forecasts
- Sensor fusion pipelines
- Validated radiative transfer models
- Real-time weather observations

The models provide attenuation in dB/km, which is then integrated over a slant path
to produce total atmospheric loss for simulation scenarios.
"""

from __future__ import annotations

import math
from typing import Literal

AtmosphereModel = Literal["none", "kruse", "simple_clear_sky"]


def attenuation_db_km(
    model: AtmosphereModel,
    wavelength_nm: float,
    visibility_km: float,
    elevation_deg: float = 45.0,
) -> float:
    """Compute atmospheric attenuation per kilometer (SCENARIO GENERATOR).

    This function returns attenuation in dB per kilometer of optical path.
    It is used for scenario generation, NOT for forecasting or measurement.

    Parameters
    ----------
    model : {"none", "kruse", "simple_clear_sky"}
        Atmospheric model to use:
        - "none": Zero attenuation (no atmosphere)
        - "kruse": Visibility-based empirical model (Kruse approximation)
        - "simple_clear_sky": Constant baseline attenuation
    wavelength_nm : float
        Optical wavelength in nanometers (e.g., 850 for QKD).
    visibility_km : float
        Meteorological visibility in kilometers.
        For kruse model, typical values:
        - >40 km: excellent visibility
        - 20-40 km: good visibility
        - 10-20 km: moderate visibility (haze)
        - <10 km: poor visibility (fog/smog)
    elevation_deg : float, optional
        Elevation angle in degrees (default: 45.0).
        Used by some models for air-mass scaling (currently unused in kruse/simple).

    Returns
    -------
    float
        Attenuation in dB per kilometer.

    Notes
    -----
    This is a SCENARIO GENERATOR. Do NOT interpret outputs as:
    - Meteorological forecasts
    - Validated atmospheric propagation
    - Sensor-fused weather observations

    For total atmospheric loss, integrate over slant path:
        slant_path_km = 1.0 / max(sin(elevation_deg * π / 180), 0.1)
        atmosphere_loss_db = attenuation_db_km(...) * slant_path_km

    References
    ----------
    Kruse model: visibility-based empirical approximation for aerosol scattering.
    See Kim, McArthur, & Korevaar (2001) for typical QKD free-space link models.
    """
    if model == "none":
        return 0.0

    if model == "simple_clear_sky":
        # Simple baseline: ~0.2 dB/km for clear sky at near-IR wavelengths
        # This is a scenario-generation constant, not a measurement
        return 0.2

    if model == "kruse":
        # Kruse visibility-based model
        # Attenuation coefficient (1/km) based on visibility
        # This is an empirical approximation, not a validated atmospheric model
        if visibility_km <= 0:
            raise ValueError(f"visibility_km must be positive, got {visibility_km}")

        # Wavelength in micrometers
        wavelength_um = wavelength_nm / 1000.0

        # Visibility-dependent size distribution parameter
        # q ≈ 1.6 for high visibility (V > 50 km)
        # q ≈ 0.585 * V^(1/3) for 6 km < V < 50 km
        # q ≈ 1.3 for V < 6 km (haze/fog)
        if visibility_km > 50.0:
            q = 1.6
        elif visibility_km > 6.0:
            q = 0.585 * (visibility_km ** (1.0 / 3.0))
        else:
            q = 1.3

        # Attenuation coefficient (1/km) from visibility
        # Beer-Lambert: I = I_0 * exp(-beta * L)
        # visibility defined as range where contrast ratio = 0.02 (2%)
        # beta = 3.91 / V * (lambda / 0.55)^(-q)
        #
        # Reference wavelength 550 nm (visible)
        beta_km = (3.91 / visibility_km) * ((wavelength_um / 0.55) ** (-q))

        # Convert to dB/km: 10 * log10(e) * beta ≈ 4.343 * beta
        attenuation = 4.343 * beta_km
        return float(attenuation)

    raise ValueError(f"Unknown atmosphere model: {model}")


def compute_slant_path_km(elevation_deg: float) -> float:
    """Compute slant-path length through atmosphere (toy air-mass proxy).

    This is a SCENARIO-GENERATION approximation, NOT a validated atmospheric
    propagation model. It provides a simple geometric scaling for atmospheric
    path length as a function of elevation angle.

    Parameters
    ----------
    elevation_deg : float
        Elevation angle in degrees (0 = horizon, 90 = zenith).

    Returns
    -------
    float
        Effective slant-path length in kilometers.

    Notes
    -----
    Uses plane-parallel approximation:
        slant_path_km = 1.0 / max(sin(elevation_deg * π / 180), 0.1)

    This is a TOY approximation. It does NOT account for:
    - Earth curvature
    - Atmospheric refraction
    - Vertical density profiles
    - Actual atmospheric scale height

    Typical values:
    - 90° (zenith): ~1 km
    - 45°: ~1.4 km
    - 30°: ~2 km
    - 10°: ~5.7 km (clamped by min_sin=0.1)
    """
    if not 0 <= elevation_deg <= 90:
        raise ValueError(f"elevation_deg must be in [0, 90], got {elevation_deg}")

    elevation_rad = elevation_deg * math.pi / 180.0
    sin_el = math.sin(elevation_rad)

    # Clamp to avoid division by zero at horizon
    sin_el_clamped = max(sin_el, 0.1)

    # Plane-parallel approximation for atmospheric path
    # Assumes ~1 km effective atmosphere thickness
    slant_path_km = 1.0 / sin_el_clamped
    return float(slant_path_km)


def compute_atmosphere_loss_db(
    model: AtmosphereModel,
    wavelength_nm: float,
    visibility_km: float,
    elevation_deg: float,
) -> float:
    """Compute total atmospheric loss for scenario generation.

    Integrates per-kilometer attenuation over slant path to produce
    total atmospheric loss contribution.

    Parameters
    ----------
    model : {"none", "kruse", "simple_clear_sky"}
        Atmospheric model to use.
    wavelength_nm : float
        Optical wavelength in nanometers.
    visibility_km : float
        Meteorological visibility in kilometers.
    elevation_deg : float
        Elevation angle in degrees.

    Returns
    -------
    float
        Total atmospheric loss in dB.

    Notes
    -----
    This is a SCENARIO GENERATOR. Calculation:
        atmosphere_loss_db = attenuation_db_km(...) * slant_path_km

    This is NOT:
    - A meteorological forecast
    - A validated atmospheric propagation model
    - A sensor-fused weather observation
    """
    if model == "none":
        return 0.0

    atten_db_km = attenuation_db_km(model, wavelength_nm, visibility_km, elevation_deg)
    slant_km = compute_slant_path_km(elevation_deg)

    atmosphere_loss = atten_db_km * slant_km
    return float(atmosphere_loss)
