"""
Satellite Link Budget Model (Option A: Scenario Generator)

IMPORTANT DISCLAIMER
--------------------
This module provides a SCENARIO GENERATOR for mapping satellite elevation
angles to channel loss values. It is NOT a physically accurate optical
link budget model.

The purpose is to create plausible loss vs. elevation curves for
demonstrating QKD security analysis concepts, NOT to predict actual
satellite-to-ground link performance.

What This Module Does
---------------------
- Maps elevation angle (degrees) to a total channel loss (dB)
- Combines geometric effects (slant range) with empirical loss factors
- Produces monotonically increasing loss as elevation decreases

What This Module Does NOT Do
----------------------------
- It does NOT correctly model free-space optical coupling (FSPL formula
  is for RF/radio waves, not for focused optical beams with apertures)
- It does NOT include beam divergence and aperture coupling geometry
- It does NOT model atmospheric turbulence, scintillation, or beam wander
- It does NOT account for pointing/tracking error distributions
- It does NOT validate against real satellite QKD link measurements

For Accurate Optical Link Budgets
---------------------------------
See optical_link.py for a stub module documenting what a proper
optical link budget would require (Option B, future work). Real satellite
QKD link budgets require:
  - Geometric optics: divergence, aperture, diffraction
  - Pointing loss: jitter, tracking error, servo bandwidth
  - Atmospheric: extinction, turbulence, Cn2 profiles
  - Validated against Micius or other satellite QKD data

References
----------
For demonstration purposes, the loss model here is loosely inspired by:
- Liao et al., Nature 549, 43 (2017) - Micius satellite QKD
- Bourgoin et al., NJP 15, 023006 (2013) - LEO QKD analysis

But the specific formulas are simplified for pedagogical clarity,
not engineering accuracy.
"""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SatLinkParams:
    """
    Parameters for the satellite link scenario generator.

    These produce plausible-looking curves but are NOT validated against
    real satellite optical links. See module docstring for details.
    """
    wavelength_m: float = 1550e-9      # 1550 nm (telecom/free-space optical common)
    altitude_m: float = 550e3          # LEO-ish
    earth_radius_m: float = 6371e3
    atmospheric_loss_db_zenith: float = 2.0   # Placeholder zenith attenuation
    atmospheric_loss_db_horizon: float = 12.0  # Placeholder horizon attenuation
    pointing_loss_db: float = 2.0      # Lumped pointing/tracking/optics loss
    system_margin_db: float = 3.0      # Detector coupling + misc losses


def slant_range_m(elevation_deg: float, p: SatLinkParams) -> float:
    """
    Compute slant range from ground station to satellite.

    This geometric calculation is correct. The slant range increases
    as elevation decreases (satellite closer to horizon).

    Parameters
    ----------
    elevation_deg : float
        Elevation angle above horizon in degrees.
    p : SatLinkParams
        Link parameters containing altitude and Earth radius.

    Returns
    -------
    float
        Slant range in meters.
    """
    el = math.radians(max(1e-6, min(89.999, elevation_deg)))
    Re = p.earth_radius_m
    h = p.altitude_m
    # Geometric slant range formula
    return math.sqrt((Re + h) ** 2 - (Re * math.cos(el)) ** 2) - Re * math.sin(el)


def fspl_db(range_m: float, wavelength_m: float) -> float:
    """
    Free-space path loss formula (RF/radio convention).

    WARNING: This formula applies to isotropic radiators in the far field
    and is used here only to produce a range-dependent loss component.
    It does NOT correctly model focused optical beam coupling to an
    aperture, which requires geometric optics (see optical_link.py).

    For optical links, the correct approach is:
    - Compute beam radius at range from divergence angle
    - Compute coupling efficiency to receiver aperture
    - This gives very different scaling than FSPL

    Parameters
    ----------
    range_m : float
        Distance in meters.
    wavelength_m : float
        Wavelength in meters.

    Returns
    -------
    float
        Path loss in dB (always positive).
    """
    return 20.0 * math.log10(4.0 * math.pi * range_m / wavelength_m)


def atmospheric_loss_db(elevation_deg: float, p: SatLinkParams) -> float:
    """
    Toy model for atmospheric loss vs elevation.

    This linearly interpolates between zenith and horizon loss values.
    Real atmospheric models would use airmass calculations and account
    for aerosols, molecular extinction, and wavelength-dependent effects.

    Parameters
    ----------
    elevation_deg : float
        Elevation angle in degrees.
    p : SatLinkParams
        Link parameters with atmospheric loss values.

    Returns
    -------
    float
        Atmospheric loss in dB.
    """
    el = max(0.0, min(90.0, elevation_deg))
    t = (90.0 - el) / 90.0  # t=0 at zenith, t=1 at horizon
    return (1 - t) * p.atmospheric_loss_db_zenith + t * p.atmospheric_loss_db_horizon


def total_channel_loss_db(elevation_deg: float, p: SatLinkParams) -> float:
    """
    Compute total channel loss for a given elevation angle.

    This combines range-dependent loss (using FSPL as proxy), atmospheric
    loss, pointing loss, and system margin into a single loss value.

    NOTE: This is a SCENARIO GENERATOR producing plausible loss values
    for QKD security curve demonstration. It is NOT a validated optical
    link budget. See module docstring and optical_link.py for details.

    Parameters
    ----------
    elevation_deg : float
        Satellite elevation angle above horizon in degrees.
    p : SatLinkParams
        Link parameters.

    Returns
    -------
    float
        Total channel loss in dB.
    """
    r = slant_range_m(elevation_deg, p)
    return (
        fspl_db(r, p.wavelength_m)
        + atmospheric_loss_db(elevation_deg, p)
        + p.pointing_loss_db
        + p.system_margin_db
    )
