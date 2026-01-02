"""
Optical Link Budget Model (Option B) - STUB MODULE

This module is a placeholder for a future physically-accurate optical link
budget model. It is NOT currently integrated into the main simulation.

The current link_budget.py module (Option A) provides a scenario generator
that maps elevation angles to loss values for demonstration purposes, but
does NOT implement a physically correct optical link budget.

Future Option B Implementation
------------------------------
A proper free-space optical link budget requires:

1. Beam Divergence and Aperture Coupling
   - Transmitter beam divergence angle (typically ~10-20 µrad for LEO QKD)
   - Receiver aperture diameter (typically 0.5-1.5 m for ground stations)
   - Geometric coupling efficiency: (D_rx / (2 * theta * L))^2
     where D_rx is receiver diameter, theta is half-angle divergence, L is range

2. Pointing Loss
   - Pointing error distribution (typically Rayleigh with RMS ~1-5 µrad)
   - Jitter-induced coupling loss (Gaussian beam approximation)
   - Tracking bandwidth effects

3. Atmospheric Effects
   - Turbulence-induced scintillation (Rytov variance)
   - Beam wander and spreading
   - Fried parameter (r0) and isoplanatic angle
   - Adaptive optics correction factor

4. Extinction and Scattering
   - Rayleigh scattering (altitude-dependent)
   - Mie scattering (aerosols, clouds)
   - Molecular absorption (wavelength-dependent)

5. Background Noise
   - Solar/lunar scattered light
   - Atmospheric emission
   - City lights and other sources

References
----------
- Bourgoin, J.-P., et al. (2013). A comprehensive design and performance
  analysis of low Earth orbit satellite quantum communication. NJP 15, 023006.
- Liao, S.-K., et al. (2017). Satellite-to-ground quantum key distribution.
  Nature 549, 43-47.
- Bedington, R., et al. (2017). Progress in satellite quantum key distribution.
  npj Quantum Information 3, 30.

Usage
-----
This module is not integrated. To use it in the future:

1. Implement the functions below with proper physics models
2. Add a CLI flag like `--link-model optical` to run.py
3. Replace the loss calculation in sweeps with optical_link_loss_db()
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math


@dataclass(frozen=True)
class OpticalLinkParams:
    """
    Parameters for a physically-accurate optical link budget.

    STUB: These parameters are defined but not yet used in calculations.
    """
    # Transmitter
    wavelength_m: float = 850e-9           # QKD wavelength (850 nm typical for Si SPAD)
    tx_divergence_urad: float = 15.0       # Half-angle beam divergence
    tx_power_dbm: float = 0.0              # Transmitter power (not used for photon counting)

    # Receiver
    rx_aperture_m: float = 1.0             # Ground station aperture diameter
    rx_fov_urad: float = 100.0             # Receiver field of view

    # Pointing
    pointing_rms_urad: float = 2.0         # RMS pointing error
    tracking_bandwidth_hz: float = 100.0   # Tracking servo bandwidth

    # Orbit
    altitude_m: float = 500e3              # Satellite altitude (LEO)
    earth_radius_m: float = 6371e3

    # Atmosphere (placeholder)
    visibility_km: float = 23.0            # Meteorological visibility
    cn2_ground: float = 1e-14              # Refractive index structure constant at ground


def geometric_coupling_loss_db(
    range_m: float,
    divergence_urad: float,
    aperture_m: float,
) -> float:
    """
    Compute geometric coupling loss for Gaussian beam.

    STUB: Simplified model, does not include diffraction or truncation.

    Parameters
    ----------
    range_m : float
        Slant range to satellite in meters.
    divergence_urad : float
        Half-angle beam divergence in microradians.
    aperture_m : float
        Receiver aperture diameter in meters.

    Returns
    -------
    float
        Geometric loss in dB (always positive = loss).
    """
    # Beam radius at range
    theta_rad = divergence_urad * 1e-6
    beam_radius = range_m * theta_rad

    # Fraction of power captured by aperture (for Gaussian beam)
    # P_captured / P_total = 1 - exp(-2 * (a/w)^2) where a = aperture radius, w = beam radius
    a = aperture_m / 2
    w = beam_radius

    if w <= 0:
        return 0.0

    coupling = 1.0 - math.exp(-2.0 * (a / w) ** 2)
    coupling = max(1e-10, coupling)  # Avoid log(0)

    return -10.0 * math.log10(coupling)


def pointing_loss_db(
    pointing_rms_urad: float,
    divergence_urad: float,
) -> float:
    """
    Compute average pointing loss for Gaussian beam with Rayleigh pointing error.

    STUB: Simplified model.

    Parameters
    ----------
    pointing_rms_urad : float
        RMS pointing error in microradians.
    divergence_urad : float
        Half-angle beam divergence in microradians.

    Returns
    -------
    float
        Pointing loss in dB.
    """
    if divergence_urad <= 0:
        return 0.0

    # For Rayleigh-distributed pointing error with RMS sigma,
    # average loss factor is approximately exp(-(sigma/theta)^2)
    ratio = pointing_rms_urad / divergence_urad
    avg_coupling = math.exp(-(ratio ** 2))
    avg_coupling = max(1e-10, avg_coupling)

    return -10.0 * math.log10(avg_coupling)


def atmospheric_extinction_db(
    elevation_deg: float,
    wavelength_m: float,
    visibility_km: float = 23.0,
) -> float:
    """
    Compute atmospheric extinction loss.

    STUB: Very simplified model. Real models need MODTRAN or similar.

    Parameters
    ----------
    elevation_deg : float
        Elevation angle in degrees.
    wavelength_m : float
        Wavelength in meters.
    visibility_km : float
        Meteorological visibility in km.

    Returns
    -------
    float
        Atmospheric extinction in dB.
    """
    # Airmass approximation
    el_rad = math.radians(max(1.0, elevation_deg))
    airmass = 1.0 / math.sin(el_rad)

    # Very crude extinction coefficient (dB per airmass at zenith)
    # This is a placeholder; real values depend on wavelength, aerosols, etc.
    zenith_extinction_db = 0.5  # Typical for 850 nm in good conditions

    return zenith_extinction_db * airmass


def optical_link_loss_db(
    elevation_deg: float,
    params: Optional[OpticalLinkParams] = None,
) -> float:
    """
    Compute total optical link loss.

    STUB: Not yet implemented. Returns placeholder value.

    Parameters
    ----------
    elevation_deg : float
        Elevation angle in degrees.
    params : OpticalLinkParams, optional
        Link parameters.

    Returns
    -------
    float
        Total link loss in dB.

    Raises
    ------
    NotImplementedError
        This function is a stub and not yet implemented.
    """
    raise NotImplementedError(
        "optical_link_loss_db is a stub for future Option B implementation. "
        "Use link_budget.total_channel_loss_db() for current scenario generation."
    )
