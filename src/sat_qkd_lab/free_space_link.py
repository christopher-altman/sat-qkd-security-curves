"""
Free-Space Optical Link Budget Model for Satellite QKD.

This module implements a physically-grounded (but simplified) free-space
optical link model for satellite-to-ground QKD. It provides:

1. Diffraction-limited beam propagation and aperture coupling
2. Pointing error/jitter loss model
3. Atmospheric extinction and turbulence (lognormal fading)
4. Day/night background noise model
5. Pass geometry and secure window estimation

Model Assumptions and Limitations
---------------------------------
- Gaussian beam approximation for transmitter
- Rayleigh-distributed pointing error
- Lognormal scintillation (weak-to-moderate turbulence regime)
- Simple airmass-based atmospheric extinction
- Day/night background as multiplicative factor on p_bg

This model is simplified for educational and demonstration purposes.
For engineering accuracy, use MODTRAN for atmosphere and validated
turbulence profiles (Hufnagel-Valley, etc.).

References
----------
- Liao et al., Nature 549, 43 (2017) - Micius satellite QKD
- Bourgoin et al., NJP 15, 023006 (2013) - LEO QKD link analysis
- Andrews & Phillips, "Laser Beam Propagation through Random Media"
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


@dataclass(frozen=True)
class FreeSpaceLinkParams:
    """
    Parameters for free-space optical satellite link.

    Attributes
    ----------
    wavelength_m : float
        Operating wavelength in meters. Default: 850 nm (Si SPAD compatible).
    tx_diameter_m : float
        Transmitter aperture diameter in meters. Default: 0.3 m (30 cm telescope).
    rx_diameter_m : float
        Receiver aperture diameter in meters. Default: 1.0 m (ground station).
    beam_divergence_rad : float or None
        Half-angle beam divergence. If None, computed from diffraction limit.
    sigma_point_rad : float
        RMS pointing error in radians. Default: 2 µrad.
    altitude_m : float
        Satellite orbital altitude in meters. Default: 500 km (LEO).
    earth_radius_m : float
        Earth radius in meters.
    atm_loss_db_zenith : float
        Atmospheric extinction at zenith in dB. Default: 0.5 dB (good conditions).
    sigma_ln : float
        Log-amplitude standard deviation for turbulence. Default: 0 (no turbulence).
    system_loss_db : float
        System losses (detector coupling, optics, etc.) in dB. Default: 3 dB.
    is_night : bool
        Night-time pass (lower background). Default: True.
    day_background_factor : float
        Multiplicative factor for p_bg during daytime. Default: 100.
    """
    wavelength_m: float = 850e-9
    tx_diameter_m: float = 0.30
    rx_diameter_m: float = 1.0
    beam_divergence_rad: Optional[float] = None
    sigma_point_rad: float = 2e-6
    altitude_m: float = 500e3
    earth_radius_m: float = 6371e3
    atm_loss_db_zenith: float = 0.5
    sigma_ln: float = 0.0
    system_loss_db: float = 3.0
    is_night: bool = True
    day_background_factor: float = 100.0

    def __post_init__(self):
        if self.wavelength_m <= 0:
            raise ValueError(f"wavelength_m must be > 0, got {self.wavelength_m}")
        if self.tx_diameter_m <= 0:
            raise ValueError(f"tx_diameter_m must be > 0, got {self.tx_diameter_m}")
        if self.rx_diameter_m <= 0:
            raise ValueError(f"rx_diameter_m must be > 0, got {self.rx_diameter_m}")
        if self.sigma_point_rad < 0:
            raise ValueError(f"sigma_point_rad must be >= 0, got {self.sigma_point_rad}")
        if self.sigma_ln < 0:
            raise ValueError(f"sigma_ln must be >= 0, got {self.sigma_ln}")

    @property
    def effective_divergence_rad(self) -> float:
        """Effective beam divergence (diffraction limit if not specified)."""
        if self.beam_divergence_rad is not None:
            return self.beam_divergence_rad
        # Diffraction-limited divergence: theta = 1.22 * lambda / D
        return 1.22 * self.wavelength_m / self.tx_diameter_m


def slant_range_m(elevation_deg: float, params: FreeSpaceLinkParams) -> float:
    """
    Compute slant range from ground station to satellite.

    Uses spherical Earth geometry to compute the line-of-sight distance
    from a ground station to a satellite at given elevation angle.

    Parameters
    ----------
    elevation_deg : float
        Elevation angle above horizon in degrees (0 = horizon, 90 = zenith).
    params : FreeSpaceLinkParams
        Link parameters containing orbital altitude.

    Returns
    -------
    float
        Slant range in meters.
    """
    el_rad = math.radians(max(0.1, min(89.99, elevation_deg)))
    Re = params.earth_radius_m
    h = params.altitude_m

    # Geometric slant range formula
    sin_el = math.sin(el_rad)
    cos_el = math.cos(el_rad)
    slant = math.sqrt((Re + h) ** 2 - (Re * cos_el) ** 2) - Re * sin_el
    return max(slant, h)  # At least orbital altitude


def beam_radius_at_range(range_m: float, params: FreeSpaceLinkParams) -> float:
    """
    Compute 1/e^2 beam radius at given range.

    For a Gaussian beam, the far-field radius grows linearly with range:
        w(R) = R * theta
    where theta is the half-angle divergence.

    Parameters
    ----------
    range_m : float
        Distance from transmitter in meters.
    params : FreeSpaceLinkParams
        Link parameters.

    Returns
    -------
    float
        Beam radius (1/e^2 intensity) in meters.
    """
    theta = params.effective_divergence_rad
    return range_m * theta


def geometric_coupling_efficiency(range_m: float, params: FreeSpaceLinkParams) -> float:
    """
    Compute geometric coupling efficiency (power fraction captured).

    Models the fraction of transmitted Gaussian beam power captured by
    a circular receiver aperture.

    For a Gaussian beam with 1/e^2 radius w and a circular aperture of
    radius a, the captured fraction is:
        eta = 1 - exp(-2 * (a/w)^2)

    Parameters
    ----------
    range_m : float
        Slant range in meters.
    params : FreeSpaceLinkParams
        Link parameters.

    Returns
    -------
    float
        Coupling efficiency (0 to 1).
    """
    w = beam_radius_at_range(range_m, params)
    a = params.rx_diameter_m / 2  # Aperture radius

    if w <= 0:
        return 1.0

    # Gaussian beam truncation formula
    eta = 1.0 - math.exp(-2.0 * (a / w) ** 2)
    return max(1e-15, eta)


def geometric_coupling_loss_db(range_m: float, params: FreeSpaceLinkParams) -> float:
    """
    Compute geometric coupling loss in dB.

    Parameters
    ----------
    range_m : float
        Slant range in meters.
    params : FreeSpaceLinkParams
        Link parameters.

    Returns
    -------
    float
        Coupling loss in dB (positive value = loss).
    """
    eta = geometric_coupling_efficiency(range_m, params)
    return -10.0 * math.log10(eta)


def pointing_loss_db(params: FreeSpaceLinkParams) -> float:
    """
    Compute average pointing loss from Gaussian jitter.

    Models pointing error as a 2D Gaussian (Rayleigh magnitude) with
    RMS sigma_point. The average coupling loss for a Gaussian beam is:
        <loss> = 1 / (1 + (sigma/theta)^2)
    where theta is the beam divergence half-angle.

    This is the "average" loss - instantaneous loss varies with jitter.

    Parameters
    ----------
    params : FreeSpaceLinkParams
        Link parameters.

    Returns
    -------
    float
        Average pointing loss in dB.
    """
    if params.sigma_point_rad <= 0:
        return 0.0

    theta = params.effective_divergence_rad
    if theta <= 0:
        return 0.0

    # Pointing loss factor for Gaussian beam with Rayleigh pointing error
    # Derived from integrating Gaussian beam coupling over pointing distribution
    ratio_sq = (params.sigma_point_rad / theta) ** 2
    avg_coupling = 1.0 / (1.0 + ratio_sq)
    return -10.0 * math.log10(max(1e-15, avg_coupling))


def atmospheric_extinction_db(elevation_deg: float, params: FreeSpaceLinkParams) -> float:
    """
    Compute atmospheric extinction loss using airmass model.

    Uses the plane-parallel atmosphere approximation:
        loss = zenith_loss * airmass
    where airmass = 1/sin(elevation) for elevation > ~10 deg.

    For very low elevations, uses Kasten-Young formula for airmass.

    Parameters
    ----------
    elevation_deg : float
        Elevation angle in degrees.
    params : FreeSpaceLinkParams
        Link parameters.

    Returns
    -------
    float
        Atmospheric extinction in dB.
    """
    el = max(1.0, min(90.0, elevation_deg))

    # Kasten-Young airmass formula (accurate to ~0.5 deg elevation)
    el_rad = math.radians(el)
    airmass = 1.0 / (math.sin(el_rad) + 0.50572 * (el + 6.07995) ** (-1.6364))

    return params.atm_loss_db_zenith * airmass


def sample_turbulence_fading(
    n_samples: int,
    sigma_ln: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample lognormal turbulence fading multipliers.

    In weak-to-moderate turbulence, intensity fluctuations follow a
    lognormal distribution. The log-amplitude variance sigma_ln^2 is
    related to the Rytov variance and scintillation index.

    Parameters
    ----------
    n_samples : int
        Number of fading samples.
    sigma_ln : float
        Log-amplitude standard deviation.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Fading multipliers (mean = 1 for sigma_ln -> 0).
    """
    if sigma_ln <= 0 or n_samples <= 0:
        return np.ones(n_samples)

    # Lognormal: I = exp(2*chi) where chi ~ N(-sigma^2/2, sigma^2)
    # This gives <I> = 1 (normalized mean intensity)
    chi = rng.normal(-sigma_ln ** 2, sigma_ln, n_samples)
    return np.exp(2 * chi)


def total_link_loss_db(
    elevation_deg: float,
    params: FreeSpaceLinkParams,
) -> float:
    """
    Compute total deterministic link loss (excluding turbulence fading).

    Combines:
    - Geometric coupling loss (diffraction + aperture)
    - Pointing loss
    - Atmospheric extinction
    - System losses

    Parameters
    ----------
    elevation_deg : float
        Elevation angle in degrees.
    params : FreeSpaceLinkParams
        Link parameters.

    Returns
    -------
    float
        Total loss in dB.
    """
    range_m = slant_range_m(elevation_deg, params)

    loss_geo = geometric_coupling_loss_db(range_m, params)
    loss_point = pointing_loss_db(params)
    loss_atm = atmospheric_extinction_db(elevation_deg, params)
    loss_sys = params.system_loss_db

    return loss_geo + loss_point + loss_atm + loss_sys


def effective_background_prob(
    base_p_bg: float,
    params: FreeSpaceLinkParams,
) -> float:
    """
    Compute effective background probability accounting for day/night.

    Daytime solar scattering dramatically increases background counts.
    This simplified model multiplies p_bg by a factor for daytime.

    Parameters
    ----------
    base_p_bg : float
        Base background probability (night-time value).
    params : FreeSpaceLinkParams
        Link parameters.

    Returns
    -------
    float
        Effective background probability.
    """
    if params.is_night:
        return base_p_bg
    return min(1.0, base_p_bg * params.day_background_factor)


def compute_pass_geometry(
    elevation_profile_deg: np.ndarray,
    time_step_s: float,
    params: FreeSpaceLinkParams,
) -> List[Dict[str, Any]]:
    """
    Compute link parameters over a satellite pass.

    Parameters
    ----------
    elevation_profile_deg : np.ndarray
        Elevation angles over time.
    time_step_s : float
        Time step between samples in seconds.
    params : FreeSpaceLinkParams
        Link parameters.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with loss, range, etc. for each time step.
    """
    results = []
    for i, el in enumerate(elevation_profile_deg):
        t = i * time_step_s
        range_m = slant_range_m(float(el), params)
        loss_db = total_link_loss_db(float(el), params)

        results.append({
            "time_s": t,
            "elevation_deg": float(el),
            "slant_range_m": range_m,
            "loss_db": loss_db,
        })

    return results


def generate_elevation_profile(
    max_elevation_deg: float,
    min_elevation_deg: float = 10.0,
    time_step_s: float = 1.0,
    pass_duration_s: float = 600.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a symmetric satellite pass elevation profile.

    Creates a simple sinusoidal-like elevation profile that rises from
    min_elevation to max_elevation and back.

    Parameters
    ----------
    max_elevation_deg : float
        Maximum elevation (at pass apex).
    min_elevation_deg : float
        Minimum elevation (at pass start/end).
    time_step_s : float
        Time step in seconds.
    pass_duration_s : float
        Total pass duration in seconds.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (time array in seconds, elevation array in degrees)
    """
    n_steps = int(pass_duration_s / time_step_s) + 1
    time_s = np.linspace(0, pass_duration_s, n_steps)

    # Sinusoidal profile: el(t) = min + (max-min) * sin^2(pi*t/T)
    phase = np.pi * time_s / pass_duration_s
    elevation_deg = min_elevation_deg + (max_elevation_deg - min_elevation_deg) * np.sin(phase) ** 2

    return time_s, elevation_deg


def estimate_secure_window(
    pass_results: List[Dict[str, Any]],
    key_rates: np.ndarray,
    time_step_s: float,
    min_key_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Estimate the secure communication window from pass results.

    The secure window is the time interval where key rate > min_key_rate.

    Parameters
    ----------
    pass_results : List[Dict[str, Any]]
        Pass geometry results.
    key_rates : np.ndarray
        Key rates at each time step.
    time_step_s : float
        Time step in seconds.
    min_key_rate : float
        Minimum key rate threshold.

    Returns
    -------
    Dict[str, Any]
        Secure window statistics.
    """
    times = np.array([r["time_s"] for r in pass_results])
    elevations = np.array([r["elevation_deg"] for r in pass_results])

    # Find intervals where key rate > threshold
    secure_mask = key_rates > min_key_rate
    n_secure = np.sum(secure_mask)

    if n_secure == 0:
        return {
            "secure_window_seconds": 0.0,
            "secure_start_s": None,
            "secure_end_s": None,
            "secure_start_elevation_deg": None,
            "secure_end_elevation_deg": None,
            "peak_key_rate": 0.0,
            "total_secret_bits": 0.0,
            "mean_key_rate_in_window": 0.0,
        }

    # Find first and last secure indices
    secure_indices = np.where(secure_mask)[0]
    first_idx = secure_indices[0]
    last_idx = secure_indices[-1]

    secure_duration = (last_idx - first_idx + 1) * time_step_s
    total_bits = float(np.sum(key_rates[secure_mask])) * time_step_s

    return {
        "secure_window_seconds": secure_duration,
        "secure_start_s": float(times[first_idx]),
        "secure_end_s": float(times[last_idx]),
        "secure_start_elevation_deg": float(elevations[first_idx]),
        "secure_end_elevation_deg": float(elevations[last_idx]),
        "peak_key_rate": float(np.max(key_rates)),
        "total_secret_bits": total_bits,
        "mean_key_rate_in_window": float(np.mean(key_rates[secure_mask])) if n_secure > 0 else 0.0,
    }


# Default parameters for common scenarios
DEFAULT_LEO_NIGHT = FreeSpaceLinkParams(
    wavelength_m=850e-9,
    tx_diameter_m=0.30,
    rx_diameter_m=1.0,
    sigma_point_rad=2e-6,
    altitude_m=500e3,
    atm_loss_db_zenith=0.5,
    sigma_ln=0.0,
    is_night=True,
)

DEFAULT_LEO_DAY = FreeSpaceLinkParams(
    wavelength_m=850e-9,
    tx_diameter_m=0.30,
    rx_diameter_m=1.0,
    sigma_point_rad=2e-6,
    altitude_m=500e3,
    atm_loss_db_zenith=0.5,
    sigma_ln=0.0,
    is_night=False,
    day_background_factor=100.0,
)
