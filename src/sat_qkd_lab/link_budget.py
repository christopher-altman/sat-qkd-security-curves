from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class SatLinkParams:
    wavelength_m: float = 1550e-9      # 1550 nm (telecom/free-space optical common)
    altitude_m: float = 550e3          # LEO-ish
    earth_radius_m: float = 6371e3
    atmospheric_loss_db_zenith: float = 2.0
    atmospheric_loss_db_horizon: float = 12.0
    pointing_loss_db: float = 2.0      # lumped (jitter, tracking, optics)
    system_margin_db: float = 3.0      # detector + coupling + misc

def slant_range_m(elevation_deg: float, p: SatLinkParams) -> float:
    """Simple geometry: ground station at Earth surface, satellite at altitude p.altitude_m."""
    el = math.radians(max(1e-6, min(89.999, elevation_deg)))
    Re = p.earth_radius_m
    h = p.altitude_m
    # Approx slant range:
    return math.sqrt((Re + h) ** 2 - (Re * math.cos(el)) ** 2) - Re * math.sin(el)

def fspl_db(range_m: float, wavelength_m: float) -> float:
    """Free-space path loss for an optical carrier (geometric spreading)."""
    return 20.0 * math.log10(4.0 * math.pi * range_m / wavelength_m)

def atmospheric_loss_db(elevation_deg: float, p: SatLinkParams) -> float:
    """Toy model: linear interpolate zenith-to-horizon loss vs elevation."""
    el = max(0.0, min(90.0, elevation_deg))
    t = (90.0 - el) / 90.0
    return (1 - t) * p.atmospheric_loss_db_zenith + t * p.atmospheric_loss_db_horizon

def total_channel_loss_db(elevation_deg: float, p: SatLinkParams) -> float:
    r = slant_range_m(elevation_deg, p)
    return fspl_db(r, p.wavelength_m) + atmospheric_loss_db(elevation_deg, p) + p.pointing_loss_db + p.system_margin_db
