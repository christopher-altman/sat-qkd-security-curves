"""
Pass model utilities for elevation-to-loss and finite-key pass sweeps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

import numpy as np

from .calibration import CalibrationModel
from .detector import DetectorParams, DEFAULT_DETECTOR
from .finite_key import FiniteKeyParams, finite_key_rate_per_pulse, finite_key_bounds
from .free_space_link import (
    FreeSpaceLinkParams,
    effective_background_prob,
    generate_elevation_profile,
    total_link_loss_db,
)
from .helpers import h2
from .sweep import compute_headroom


@dataclass(frozen=True)
class PassModelParams:
    max_elevation_deg: float = 60.0
    min_elevation_deg: float = 10.0
    pass_seconds: float = 300.0
    dt_seconds: float = 1.0
    flip_prob: float = 0.005
    rep_rate_hz: float = 1e8
    qber_abort_threshold: float = 0.11
    background_mode: str = "night"


def elevation_to_loss_db(elevation_deg: float, link_params: FreeSpaceLinkParams) -> float:
    """Map elevation angle to total channel loss (dB)."""
    return float(total_link_loss_db(float(elevation_deg), link_params))


def background_prob(
    p_bg: float,
    link_params: FreeSpaceLinkParams,
    mode: str,
) -> float:
    """Compute background probability for a day/night mode."""
    if mode not in ("day", "night"):
        raise ValueError(f"background mode must be 'day' or 'night', got {mode}")
    params = FreeSpaceLinkParams(
        wavelength_m=link_params.wavelength_m,
        tx_diameter_m=link_params.tx_diameter_m,
        rx_diameter_m=link_params.rx_diameter_m,
        beam_divergence_rad=link_params.beam_divergence_rad,
        sigma_point_rad=link_params.sigma_point_rad,
        altitude_m=link_params.altitude_m,
        earth_radius_m=link_params.earth_radius_m,
        atm_loss_db_zenith=link_params.atm_loss_db_zenith,
        sigma_ln=link_params.sigma_ln,
        system_loss_db=link_params.system_loss_db,
        is_night=(mode == "night"),
        day_background_factor=link_params.day_background_factor,
    )
    return float(effective_background_prob(p_bg, params))


def expected_qber(p_sig: float, p_bg: float, flip_prob: float) -> float:
    """Expected QBER from signal/background mixing and intrinsic flips."""
    p_click = p_sig + p_bg - p_sig * p_bg
    if p_click <= 0:
        return float("nan")
    qber = (0.5 * p_bg + flip_prob * p_sig) / p_click
    return min(0.5, max(0.0, qber))


def _asymptotic_key_rate_per_pulse(
    qber_mean: float,
    sifted_fraction: float,
    ec_efficiency: float,
) -> float:
    """Asymptotic key rate per pulse with BB84 secret fraction."""
    if qber_mean != qber_mean:
        return 0.0
    secret_fraction = max(0.0, 1.0 - ec_efficiency * h2(qber_mean) - h2(qber_mean))
    return max(0.0, sifted_fraction * secret_fraction)


def compute_pass_records(
    params: PassModelParams,
    link_params: Optional[FreeSpaceLinkParams] = None,
    detector: Optional[DetectorParams] = None,
    finite_key: Optional[FiniteKeyParams] = None,
    calibration: Optional[CalibrationModel] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Compute per-time-step pass records and summary outputs.

    Returns list of records (for plotting) and a summary dict.
    """
    link = link_params if link_params is not None else FreeSpaceLinkParams()
    det = detector if detector is not None else DEFAULT_DETECTOR
    fk = finite_key if finite_key is not None else FiniteKeyParams()

    time_s, elevation_deg = generate_elevation_profile(
        max_elevation_deg=params.max_elevation_deg,
        min_elevation_deg=params.min_elevation_deg,
        time_step_s=params.dt_seconds,
        pass_duration_s=params.pass_seconds,
    )

    records: List[Dict[str, Any]] = []
    total_bits = 0.0
    key_rate_bps_values: List[float] = []
    secret_bits_dt_values: List[float] = []

    for t_s, el in zip(time_s, elevation_deg):
        loss_db = elevation_to_loss_db(float(el), link)
        eta_ch = 10 ** (-loss_db / 10.0)
        p_bg_eff = background_prob(det.p_bg, link, params.background_mode)
        p_sig = det.eta * eta_ch
        p_click = p_sig + p_bg_eff - p_sig * p_bg_eff
        qber_mean = expected_qber(p_sig, p_bg_eff, params.flip_prob)

        n_sent = int(round(params.rep_rate_hz * params.dt_seconds))
        n_received = int(round(n_sent * p_click))
        n_sifted = int(round(n_received * 0.5))
        n_errors = int(round((0.0 if qber_mean != qber_mean else qber_mean) * n_sifted))

        if finite_key is not None:
            fk_result = finite_key_rate_per_pulse(
                n_sent=n_sent,
                n_sifted=n_sifted,
                n_errors=n_errors,
                params=fk,
                qber_abort_threshold=params.qber_abort_threshold,
            )
            if n_sifted > 0:
                if fk.m_pe is not None:
                    m_pe = max(1, min(n_sifted, fk.m_pe))
                else:
                    m_pe = max(1, int(fk.pe_frac * n_sifted))
            else:
                m_pe = 0
            qber_bounds = finite_key_bounds(
                n_sifted=n_sifted,
                n_errors=n_errors,
                eps_pe=fk.eps_pe,
                m_pe=m_pe if m_pe > 0 else None,
            )
            qber_ci_low = qber_bounds["qber_lower"]
            qber_ci_high = qber_bounds["qber_upper"]
            key_rate_per_pulse = fk_result["key_rate_per_pulse"]
        else:
            sifted_fraction = (n_sifted / n_sent) if n_sent > 0 else 0.0
            key_rate_per_pulse = _asymptotic_key_rate_per_pulse(
                qber_mean=qber_mean,
                sifted_fraction=sifted_fraction,
                ec_efficiency=fk.ec_efficiency,
            )
            qber_ci_low = None
            qber_ci_high = None

        key_rate_bps = params.rep_rate_hz * key_rate_per_pulse
        secret_bits_dt = key_rate_bps * params.dt_seconds

        record = {
            "time_s": float(t_s),
            "elevation_deg": float(el),
            "loss_db": float(loss_db),
            "qber_mean": float(qber_mean),
            "key_rate_per_pulse": float(key_rate_per_pulse),
            "key_rate_bps": float(key_rate_bps),
            "secret_bits_dt": float(secret_bits_dt),
            "headroom": compute_headroom(
                qber_mean,
                params.qber_abort_threshold,
                qber_ci_low=qber_ci_low,
                qber_ci_high=qber_ci_high,
            )["headroom"],
        }
        if qber_ci_low is not None and qber_ci_high is not None:
            record["qber_ci_low"] = float(qber_ci_low)
            record["qber_ci_high"] = float(qber_ci_high)

        if calibration is not None:
            record = calibration.apply_to_record(record, loss_db=record["loss_db"])
            record["headroom"] = compute_headroom(
                record["qber_mean"],
                params.qber_abort_threshold,
                qber_ci_low=record.get("qber_ci_low"),
                qber_ci_high=record.get("qber_ci_high"),
            )["headroom"]
            record["key_rate_bps"] = params.rep_rate_hz * record["key_rate_per_pulse"]
            record["secret_bits_dt"] = record["key_rate_bps"] * params.dt_seconds

        records.append(record)
        total_bits += record["secret_bits_dt"]
        key_rate_bps_values.append(record["key_rate_bps"])
        secret_bits_dt_values.append(record["secret_bits_dt"])

    secure_mask = np.array(secret_bits_dt_values) > 0.0
    if np.any(secure_mask):
        idx = np.where(secure_mask)[0]
        start = float(time_s[idx[0]])
        end = float(time_s[idx[-1]])
        secure_window_seconds = float((idx[-1] - idx[0] + 1) * params.dt_seconds)
    else:
        start = None
        end = None
        secure_window_seconds = 0.0

    summary = {
        "qber_abort": float(params.qber_abort_threshold),
        "peak_key_rate_bps": float(max(key_rate_bps_values)) if key_rate_bps_values else 0.0,
        "total_secret_bits": float(total_bits),
        "secure_window_seconds": float(secure_window_seconds),
        "secure_window": {
            "t_start_seconds": start,
            "t_end_seconds": end,
        },
    }

    return records, summary


def records_to_time_series(
    records: Sequence[Dict[str, Any]],
    include_ci: bool,
) -> Dict[str, List[float]]:
    """Convert per-step records to time-series arrays for JSON output."""
    time_series: Dict[str, List[float]] = {
        "t_seconds": [float(r["time_s"]) for r in records],
        "elevation_deg": [float(r["elevation_deg"]) for r in records],
        "loss_db": [float(r["loss_db"]) for r in records],
        "qber_mean": [float(r["qber_mean"]) for r in records],
        "key_rate_per_pulse": [float(r["key_rate_per_pulse"]) for r in records],
        "key_rate_bps": [float(r["key_rate_bps"]) for r in records],
        "secret_bits_dt": [float(r["secret_bits_dt"]) for r in records],
        "headroom": [float(r["headroom"]) for r in records],
    }
    if include_ci:
        time_series["qber_ci_low"] = [float(r["qber_ci_low"]) for r in records]
        time_series["qber_ci_high"] = [float(r["qber_ci_high"]) for r in records]

    return time_series
