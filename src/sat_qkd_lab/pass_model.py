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
from .pointing import PointingParams, simulate_pointing_profile
from .polarization_drift import adjust_qber_for_angle, coincidence_matrix_from_qber


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


@dataclass(frozen=True)
class FadingParams:
    enabled: bool = False
    sigma_ln: float = 0.3
    n_samples: int = 50
    seed: int = 0


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


def sample_fading_factors(
    n_samples: int,
    sigma_ln: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample lognormal fading factors with unit mean.
    """
    if n_samples <= 0:
        return np.array([], dtype=float)
    if sigma_ln <= 0:
        return np.ones(n_samples, dtype=float)
    mu = -0.5 * sigma_ln ** 2
    return rng.lognormal(mean=mu, sigma=sigma_ln, size=n_samples)


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
    fading: Optional[FadingParams] = None,
    pointing: Optional[PointingParams] = None,
    background_scale: Optional[Sequence[float]] = None,
    polarization_angle_rad: Optional[Sequence[float]] = None,
    fading_series: Optional[Sequence[float]] = None,
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

    lock_state = None
    trans_multiplier = None
    dropout_count = 0
    lock_fraction = 0.0
    if pointing is not None:
        lock_state, trans_multiplier, dropout_count, lock_fraction = simulate_pointing_profile(
            time_s,
            pointing,
        )

    records: List[Dict[str, Any]] = []
    total_bits = 0.0
    key_rate_bps_values: List[float] = []
    key_rate_per_pulse_values: List[float] = []
    secret_bits_dt_values: List[float] = []

    rng_fading = np.random.default_rng(fading.seed) if fading is not None else None

    for t_s, el in zip(time_s, elevation_deg):
        loss_db = elevation_to_loss_db(float(el), link)
        p_bg_eff = background_prob(det.p_bg, link, params.background_mode)
        if background_scale is not None:
            idx = int(round(t_s / params.dt_seconds)) if params.dt_seconds > 0 else 0
            idx = min(max(idx, 0), len(background_scale) - 1)
            p_bg_eff = max(0.0, min(1.0, p_bg_eff * float(background_scale[idx])))
        eta_ch_mean = 10 ** (-loss_db / 10.0)
        if trans_multiplier is not None:
            idx = int(round(t_s / params.dt_seconds)) if params.dt_seconds > 0 else 0
            idx = min(max(idx, 0), len(trans_multiplier) - 1)
            eta_ch_mean *= float(trans_multiplier[idx])
        if fading_series is not None:
            idx = int(round(t_s / params.dt_seconds)) if params.dt_seconds > 0 else 0
            idx = min(max(idx, 0), len(fading_series) - 1)
            eta_ch_mean *= max(0.0, float(fading_series[idx]))

        n_sent = int(round(params.rep_rate_hz * params.dt_seconds))
        if fading is not None and fading.enabled:
            samples = sample_fading_factors(fading.n_samples, fading.sigma_ln, rng_fading)
            qbers = []
            key_rates = []
            for sample in samples:
                eta_ch = eta_ch_mean * sample
                p_sig = det.eta * eta_ch
                p_click = p_sig + p_bg_eff - p_sig * p_bg_eff
                qber_sample = expected_qber(p_sig, p_bg_eff, params.flip_prob)
                if polarization_angle_rad is not None:
                    idx = int(round(t_s / params.dt_seconds)) if params.dt_seconds > 0 else 0
                    idx = min(max(idx, 0), len(polarization_angle_rad) - 1)
                    angle = float(polarization_angle_rad[idx])
                    qber_sample = adjust_qber_for_angle(qber_sample, angle)
                n_received = int(round(n_sent * p_click))
                n_sifted = int(round(n_received * 0.5))
                if finite_key is not None:
                    n_errors = int(round((0.0 if qber_sample != qber_sample else qber_sample) * n_sifted))
                    fk_result = finite_key_rate_per_pulse(
                        n_sent=n_sent,
                        n_sifted=n_sifted,
                        n_errors=n_errors,
                        params=fk,
                        qber_abort_threshold=params.qber_abort_threshold,
                    )
                    key_rate = fk_result["key_rate_per_pulse"]
                else:
                    sifted_fraction = (n_sifted / n_sent) if n_sent > 0 else 0.0
                    key_rate = _asymptotic_key_rate_per_pulse(
                        qber_mean=qber_sample,
                        sifted_fraction=sifted_fraction,
                        ec_efficiency=fk.ec_efficiency,
                    )
                qbers.append(qber_sample)
                key_rates.append(key_rate)

            qber_mean = float(np.mean(qbers)) if qbers else float("nan")
            key_rate_per_pulse = float(np.mean(key_rates)) if key_rates else 0.0
            if qbers:
                qber_ci_low = float(np.quantile(qbers, 0.025))
                qber_ci_high = float(np.quantile(qbers, 0.975))
            else:
                qber_ci_low = None
                qber_ci_high = None
        else:
            eta_ch = eta_ch_mean
            p_sig = det.eta * eta_ch
            p_click = p_sig + p_bg_eff - p_sig * p_bg_eff
            qber_mean = expected_qber(p_sig, p_bg_eff, params.flip_prob)
            if polarization_angle_rad is not None:
                idx = int(round(t_s / params.dt_seconds)) if params.dt_seconds > 0 else 0
                idx = min(max(idx, 0), len(polarization_angle_rad) - 1)
                angle = float(polarization_angle_rad[idx])
                qber_mean = adjust_qber_for_angle(qber_mean, angle)
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

        finite_key_info = None
        if finite_key is not None:
            p_click_mean = det.eta * eta_ch_mean + p_bg_eff - det.eta * eta_ch_mean * p_bg_eff
            n_sifted_mean = int(round(n_sent * 0.5 * p_click_mean))
            n_errors_mean = int(round((0.0 if qber_mean != qber_mean else qber_mean) * n_sifted_mean))
            fk_status = finite_key_rate_per_pulse(
                n_sent=n_sent,
                n_sifted=n_sifted_mean,
                n_errors=n_errors_mean,
                params=fk,
                qber_abort_threshold=params.qber_abort_threshold,
            )
            finite_key_info = fk_status["finite_key"]
            if finite_key_info["status"] != "secure":
                key_rate_per_pulse = 0.0

        key_rate_bps = params.rep_rate_hz * key_rate_per_pulse
        secret_bits_dt = key_rate_bps * params.dt_seconds

        qber_z = None
        qber_x = None
        matrix_z = None
        matrix_x = None
        if polarization_angle_rad is not None:
            idx = int(round(t_s / params.dt_seconds)) if params.dt_seconds > 0 else 0
            idx = min(max(idx, 0), len(polarization_angle_rad) - 1)
            angle = float(polarization_angle_rad[idx])
            qber_z = adjust_qber_for_angle(qber_mean, angle, phase_offset_rad=0.0)
            qber_x = adjust_qber_for_angle(qber_mean, angle, phase_offset_rad=0.25 * math.pi)
            p_click_mean = det.eta * eta_ch_mean + p_bg_eff - det.eta * eta_ch_mean * p_bg_eff
            n_sifted_mean = int(round(n_sent * 0.5 * p_click_mean))
            matrix_z = coincidence_matrix_from_qber(n_sifted_mean, qber_z)
            matrix_x = coincidence_matrix_from_qber(n_sifted_mean, qber_x)

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
        if finite_key_info is not None:
            record["finite_key"] = finite_key_info
        if background_scale is not None:
            record["background_prob"] = float(p_bg_eff)
        if qber_z is not None and qber_x is not None:
            record["qber_z"] = float(qber_z)
            record["qber_x"] = float(qber_x)
            record["matrix_z"] = matrix_z
            record["matrix_x"] = matrix_x
        if lock_state is not None and trans_multiplier is not None:
            idx = int(round(t_s / params.dt_seconds)) if params.dt_seconds > 0 else 0
            idx = min(max(idx, 0), len(lock_state) - 1)
            record["pointing_locked"] = bool(lock_state[idx])
            record["pointing_transmittance"] = float(trans_multiplier[idx])
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
        key_rate_per_pulse_values.append(record["key_rate_per_pulse"])
        secret_bits_dt_values.append(record["secret_bits_dt"])

    secure_mask = np.array(secret_bits_dt_values) > 0.0
    if np.any(secure_mask):
        idx = np.where(secure_mask)[0]
        start = float(time_s[idx[0]])
        end = float(time_s[idx[-1]])
        start_elev = float(elevation_deg[idx[0]])
        end_elev = float(elevation_deg[idx[-1]])
        secure_window_seconds = float((idx[-1] - idx[0] + 1) * params.dt_seconds)
        secure_window_seconds_total = float(np.sum(secure_mask) * params.dt_seconds)
        transitions = np.diff(secure_mask.astype(int))
        secure_segments = int(np.sum(transitions == 1) + (1 if secure_mask[0] else 0))
        key_rate_in_window = [key_rate_per_pulse_values[i] for i in idx]
        mean_key_rate_in_window = float(np.mean(key_rate_in_window)) if key_rate_in_window else 0.0
    else:
        start = None
        end = None
        start_elev = float(params.min_elevation_deg)
        end_elev = float(params.min_elevation_deg)
        secure_window_seconds = 0.0
        secure_window_seconds_total = 0.0
        secure_segments = 0
        mean_key_rate_in_window = 0.0

    summary = {
        "qber_abort": float(params.qber_abort_threshold),
        "peak_key_rate_bps": float(max(key_rate_bps_values)) if key_rate_bps_values else 0.0,
        "peak_key_rate": float(max(key_rate_per_pulse_values)) if key_rate_per_pulse_values else 0.0,
        "total_secret_bits": float(total_bits),
        "secure_window_seconds": float(secure_window_seconds),
        "secure_window_seconds_total": float(secure_window_seconds_total),
        "secure_window_segments": int(secure_segments),
        "secure_start_s": float(start) if start is not None else 0.0,
        "secure_end_s": float(end) if end is not None else 0.0,
        "secure_start_elevation_deg": float(start_elev),
        "secure_end_elevation_deg": float(end_elev),
        "mean_key_rate_in_window": float(mean_key_rate_in_window),
        "secure_window": {
            "t_start_seconds": start,
            "t_end_seconds": end,
        },
    }
    if pointing is not None:
        summary["lock_fraction"] = float(lock_fraction)
        summary["dropout_count"] = int(dropout_count)

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
    if "background_prob" in records[0]:
        time_series["background_prob"] = [float(r["background_prob"]) for r in records]
    if "pointing_locked" in records[0]:
        time_series["pointing_locked"] = [int(bool(r.get("pointing_locked"))) for r in records]
    if "qber_z" in records[0]:
        time_series["qber_z"] = [float(r["qber_z"]) for r in records]
        time_series["qber_x"] = [float(r["qber_x"]) for r in records]
        time_series["matrix_z"] = [r["matrix_z"] for r in records]
        time_series["matrix_x"] = [r["matrix_x"] for r in records]
    if include_ci and "qber_ci_low" in records[0]:
        time_series["qber_ci_low"] = [float(r["qber_ci_low"]) for r in records]
        time_series["qber_ci_high"] = [float(r["qber_ci_high"]) for r in records]

    return time_series
