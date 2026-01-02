from __future__ import annotations
from typing import List, Dict, Any, Sequence, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .bb84 import simulate_bb84, Attack
from .link_budget import SatLinkParams, total_channel_loss_db
from .detector import DetectorParams, DEFAULT_DETECTOR
from .finite_key import FiniteKeyParams, finite_key_rate_per_pulse, compare_asymptotic_vs_finite
from .free_space_link import (
    FreeSpaceLinkParams,
    total_link_loss_db as free_space_loss_db,
    effective_background_prob,
    generate_elevation_profile,
    estimate_secure_window,
    sample_turbulence_fading,
)


def sweep_loss(
    loss_db_values: Sequence[float],
    flip_prob: float = 0.0,
    attack: Attack = "none",
    n_pulses: int = 200_000,
    seed: int = 0,
    detector: Optional[DetectorParams] = None,
) -> List[Dict[str, Any]]:
    """
    Sweep over loss values and run BB84 simulation for each.

    Parameters
    ----------
    loss_db_values : Sequence[float]
        Channel loss values in dB.
    flip_prob : float
        Intrinsic bit-flip probability.
    attack : Attack
        Attack type ("none" or "intercept_resend").
    n_pulses : int
        Number of pulses per simulation.
    seed : int
        Base random seed (incremented for each loss value).
    detector : DetectorParams, optional
        Detector model parameters.

    Returns
    -------
    List[Dict[str, Any]]
        List of result dictionaries, one per loss value.
    """
    det = detector if detector is not None else DEFAULT_DETECTOR
    out: List[Dict[str, Any]] = []
    for i, loss_db in enumerate(loss_db_values):
        s = simulate_bb84(
            n_pulses=n_pulses,
            loss_db=float(loss_db),
            flip_prob=float(flip_prob),
            attack=attack,
            seed=seed + i,
            detector=det,
        )
        out.append({
            "loss_db": float(loss_db),
            "flip_prob": float(flip_prob),
            "attack": str(attack),
            "n_sent": s.n_sent,
            "n_received": s.n_received,
            "n_sifted": s.n_sifted,
            "qber": s.qber,
            "secret_fraction": s.secret_fraction,
            "key_rate_per_pulse": s.key_rate_per_pulse,
            "n_secret_est": s.n_secret_est,
            "aborted": bool(s.aborted),
        })
    return out


def sweep_satellite_pass(
    elevation_deg_values: Sequence[float],
    flip_prob: float = 0.0,
    attack: Attack = "none",
    n_pulses: int = 200_000,
    seed: int = 0,
    link_params: SatLinkParams | None = None,
    detector: Optional[DetectorParams] = None,
) -> List[Dict[str, Any]]:
    """
    Sweep over satellite elevation angles and run BB84 simulation for each.

    Maps elevation angles to loss via the satellite link budget model,
    then runs BB84 simulations.
    """
    p = link_params or SatLinkParams()
    det = detector if detector is not None else DEFAULT_DETECTOR
    out: List[Dict[str, Any]] = []
    for i, el in enumerate(elevation_deg_values):
        loss_db = total_channel_loss_db(float(el), p)
        s = simulate_bb84(
            n_pulses=n_pulses,
            loss_db=float(loss_db),
            flip_prob=float(flip_prob),
            attack=attack,
            seed=seed + i,
            detector=det,
        )
        out.append({
            "elevation_deg": float(el),
            "loss_db": float(loss_db),
            "flip_prob": float(flip_prob),
            "attack": str(attack),
            "qber": s.qber,
            "secret_fraction": s.secret_fraction,
            "key_rate_per_pulse": s.key_rate_per_pulse,
            "n_secret_est": s.n_secret_est,
            "n_sifted": s.n_sifted,
            "aborted": bool(s.aborted),
        })
    return out


# --- Monte Carlo sweep with confidence intervals ---

def _run_single_trial(args: tuple) -> Dict[str, Any]:
    """Worker function for parallel sweep trials."""
    loss_db, flip_prob, attack, n_pulses, seed, det_eta, det_p_bg = args
    det = DetectorParams(eta=det_eta, p_bg=det_p_bg)
    s = simulate_bb84(
        n_pulses=n_pulses,
        loss_db=loss_db,
        flip_prob=flip_prob,
        attack=attack,
        seed=seed,
        detector=det,
    )
    return {
        "qber": s.qber,
        "secret_fraction": s.secret_fraction,
        "key_rate_per_pulse": s.key_rate_per_pulse,
        "n_sifted": s.n_sifted,
        "n_sent": s.n_sent,
        "n_secret_est": s.n_secret_est,
        "aborted": s.aborted,
    }


def sweep_loss_with_ci(
    loss_db_values: Sequence[float],
    flip_prob: float = 0.0,
    attack: Attack = "none",
    n_pulses: int = 200_000,
    seed: int = 0,
    n_trials: int = 10,
    detector: Optional[DetectorParams] = None,
    n_workers: int = 1,
) -> List[Dict[str, Any]]:
    """
    Sweep over loss values with Monte Carlo confidence intervals.

    Runs multiple independent trials for each loss value and computes
    mean, standard deviation, and 95% confidence intervals for key metrics.

    Parameters
    ----------
    loss_db_values : Sequence[float]
        Channel loss values in dB.
    flip_prob : float
        Intrinsic bit-flip probability.
    attack : Attack
        Attack type ("none" or "intercept_resend").
    n_pulses : int
        Number of pulses per simulation.
    seed : int
        Base random seed.
    n_trials : int
        Number of independent trials per loss value.
    detector : DetectorParams, optional
        Detector model parameters.
    n_workers : int
        Number of parallel workers (1 = sequential).

    Returns
    -------
    List[Dict[str, Any]]
        List of result dictionaries with mean, std, and CI bounds.

    Notes
    -----
    Uses numpy.random.SeedSequence to generate independent RNG streams
    for each (loss_index, trial) pair. This avoids RNG collisions when
    sweeping over parameter grids with the same base seed.
    """
    det = detector if detector is not None else DEFAULT_DETECTOR
    out: List[Dict[str, Any]] = []

    # Create SeedSequence for proper RNG independence across grid points
    n_points = len(loss_db_values)
    total_streams = n_points * n_trials
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(total_streams)

    for i, loss_db in enumerate(loss_db_values):
        # Prepare arguments for all trials at this loss value
        # Use SeedSequence-derived seeds for independence
        trial_args = [
            (float(loss_db), float(flip_prob), attack, n_pulses,
             int(child_seeds[i * n_trials + t].generate_state(1)[0]), det.eta, det.p_bg)
            for t in range(n_trials)
        ]

        # Run trials (parallel or sequential)
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                trial_results = list(executor.map(_run_single_trial, trial_args))
        else:
            trial_results = [_run_single_trial(a) for a in trial_args]

        # Compute statistics
        qbers = np.array([r["qber"] for r in trial_results])
        sfs = np.array([r["secret_fraction"] for r in trial_results])
        key_rates = np.array([r["key_rate_per_pulse"] for r in trial_results])
        sifteds = np.array([r["n_sifted"] for r in trial_results])
        secrets = np.array([r["n_secret_est"] for r in trial_results])
        aborteds = np.array([r["aborted"] for r in trial_results])

        # Handle NaN QBER values (from aborted runs)
        qbers_valid = qbers[~np.isnan(qbers)]
        n_valid = len(qbers_valid)

        if n_valid > 0:
            qber_mean = float(np.mean(qbers_valid))
            qber_std = float(np.std(qbers_valid, ddof=1)) if n_valid > 1 else 0.0
            qber_se = qber_std / np.sqrt(n_valid) if n_valid > 1 else 0.0
            qber_ci_low_raw = qber_mean - 1.96 * qber_se
            qber_ci_high_raw = qber_mean + 1.96 * qber_se
            # Clamp QBER CI to physical bounds [0.0, 0.5]
            qber_ci_low = max(0.0, min(0.5, qber_ci_low_raw))
            qber_ci_high = max(0.0, min(0.5, qber_ci_high_raw))
        else:
            qber_mean = float("nan")
            qber_std = float("nan")
            qber_ci_low = float("nan")
            qber_ci_high = float("nan")

        # Secret fraction: mean over all trials (including aborted, which have sf=0)
        sf_mean_all = float(np.mean(sfs))
        sf_std = float(np.std(sfs, ddof=1)) if n_trials > 1 else 0.0
        sf_se = sf_std / np.sqrt(n_trials) if n_trials > 1 else 0.0
        sf_ci_low = sf_mean_all - 1.96 * sf_se
        sf_ci_high = sf_mean_all + 1.96 * sf_se

        # Secret fraction: mean over non-aborted trials only
        nonaborted_mask = ~aborteds
        sfs_nonaborted = sfs[nonaborted_mask]
        if len(sfs_nonaborted) > 0:
            sf_mean_nonaborted = float(np.mean(sfs_nonaborted))
        else:
            sf_mean_nonaborted = 0.0

        abort_rate = float(np.mean(aborteds))
        nonabort_rate = 1.0 - abort_rate

        # Key rate statistics (key_rate_per_pulse is 0 for aborted trials)
        kr_mean = float(np.mean(key_rates))
        kr_std = float(np.std(key_rates, ddof=1)) if n_trials > 1 else 0.0
        kr_se = kr_std / np.sqrt(n_trials) if n_trials > 1 else 0.0
        kr_ci_low_raw = kr_mean - 1.96 * kr_se
        kr_ci_high_raw = kr_mean + 1.96 * kr_se
        # Clamp key rate CI to non-negative (key rate cannot be negative)
        kr_ci_low = max(0.0, kr_ci_low_raw)
        kr_ci_high = max(0.0, kr_ci_high_raw)

        out.append({
            "loss_db": float(loss_db),
            "flip_prob": float(flip_prob),
            "attack": str(attack),
            "n_trials": n_trials,
            # QBER statistics
            "qber_mean": qber_mean,
            "qber_std": qber_std,
            "qber_ci_low": qber_ci_low,
            "qber_ci_high": qber_ci_high,
            "qber_n_valid": n_valid,
            # Secret fraction statistics (all trials)
            "secret_fraction_mean": sf_mean_all,  # backward compat
            "secret_fraction_mean_all": sf_mean_all,
            "secret_fraction_std": sf_std,
            "secret_fraction_ci_low": max(0.0, min(1.0, sf_ci_low)),
            "secret_fraction_ci_high": max(0.0, min(1.0, sf_ci_high)),
            # Secret fraction (non-aborted trials only)
            "secret_fraction_mean_nonaborted": sf_mean_nonaborted,
            # Key rate per pulse statistics
            "key_rate_per_pulse_mean": kr_mean,
            "key_rate_per_pulse_std": kr_std,
            "key_rate_per_pulse_ci_low": kr_ci_low,
            "key_rate_per_pulse_ci_high": kr_ci_high,
            # Other statistics
            "n_sifted_mean": float(np.mean(sifteds)),
            "n_secret_est_mean": float(np.mean(secrets)),
            "abort_rate": abort_rate,
            "nonabort_rate": nonabort_rate,
        })

    return out


def sweep_loss_finite_key(
    loss_db_values: Sequence[float],
    flip_prob: float = 0.0,
    attack: Attack = "none",
    n_pulses: int = 200_000,
    seed: int = 0,
    detector: Optional[DetectorParams] = None,
    finite_key_params: Optional[FiniteKeyParams] = None,
) -> List[Dict[str, Any]]:
    """
    Sweep over loss values with finite-key analysis.

    Runs BB84 simulation and computes finite-key secret key length
    using Hoeffding bounds for parameter estimation.

    Parameters
    ----------
    loss_db_values : Sequence[float]
        Channel loss values in dB.
    flip_prob : float
        Intrinsic bit-flip probability.
    attack : Attack
        Attack type ("none" or "intercept_resend").
    n_pulses : int
        Number of pulses per simulation.
    seed : int
        Base random seed.
    detector : DetectorParams, optional
        Detector model parameters.
    finite_key_params : FiniteKeyParams, optional
        Finite-key security parameters.

    Returns
    -------
    List[Dict[str, Any]]
        List of result dictionaries with both asymptotic and finite-key metrics.
    """
    det = detector if detector is not None else DEFAULT_DETECTOR
    fk_params = finite_key_params if finite_key_params is not None else FiniteKeyParams()
    out: List[Dict[str, Any]] = []

    for i, loss_db in enumerate(loss_db_values):
        s = simulate_bb84(
            n_pulses=n_pulses,
            loss_db=float(loss_db),
            flip_prob=float(flip_prob),
            attack=attack,
            seed=seed + i,
            detector=det,
        )

        # Compute finite-key analysis
        # If BB84 aborted, propagate abort to finite-key (no key extractable)
        if s.aborted:
            fk_result = {
                "qber_hat": s.qber,
                "qber_upper": 1.0,  # Worst case
                "qber_lower": 0.0,
                "n_sifted": s.n_sifted,
                "m_pe": 0,
                "ell_bits": 0.0,
                "l_secret_bits": 0.0,
                "key_rate_per_pulse": 0.0,
                "key_rate_per_sifted": 0.0,
                "aborted": True,
                "leak_ec_bits": 0.0,
                "delta_eps_bits": 0.0,
            }
        else:
            n_errors = int(round(s.qber * s.n_sifted)) if s.n_sifted > 0 else 0
            fk_result = finite_key_rate_per_pulse(
                n_sent=s.n_sent,
                n_sifted=s.n_sifted,
                n_errors=n_errors,
                params=fk_params,
            )

        # Compare asymptotic vs finite
        comparison = compare_asymptotic_vs_finite(
            n_sent=s.n_sent,
            n_sifted=s.n_sifted,
            qber_observed=s.qber if not s.aborted else 0.5,
            params=fk_params,
        )

        out.append({
            "loss_db": float(loss_db),
            "flip_prob": float(flip_prob),
            "attack": str(attack),
            "n_sent": s.n_sent,
            "n_received": s.n_received,
            "n_sifted": s.n_sifted,
            # Asymptotic results
            "qber": s.qber,
            "secret_fraction": s.secret_fraction,
            "key_rate_per_pulse_asymptotic": s.key_rate_per_pulse,
            "n_secret_est_asymptotic": s.n_secret_est,
            "aborted": bool(s.aborted),
            # Finite-key results
            "qber_hat": fk_result["qber_hat"],
            "qber_upper": fk_result["qber_upper"],
            "m_pe": fk_result["m_pe"],
            "ell_bits": fk_result.get("ell_bits", fk_result["l_secret_bits"]),
            "l_secret_bits": fk_result["l_secret_bits"],
            "key_rate_per_pulse_finite": fk_result["key_rate_per_pulse"],
            "leak_ec_bits": fk_result["leak_ec_bits"],
            "delta_eps_bits": fk_result["delta_eps_bits"],
            "finite_key_aborted": fk_result["aborted"],
            # Comparison metrics
            "asymptotic_rate": comparison["asymptotic_rate"],
            "finite_rate": comparison["finite_rate"],
            "finite_size_penalty": comparison["penalty_factor"],
            # Security parameters used
            "eps_pe": fk_params.eps_pe,
            "eps_sec": fk_params.eps_sec,
            "eps_cor": fk_params.eps_cor,
            "eps_total": fk_params.eps_total,
            "ec_efficiency": fk_params.ec_efficiency,
            "f_ec": fk_params.ec_efficiency,
            "pe_frac": fk_params.pe_frac,
        })

    return out


def sweep_finite_key_vs_n_sent(
    loss_db: float,
    n_sent_values: Sequence[int],
    flip_prob: float = 0.0,
    attack: Attack = "none",
    seed: int = 0,
    detector: Optional[DetectorParams] = None,
    finite_key_params: Optional[FiniteKeyParams] = None,
) -> List[Dict[str, Any]]:
    """
    Sweep finite-key rate versus total sent pulses for a fixed loss.

    Parameters
    ----------
    loss_db : float
        Channel loss in dB.
    n_sent_values : Sequence[int]
        Total pulses to simulate per point.
    flip_prob : float
        Intrinsic bit-flip probability.
    attack : Attack
        Attack type (\"none\" or \"intercept_resend\").
    seed : int
        Base random seed.
    detector : DetectorParams, optional
        Detector model parameters.
    finite_key_params : FiniteKeyParams, optional
        Finite-key security parameters.

    Returns
    -------
    List[Dict[str, Any]]
        List of finite-key results keyed by n_sent.
    """
    det = detector if detector is not None else DEFAULT_DETECTOR
    fk_params = finite_key_params if finite_key_params is not None else FiniteKeyParams()
    out: List[Dict[str, Any]] = []

    for i, n_sent in enumerate(n_sent_values):
        s = simulate_bb84(
            n_pulses=int(n_sent),
            loss_db=float(loss_db),
            flip_prob=float(flip_prob),
            attack=attack,
            seed=seed + i,
            detector=det,
        )

        if s.aborted or s.n_sifted <= 0 or np.isnan(s.qber):
            fk_result = {
                "qber_hat": s.qber,
                "qber_upper": 1.0,
                "m_pe": 0,
                "ell_bits": 0.0,
                "l_secret_bits": 0.0,
                "key_rate_per_pulse": 0.0,
                "leak_ec_bits": 0.0,
                "delta_eps_bits": 0.0,
            }
        else:
            n_errors = int(round(s.qber * s.n_sifted))
            fk_result = finite_key_rate_per_pulse(
                n_sent=s.n_sent,
                n_sifted=s.n_sifted,
                n_errors=n_errors,
                params=fk_params,
            )

        out.append({
            "loss_db": float(loss_db),
            "attack": str(attack),
            "n_sent": s.n_sent,
            "n_sifted": s.n_sifted,
            "qber_hat": fk_result["qber_hat"],
            "qber_upper": fk_result["qber_upper"],
            "m_pe": fk_result["m_pe"],
            "ell_bits": fk_result.get("ell_bits", fk_result["l_secret_bits"]),
            "leak_ec_bits": fk_result["leak_ec_bits"],
            "delta_eps_bits": fk_result["delta_eps_bits"],
            "key_rate_per_pulse_finite": fk_result["key_rate_per_pulse"],
        })

    return out


def compute_summary_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics across a sweep.

    Parameters
    ----------
    records : List[Dict[str, Any]]
        List of sweep result dictionaries.

    Returns
    -------
    Dict[str, Any]
        Summary statistics including min/max loss where key is extractable.
    """
    loss_vals = [r["loss_db"] for r in records]

    # Find key metric field (handle both regular and CI sweeps)
    # For CI sweeps, use secret_fraction_mean_nonaborted to define positive key region
    if "secret_fraction_mean_nonaborted" in records[0]:
        sf_key = "secret_fraction_mean_nonaborted"
    elif "secret_fraction_mean" in records[0]:
        sf_key = "secret_fraction_mean"
    else:
        sf_key = "secret_fraction"

    sf_vals = [r[sf_key] for r in records]

    # Find loss range where secret fraction > 0
    positive_sf = [(l, sf) for l, sf in zip(loss_vals, sf_vals) if sf > 0]

    if positive_sf:
        max_loss_positive = max(l for l, _ in positive_sf)
        min_loss_positive = min(l for l, _ in positive_sf)
    else:
        max_loss_positive = None
        min_loss_positive = None

    return {
        "loss_min": min(loss_vals),
        "loss_max": max(loss_vals),
        "loss_range_positive_key": {
            "min": min_loss_positive,
            "max": max_loss_positive,
        },
        "n_points": len(records),
    }


def sweep_pass(
    elevation_deg_values: Sequence[float],
    time_s_values: Sequence[float],
    flip_prob: float = 0.0,
    attack: Attack = "none",
    n_pulses: int = 200_000,
    seed: int = 0,
    detector: Optional[DetectorParams] = None,
    link_params: Optional[FreeSpaceLinkParams] = None,
    turbulence: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Sweep over a satellite pass and compute QKD metrics.

    Uses the free-space optical link model for physically-grounded loss
    calculations.

    Parameters
    ----------
    elevation_deg_values : Sequence[float]
        Elevation angles over the pass.
    time_s_values : Sequence[float]
        Time values corresponding to elevations.
    flip_prob : float
        Intrinsic bit-flip probability.
    attack : Attack
        Attack type.
    n_pulses : int
        Number of pulses per time step.
    seed : int
        Random seed.
    detector : DetectorParams, optional
        Base detector parameters (p_bg may be modified for day/night).
    link_params : FreeSpaceLinkParams, optional
        Free-space link parameters.
    turbulence : bool
        Enable lognormal turbulence fading.

    Returns
    -------
    Tuple[List[Dict[str, Any]], Dict[str, Any]]
        (list of per-point results, secure window summary)
    """
    det_base = detector if detector is not None else DEFAULT_DETECTOR
    link = link_params if link_params is not None else FreeSpaceLinkParams()

    rng = np.random.default_rng(seed)
    out: List[Dict[str, Any]] = []

    for i, (el, t) in enumerate(zip(elevation_deg_values, time_s_values)):
        el_f = float(el)
        t_f = float(t)

        # Compute loss from free-space model
        loss_db = free_space_loss_db(el_f, link)

        # Adjust background for day/night
        p_bg_eff = effective_background_prob(det_base.p_bg, link)
        det = DetectorParams(eta=det_base.eta, p_bg=p_bg_eff)

        # Apply turbulence fading if enabled
        if turbulence and link.sigma_ln > 0:
            fading = sample_turbulence_fading(1, link.sigma_ln, rng)[0]
            # Fading < 1 means additional loss
            if fading < 1.0 and fading > 0:
                loss_db += -10.0 * np.log10(fading)

        # Run BB84 simulation
        s = simulate_bb84(
            n_pulses=n_pulses,
            loss_db=loss_db,
            flip_prob=flip_prob,
            attack=attack,
            seed=seed + i,
            detector=det,
        )

        out.append({
            "time_s": t_f,
            "elevation_deg": el_f,
            "loss_db": loss_db,
            "p_bg_effective": p_bg_eff,
            "flip_prob": float(flip_prob),
            "attack": str(attack),
            "n_sent": s.n_sent,
            "n_received": s.n_received,
            "n_sifted": s.n_sifted,
            "qber": s.qber,
            "secret_fraction": s.secret_fraction,
            "key_rate_per_pulse": s.key_rate_per_pulse,
            "n_secret_est": s.n_secret_est,
            "aborted": bool(s.aborted),
        })

    # Compute secure window
    key_rates = np.array([r["key_rate_per_pulse"] for r in out])
    time_step_s = float(time_s_values[1] - time_s_values[0]) if len(time_s_values) > 1 else 1.0
    secure_window = estimate_secure_window(out, key_rates, time_step_s)

    return out, secure_window
