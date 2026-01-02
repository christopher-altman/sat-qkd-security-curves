"""
Decoy-State BB84 Module (Asymptotic Analysis)

This module implements decoy-state BB84 quantum key distribution with
asymptotic key rate analysis. Decoy states are used in practical QKD
to defeat photon-number-splitting (PNS) attacks that exploit multi-photon
pulses from weak coherent sources.

Theory
------
In standard BB84 with weak coherent pulses, the source emits pulses with
Poissonian photon number distribution. Multi-photon pulses are vulnerable
to PNS attacks where Eve keeps one photon and forwards the rest.

Decoy-state protocols use multiple intensity levels to estimate the
single-photon contribution to the observed gain and error rate, enabling
secure key extraction even with imperfect sources.

This implementation uses the vacuum + weak decoy protocol with three
intensity levels:
  - Signal (mu_s): Main intensity for key generation (~0.6)
  - Decoy (mu_d): Lower intensity for estimation (~0.1)
  - Vacuum (mu_v=0): No photons, measures background

Key Rate Formula (Asymptotic)
-----------------------------
R = p_s * { Q_1 * [1 - h2(e_1)] - Q_mu_s * f_ec * h2(E_mu_s) }

Where:
  - p_s: Probability of sending signal state
  - Q_1: Single-photon gain (estimated via decoy bounds)
  - e_1: Single-photon error rate (estimated via decoy bounds)
  - Q_mu_s: Overall gain for signal intensity
  - E_mu_s: Overall QBER for signal intensity
  - f_ec: Error correction efficiency (~1.16)
  - h2: Binary entropy function

References
----------
- Hwang, W.-Y. (2003). Quantum key distribution with high loss: toward global secure communication. PRL 91, 057901.
- Lo, H.-K., Ma, X., & Chen, K. (2005). Decoy state quantum key distribution. PRL 94, 230504.
- Ma, X., et al. (2005). Practical decoy state for quantum key distribution. PRA 72, 012326.

Limitations
-----------
This is an ASYMPTOTIC analysis for demonstrating the decoy-state concept.
It does not include:
  - Finite-key effects (statistical fluctuations in parameter estimation)
  - Composable security bounds
  - Detector efficiency mismatch attacks
  - Source imperfections beyond Poissonian statistics
"""
from __future__ import annotations
import math
from typing import Dict, Any, List, Sequence, Optional
from dataclasses import dataclass
import numpy as np

from .helpers import h2
from .detector import DetectorParams, DEFAULT_DETECTOR


@dataclass(frozen=True)
class DecoyParams:
    """
    Parameters for decoy-state BB84 protocol.

    Attributes
    ----------
    mu_s : float
        Signal intensity (mean photon number). Typical: 0.4-0.8.
    mu_d : float
        Decoy intensity (mean photon number). Typical: 0.1-0.2.
    mu_v : float
        Vacuum intensity. Must be 0.
    p_s : float
        Probability of sending signal state.
    p_d : float
        Probability of sending decoy state.
    p_v : float
        Probability of sending vacuum state.
        Note: p_s + p_d + p_v must equal 1.
    mu_s_sigma : float
        Std dev for signal intensity fluctuations (>= 0).
    mu_d_sigma : float
        Std dev for decoy intensity fluctuations (>= 0).
    """
    mu_s: float = 0.6
    mu_d: float = 0.1
    mu_v: float = 0.0
    p_s: float = 0.8
    p_d: float = 0.15
    p_v: float = 0.05
    mu_s_sigma: float = 0.0
    mu_d_sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.mu_v != 0.0:
            raise ValueError("Vacuum intensity mu_v must be 0")
        if self.mu_s <= self.mu_d:
            raise ValueError("Signal intensity must exceed decoy intensity")
        if not math.isclose(self.p_s + self.p_d + self.p_v, 1.0, rel_tol=1e-9):
            raise ValueError(f"Probabilities must sum to 1, got {self.p_s + self.p_d + self.p_v}")
        if any(p < 0 for p in [self.p_s, self.p_d, self.p_v]):
            raise ValueError("All probabilities must be non-negative")
        if self.mu_s_sigma < 0.0:
            raise ValueError("mu_s_sigma must be >= 0")
        if self.mu_d_sigma < 0.0:
            raise ValueError("mu_d_sigma must be >= 0")


DEFAULT_DECOY = DecoyParams()


def poisson_n(mu: float, n: int) -> float:
    """Probability of n photons for mean intensity mu (Poisson distribution)."""
    if mu == 0:
        return 1.0 if n == 0 else 0.0
    return (mu ** n) * math.exp(-mu) / math.factorial(n)


def simulate_decoy_bb84(
    n_pulses: int = 200_000,
    loss_db: float = 0.0,
    flip_prob: float = 0.0,
    decoy: Optional[DecoyParams] = None,
    detector: Optional[DetectorParams] = None,
    ec_efficiency: float = 1.16,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """
    Simulate decoy-state BB84 and compute asymptotic key rate.

    This simulation:
    1. Generates pulses with intensity chosen by probability distribution
    2. For each pulse, draws photon number from Poisson distribution
    3. Applies channel loss and detection with background clicks
    4. Estimates single-photon parameters via decoy-state bounds
    5. Computes asymptotic secret key rate

    Parameters
    ----------
    n_pulses : int
        Total number of pulses to send.
    loss_db : float
        Channel loss in dB.
    flip_prob : float
        Intrinsic bit-flip probability.
    decoy : DecoyParams, optional
        Decoy protocol parameters.
    detector : DetectorParams, optional
        Detector model parameters.
    ec_efficiency : float
        Error correction efficiency factor.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Results including gains, error rates, bounds, and key rate.
    """
    rng = np.random.default_rng(seed)
    dec = decoy if decoy is not None else DEFAULT_DECOY
    det = detector if detector is not None else DEFAULT_DETECTOR

    # Channel parameters
    trans = 10 ** (-loss_db / 10.0)

    # Allocate pulses to each intensity
    intensity_choice = rng.choice(
        [0, 1, 2],
        size=n_pulses,
        p=[dec.p_s, dec.p_d, dec.p_v]
    )
    n_signal = np.sum(intensity_choice == 0)
    n_decoy = np.sum(intensity_choice == 1)
    n_vacuum = np.sum(intensity_choice == 2)

    def simulate_intensity(mu: float, n_total: int) -> tuple:
        """Simulate n_total pulses at intensity mu, return (n_click, n_error)."""
        if n_total == 0:
            return 0, 0

        # Generate photon numbers (Poisson)
        photon_nums = rng.poisson(mu, size=n_total)

        # Alice's bits and bases
        a_bits = rng.integers(0, 2, size=n_total, dtype=np.int8)
        a_basis = rng.integers(0, 2, size=n_total, dtype=np.int8)

        # For each pulse, at least one photon must survive for signal click
        # P(at least one photon detected) = 1 - (1 - eta_ch)^n for n photons
        survive_probs = 1.0 - np.power(1.0 - det.eta * trans, photon_nums)
        sig_click = rng.random(n_total) < survive_probs

        # Background clicks
        bg_click = rng.random(n_total) < det.p_bg
        click = sig_click | bg_click
        bg_only = bg_click & (~sig_click)

        # Bob's measurement
        idx = np.where(click)[0]
        if len(idx) == 0:
            return 0, 0

        b_basis = rng.integers(0, 2, size=len(idx), dtype=np.int8)

        # Determine Bob's bits
        b_bits = a_bits[idx].copy()

        # Basis mismatch -> random bit
        mismatch = b_basis != a_basis[idx]
        b_bits[mismatch] = rng.integers(0, 2, size=np.sum(mismatch), dtype=np.int8)

        # Background-only clicks -> random bit
        bg_mask = bg_only[idx]
        b_bits[bg_mask] = rng.integers(0, 2, size=np.sum(bg_mask), dtype=np.int8)

        # Intrinsic noise
        if flip_prob > 0:
            flips = rng.random(len(idx)) < flip_prob
            b_bits[flips] ^= 1

        # Sifting
        sift = b_basis == a_basis[idx]
        a_sift = a_bits[idx][sift]
        b_sift = b_bits[sift]

        n_sifted = len(a_sift)
        if n_sifted == 0:
            return 0, 0

        n_errors = int(np.sum(a_sift != b_sift))
        return n_sifted, n_errors

    afterpulse_enabled = det.p_afterpulse > 0.0 and det.afterpulse_window > 0
    realism_enabled = (
        dec.mu_s_sigma > 0.0
        or dec.mu_d_sigma > 0.0
        or afterpulse_enabled
        or det.dead_time_pulses > 0
        or det.eta_z != det.eta
        or det.eta_x != det.eta
    )

    def _sample_truncated_mu(mean: float, sigma: float, max_tries: int = 10) -> float:
        if sigma <= 0.0:
            return mean
        for _ in range(max_tries):
            val = float(rng.normal(mean, sigma))
            if val >= 0.0:
                return val
        return max(0.0, mean)

    if realism_enabled:
        n_sift_s = n_err_s = 0
        n_sift_d = n_err_d = 0
        n_sift_v = n_err_v = 0
        afterpulse_counter = 0
        afterpulse_age = 0
        dead_time_counter = 0

        for i in range(n_pulses):
            has_afterpulse = afterpulse_counter > 0
            if afterpulse_counter > 0:
                afterpulse_counter -= 1
                afterpulse_age += 1
            if dead_time_counter > 0:
                dead_time_counter -= 1
                continue

            intensity = int(intensity_choice[i])
            if intensity == 0:
                mu = dec.mu_s
                sigma = dec.mu_s_sigma
            elif intensity == 1:
                mu = dec.mu_d
                sigma = dec.mu_d_sigma
            else:
                mu = dec.mu_v
                sigma = 0.0

            mu_eff = _sample_truncated_mu(mu, sigma)

            photon_num = int(rng.poisson(mu_eff))
            a_bit = int(rng.integers(0, 2))
            a_basis = int(rng.integers(0, 2))
            b_basis = int(rng.integers(0, 2))

            eta_eff = det.eta_z if b_basis == 0 else det.eta_x
            eta_ch = eta_eff * trans
            if photon_num > 0:
                survive_prob = 1.0 - (1.0 - eta_ch) ** photon_num
            else:
                survive_prob = 0.0

            sig_click = rng.random() < survive_prob
            if has_afterpulse:
                if det.afterpulse_decay > 0.0:
                    decay = math.exp(-afterpulse_age / det.afterpulse_decay)
                else:
                    decay = 1.0
                afterpulse_bump = det.p_afterpulse * decay
            else:
                afterpulse_bump = 0.0
            p_bg_eff = det.p_bg + afterpulse_bump
            if p_bg_eff > 1.0:
                p_bg_eff = 1.0
            bg_click = rng.random() < p_bg_eff
            click = sig_click or bg_click

            if not click:
                continue

            afterpulse_counter = det.afterpulse_window
            afterpulse_age = 0
            dead_time_counter = det.dead_time_pulses

            b_bit = a_bit
            if b_basis != a_basis:
                b_bit = int(rng.integers(0, 2))
            if bg_click and not sig_click:
                b_bit = int(rng.integers(0, 2))
            if flip_prob > 0.0 and rng.random() < flip_prob:
                b_bit ^= 1

            if b_basis == a_basis:
                if intensity == 0:
                    n_sift_s += 1
                    if b_bit != a_bit:
                        n_err_s += 1
                elif intensity == 1:
                    n_sift_d += 1
                    if b_bit != a_bit:
                        n_err_d += 1
                else:
                    n_sift_v += 1
                    if b_bit != a_bit:
                        n_err_v += 1
    else:
        # Simulate each intensity with vectorized sampling (fast path).
        n_sift_s, n_err_s = simulate_intensity(dec.mu_s, int(n_signal))
        n_sift_d, n_err_d = simulate_intensity(dec.mu_d, int(n_decoy))
        n_sift_v, n_err_v = simulate_intensity(dec.mu_v, int(n_vacuum))

    # Compute gains (sifted detection probability)
    # Note: We measure sifted gain = n_sifted / n_sent (already includes 1/2 sifting factor)
    Q_s = n_sift_s / n_signal if n_signal > 0 else 0.0
    Q_d = n_sift_d / n_decoy if n_decoy > 0 else 0.0
    Q_v = n_sift_v / n_vacuum if n_vacuum > 0 else 0.0

    # Compute error rates
    E_s = n_err_s / n_sift_s if n_sift_s > 0 else 0.5
    E_d = n_err_d / n_sift_d if n_sift_d > 0 else 0.5
    E_v = n_err_v / n_sift_v if n_sift_v > 0 else 0.5

    # Decoy-state bounds for single-photon parameters
    # Using vacuum + weak decoy protocol (GLLP + decoy bounds)

    # Lower bound on Y_1 (single-photon yield)
    # Y_1 >= (mu_s * Q_d * exp(mu_d) - mu_d * Q_s * exp(mu_s) - (mu_s - mu_d) * Q_v) / (mu_s * mu_d - mu_d^2)
    # Simplified for mu_v = 0:
    mu_s, mu_d = dec.mu_s, dec.mu_d

    # More stable formula using the standard decoy bound
    # Y_1_lower = (Q_d * exp(mu_d) - Q_v * exp(mu_d) * (mu_d^2 / mu_s^2) - Q_s * exp(mu_s) * (...)) / ...
    # We use the simplified vacuum+weak decoy bound:

    exp_s = math.exp(mu_s)
    exp_d = math.exp(mu_d)

    # Y_1 lower bound (Eq. 4 from Lo, Ma, Chen 2005)
    # Y_1 >= (mu_s / (mu_s * mu_d - mu_d^2)) * (Q_d * exp(mu_d) - Q_v * (mu_d^2/2) - (mu_d^2 / mu_s^2) * Q_s * exp(mu_s))
    # Simplified for practical use:
    numerator = mu_s * (Q_d * exp_d - Q_v) - mu_d * mu_d * Q_v / 2 - (mu_d / mu_s) * Q_s * exp_s * mu_d
    denominator = mu_s * mu_d - mu_d * mu_d

    if denominator > 1e-12 and numerator > 0:
        Y1_lower = numerator / denominator
    else:
        # Fallback: simple estimate
        Y1_lower = max(0.0, (Q_d * exp_d - Q_v) / mu_d) if mu_d > 0 else 0.0

    # Ensure Y1 is in valid range
    Y1_lower = max(0.0, min(1.0, Y1_lower))

    # Q_1 = Y_1 * mu * exp(-mu) (single-photon gain contribution)
    Q1_lower = Y1_lower * mu_s * math.exp(-mu_s)

    # Upper bound on e_1 (single-photon error rate)
    # e_1 <= (E_d * Q_d * exp(mu_d) - e_0 * Y_0) / (Y_1 * mu_d)
    # where e_0 = 0.5 (background errors are random), Y_0 = Q_v * exp(0) = Q_v

    e0 = 0.5  # Background error rate
    Y0 = Q_v  # Vacuum yield (background clicks only)

    if Y1_lower > 1e-12 and mu_d > 0:
        e1_upper = (E_d * Q_d * exp_d - e0 * Y0) / (Y1_lower * mu_d)
        e1_upper = max(0.0, min(0.5, e1_upper))
    else:
        e1_upper = 0.5  # Pessimistic bound

    # Asymptotic key rate (per pulse, for signal states)
    # R = Q_1 * [1 - h2(e_1)] - Q_s * f_ec * h2(E_s)
    if Q1_lower > 0 and e1_upper < 0.5:
        key_rate_asymptotic = max(0.0,
            Q1_lower * (1.0 - h2(e1_upper)) - Q_s * ec_efficiency * h2(E_s)
        )
    else:
        key_rate_asymptotic = 0.0

    # Key rate per sifted signal bit
    if n_sift_s > 0:
        secret_fraction = key_rate_asymptotic * n_signal / n_sift_s
        secret_fraction = max(0.0, min(1.0, secret_fraction))
    else:
        secret_fraction = 0.0

    return {
        "loss_db": float(loss_db),
        "n_pulses": n_pulses,
        # Per-intensity statistics
        "n_signal": int(n_signal),
        "n_decoy": int(n_decoy),
        "n_vacuum": int(n_vacuum),
        "n_sift_signal": n_sift_s,
        "n_sift_decoy": n_sift_d,
        "n_sift_vacuum": n_sift_v,
        # Gains
        "Q_signal": float(Q_s),
        "Q_decoy": float(Q_d),
        "Q_vacuum": float(Q_v),
        # Error rates
        "E_signal": float(E_s),
        "E_decoy": float(E_d),
        "E_vacuum": float(E_v),
        # Decoy bounds
        "Y1_lower": float(Y1_lower),
        "Q1_lower": float(Q1_lower),
        "e1_upper": float(e1_upper),
        # Key rates
        "key_rate_asymptotic": float(key_rate_asymptotic),
        "secret_fraction": float(secret_fraction),
        # Parameters used
        "mu_s": float(dec.mu_s),
        "mu_d": float(dec.mu_d),
        "mu_v": float(dec.mu_v),
        "eta": float(det.eta),
        "p_bg": float(det.p_bg),
        "flip_prob": float(flip_prob),
        "ec_efficiency": float(ec_efficiency),
        "seed": seed,
    }


def sweep_decoy_loss(
    loss_db_values: Sequence[float],
    flip_prob: float = 0.0,
    n_pulses: int = 200_000,
    seed: int = 0,
    decoy: Optional[DecoyParams] = None,
    detector: Optional[DetectorParams] = None,
    n_trials: int = 1,
) -> List[Dict[str, Any]]:
    """
    Sweep over loss values for decoy-state BB84.

    Parameters
    ----------
    loss_db_values : Sequence[float]
        Channel loss values in dB.
    flip_prob : float
        Intrinsic bit-flip probability.
    n_pulses : int
        Number of pulses per simulation.
    seed : int
        Base random seed.
    decoy : DecoyParams, optional
        Decoy protocol parameters.
    detector : DetectorParams, optional
        Detector model parameters.
    n_trials : int
        Number of trials per loss value (for averaging).

    Returns
    -------
    List[Dict[str, Any]]
        List of result dictionaries.

    Notes
    -----
    Uses numpy.random.SeedSequence to generate independent RNG streams
    for each (loss_index, trial) pair. This avoids RNG collisions when
    sweeping over parameter grids with the same base seed.
    """
    out: List[Dict[str, Any]] = []

    # Create a SeedSequence from the base seed for proper RNG independence
    # Each (loss_index, trial) pair gets a unique, independent stream
    n_points = len(loss_db_values)
    total_streams = n_points * max(1, n_trials)
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(total_streams)

    for i, loss_db in enumerate(loss_db_values):
        if n_trials == 1:
            # Use integer seed derived from SeedSequence for compatibility
            point_seed = int(child_seeds[i].generate_state(1)[0])
            result = simulate_decoy_bb84(
                n_pulses=n_pulses,
                loss_db=float(loss_db),
                flip_prob=flip_prob,
                decoy=decoy,
                detector=detector,
                seed=point_seed,
            )
            out.append(result)
        else:
            # Multiple trials with averaging
            trial_results = []
            for t in range(n_trials):
                stream_idx = i * n_trials + t
                trial_seed = int(child_seeds[stream_idx].generate_state(1)[0])
                r = simulate_decoy_bb84(
                    n_pulses=n_pulses,
                    loss_db=float(loss_db),
                    flip_prob=flip_prob,
                    decoy=decoy,
                    detector=detector,
                    seed=trial_seed,
                )
                trial_results.append(r)

            # Average key metrics
            key_rates = np.array([r["key_rate_asymptotic"] for r in trial_results])
            secret_fracs = np.array([r["secret_fraction"] for r in trial_results])
            e_signals = np.array([r["E_signal"] for r in trial_results])

            # Compute CI for key rate (clamped to non-negative)
            kr_mean = float(np.mean(key_rates))
            kr_std = float(np.std(key_rates, ddof=1)) if n_trials > 1 else 0.0
            kr_se = kr_std / np.sqrt(n_trials) if n_trials > 1 else 0.0
            kr_ci_low = max(0.0, kr_mean - 1.96 * kr_se)
            kr_ci_high = kr_mean + 1.96 * kr_se

            avg_result = trial_results[0].copy()
            avg_result["key_rate_asymptotic"] = kr_mean
            avg_result["key_rate_std"] = kr_std
            avg_result["key_rate_ci_low"] = kr_ci_low
            avg_result["key_rate_ci_high"] = kr_ci_high
            avg_result["secret_fraction"] = float(np.mean(secret_fracs))
            avg_result["secret_fraction_std"] = float(np.std(secret_fracs, ddof=1)) if n_trials > 1 else 0.0
            avg_result["E_signal_mean"] = float(np.mean(e_signals))
            avg_result["n_trials"] = n_trials
            out.append(avg_result)

    return out
