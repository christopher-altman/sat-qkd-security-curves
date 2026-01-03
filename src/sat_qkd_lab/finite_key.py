"""
Finite-key analysis for BB84 QKD.

This module provides conservative finite-size security analysis using
Hoeffding-type concentration bounds. It estimates the extractable secret
key length given finite sample sizes and explicit security parameters.

References:
- Tomamichel et al., "Tight finite-key analysis for quantum cryptography"
  Nature Communications 3, 634 (2012)
- Lim et al., "Concise security bounds for practical decoy-state QKD"
  Phys. Rev. A 89, 022307 (2014)
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional

from .helpers import h2


@dataclass(frozen=True)
class FiniteKeyParams:
    """Security parameters for finite-key analysis.

    Attributes
    ----------
    eps_pe : float
        Failure probability for parameter estimation (Hoeffding bound).
        Default: 1e-10
    eps_sec : float
        Secrecy failure probability (privacy amplification).
        Default: 1e-10
    eps_cor : float
        Correctness failure probability (error correction verification).
        Default: 1e-15
    ec_efficiency : float
        Error correction efficiency factor (>= 1.0).
        Leaked bits = ec_efficiency * n * h2(QBER).
        Default: 1.16 (typical for practical codes)
    pe_frac : float
        Fraction of sifted bits used for parameter estimation (0, 1].
        Default: 0.5
    m_pe : int, optional
        Explicit parameter estimation sample size. If provided, overrides pe_frac.
    """
    eps_pe: float = 1e-10
    eps_sec: float = 1e-10
    eps_cor: float = 1e-15
    ec_efficiency: float = 1.16
    pe_frac: float = 0.5
    m_pe: Optional[int] = None

    def __post_init__(self):
        if self.eps_pe <= 0 or self.eps_pe >= 1:
            raise ValueError(f"eps_pe must be in (0, 1), got {self.eps_pe}")
        if self.eps_sec <= 0 or self.eps_sec >= 1:
            raise ValueError(f"eps_sec must be in (0, 1), got {self.eps_sec}")
        if self.eps_cor <= 0 or self.eps_cor >= 1:
            raise ValueError(f"eps_cor must be in (0, 1), got {self.eps_cor}")
        if self.ec_efficiency < 1.0:
            raise ValueError(f"ec_efficiency must be >= 1.0, got {self.ec_efficiency}")
        if self.pe_frac <= 0 or self.pe_frac > 1.0:
            raise ValueError(f"pe_frac must be in (0, 1], got {self.pe_frac}")
        if self.m_pe is not None and self.m_pe < 1:
            raise ValueError(f"m_pe must be >= 1 when provided, got {self.m_pe}")

    @property
    def eps_total(self) -> float:
        """Total security parameter (sum of failure probabilities)."""
        return self.eps_pe + self.eps_sec + self.eps_cor

    @property
    def f_ec(self) -> float:
        """Alias for error correction inefficiency factor."""
        return self.ec_efficiency


def hoeffding_bound(n_samples: int, observed_rate: float, eps: float) -> float:
    """
    Compute Hoeffding upper bound on true rate given observed rate.

    Given n_samples observations with observed error rate p_hat,
    returns an upper bound p_upper such that:
        P(p_true > p_upper) <= eps

    Parameters
    ----------
    n_samples : int
        Number of samples used for estimation.
    observed_rate : float
        Observed error rate (e.g., QBER).
    eps : float
        Failure probability for the bound.

    Returns
    -------
    float
        Upper bound on true rate, clamped to [0, 1].
    """
    if n_samples <= 0:
        return 1.0  # No samples, worst case
    if eps <= 0:
        return 1.0  # Zero tolerance, worst case

    # Hoeffding: P(p_true > p_hat + delta) <= exp(-2 * n * delta^2)
    # Solving for delta: delta = sqrt(ln(1/eps) / (2n))
    delta = math.sqrt(math.log(1.0 / eps) / (2.0 * n_samples))

    upper = observed_rate + delta
    return min(1.0, max(0.0, upper))


def hoeffding_lower_bound(n_samples: int, observed_rate: float, eps: float) -> float:
    """
    Compute Hoeffding lower bound on true rate given observed rate.

    Parameters
    ----------
    n_samples : int
        Number of samples used for estimation.
    observed_rate : float
        Observed error rate.
    eps : float
        Failure probability for the bound.

    Returns
    -------
    float
        Lower bound on true rate, clamped to [0, 1].
    """
    if n_samples <= 0:
        return 0.0

    delta = math.sqrt(math.log(1.0 / eps) / (2.0 * n_samples))
    lower = observed_rate - delta
    return min(1.0, max(0.0, lower))


def finite_key_bounds(
    n_sifted: int,
    n_errors: int,
    eps_pe: float = 1e-10,
    m_pe: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute finite-key bounds on QBER using Hoeffding inequality.

    Parameters
    ----------
    n_sifted : int
        Number of sifted key bits.
    n_errors : int
        Number of errors observed in the sample.
    eps_pe : float
        Parameter estimation failure probability.
    m_pe : int, optional
        Sample size used for parameter estimation. If None, uses n_sifted.

    Returns
    -------
    dict
        Dictionary with keys:
        - qber_hat: observed QBER
        - qber_upper: upper bound on true QBER
        - qber_lower: lower bound on true QBER
        - n_sifted: sample size used
    """
    if n_sifted <= 0:
        return {
            "qber_hat": float("nan"),
            "qber_upper": 1.0,
            "qber_lower": 0.0,
            "n_sifted": 0,
        }

    qber_hat = n_errors / n_sifted
    m_eff = n_sifted if m_pe is None else max(1, min(n_sifted, m_pe))
    qber_upper = hoeffding_bound(m_eff, qber_hat, eps_pe)
    qber_lower = hoeffding_lower_bound(m_eff, qber_hat, eps_pe)

    return {
        "qber_hat": qber_hat,
        "qber_upper": min(0.5, qber_upper),  # QBER physical max is 0.5
        "qber_lower": qber_lower,
        "n_sifted": n_sifted,
        "m_pe": m_eff,
    }


def finite_key_secret_length(
    n_sifted: int,
    qber_upper: float,
    params: Optional[FiniteKeyParams] = None,
    qber_abort_threshold: float = 0.11,
) -> Dict[str, Any]:
    """
    Compute finite-key secret key length using conservative bounds.

    Uses a toy-but-recognizable BB84 finite-key bound:
        ell = n * max(0, 1 - 2*h2(Q_upper)) - leak_EC - delta_eps_bits

    where:
        - n is the number of sifted bits available for key generation
        - h is binary entropy
        - Q_upper is the upper bound on QBER
        - leak_EC is error correction leakage
        - delta_eps_bits = 2*log2(2/eps_sec) + log2(2/eps_cor)

    Parameters
    ----------
    n_sifted : int
        Number of sifted key bits.
    qber_upper : float
        Upper bound on QBER (from parameter estimation).
    params : FiniteKeyParams, optional
        Security parameters. Uses defaults if None.
    qber_abort_threshold : float
        QBER threshold for abort. Default 0.11.

    Returns
    -------
    dict
        Dictionary with:
        - ell_bits: extractable secret key bits (float, >= 0)
        - l_secret_bits: extractable secret key bits (legacy alias, >= 0)
        - key_rate_per_sifted: secret bits per sifted bit
        - aborted: whether protocol would abort
        - leak_ec_bits: bits leaked to error correction
        - delta_eps_bits: finite-key epsilon penalty in bits
    """
    if params is None:
        params = FiniteKeyParams()

    if n_sifted <= 0:
        return {
            "ell_bits": 0.0,
            "l_secret_bits": 0.0,
            "key_rate_per_sifted": 0.0,
            "aborted": False,
            "leak_ec_bits": 0.0,
            "delta_eps_bits": 0.0,
            "qber_upper": qber_upper,
            "finite_key_bound": 0.0,
            "finite_key_status": "insecure",
            "finite_key_reason": "NO-SECRET-KEY: no sifted bits",
        }

    # Error correction leakage: f_ec * n * h2(Q)
    # Use qber_upper for conservative estimate
    leak_ec_bits = params.ec_efficiency * n_sifted * h2(qber_upper)

    # Finite-key penalty term for secrecy and correctness
    delta_eps_bits = 2.0 * math.log2(2.0 / params.eps_sec) + math.log2(2.0 / params.eps_cor)

    # Secret fraction for BB84 (phase error = bit error), with finite-size penalty
    secret_fraction = max(0.0, 1.0 - 2.0 * h2(qber_upper))

    # Extractable bits
    raw_bound_bits = n_sifted * secret_fraction - leak_ec_bits - delta_eps_bits

    aborted = bool(qber_upper > qber_abort_threshold or qber_upper >= 0.5)
    if aborted:
        status = "insecure"
        reason = "NO-SECRET-KEY: qber_upper exceeds abort threshold"
    elif raw_bound_bits <= 0.0:
        status = "insecure"
        reason = "NO-SECRET-KEY: finite-key bound non-positive"
    else:
        status = "secure"
        reason = "finite-key bound positive"

    ell_bits = max(0.0, raw_bound_bits) if status == "secure" else 0.0

    key_rate_per_sifted = ell_bits / n_sifted if n_sifted > 0 and status == "secure" else 0.0

    return {
        "ell_bits": ell_bits,
        "l_secret_bits": ell_bits,
        "key_rate_per_sifted": key_rate_per_sifted,
        "aborted": aborted,
        "leak_ec_bits": leak_ec_bits,
        "delta_eps_bits": delta_eps_bits,
        "qber_upper": qber_upper,
        "finite_key_bound": raw_bound_bits,
        "finite_key_status": status,
        "finite_key_reason": reason,
    }


def finite_key_rate_per_pulse(
    n_sent: int,
    n_sifted: int,
    n_errors: int,
    params: Optional[FiniteKeyParams] = None,
    qber_abort_threshold: float = 0.11,
) -> Dict[str, Any]:
    """
    Compute finite-key rate per sent pulse (end-to-end).

    This is the main entry point for finite-key analysis of a BB84 run.
    It combines parameter estimation bounds with secret key length estimation.

    Parameters
    ----------
    n_sent : int
        Total number of pulses sent.
    n_sifted : int
        Number of sifted key bits.
    n_errors : int
        Number of errors in sifted key (for QBER estimation).
    params : FiniteKeyParams, optional
        Security parameters.
    qber_abort_threshold : float
        QBER threshold for abort.

    Returns
    -------
    dict
        Complete finite-key analysis results including:
        - All fields from finite_key_bounds
        - All fields from finite_key_secret_length
        - key_rate_per_pulse: secret bits per sent pulse
        - n_sent: total pulses
    """
    if params is None:
        params = FiniteKeyParams()

    # Step 1: Bound QBER with parameter estimation
    if n_sifted <= 0:
        m_pe = 0
    elif params.m_pe is not None:
        m_pe = max(1, min(n_sifted, params.m_pe))
    else:
        m_pe = max(1, int(params.pe_frac * n_sifted))

    bounds = finite_key_bounds(n_sifted, n_errors, params.eps_pe, m_pe=m_pe)

    # Step 2: Compute secret key length
    secret = finite_key_secret_length(
        n_sifted,
        bounds["qber_upper"],
        params,
        qber_abort_threshold,
    )

    # Step 3: Compute per-pulse rate
    key_rate_per_pulse = secret["ell_bits"] / n_sent if n_sent > 0 else 0.0
    finite_key = {
        "bound": float(secret["finite_key_bound"]),
        "status": secret["finite_key_status"],
        "reason": secret["finite_key_reason"],
    }

    return {
        **bounds,
        **secret,
        "key_rate_per_pulse": key_rate_per_pulse,
        "n_sent": n_sent,
        "m_pe": m_pe,
        "pe_frac": params.pe_frac,
        "eps_pe": params.eps_pe,
        "eps_sec": params.eps_sec,
        "eps_cor": params.eps_cor,
        "eps_total": params.eps_total,
        "ec_efficiency": params.ec_efficiency,
        "f_ec": params.ec_efficiency,
        "finite_key": finite_key,
    }


def composable_finite_key_report(
    n_sent: int,
    n_sifted: int,
    n_errors: int,
    params: Optional[FiniteKeyParams] = None,
    qber_abort_threshold: float = 0.11,
) -> Dict[str, Any]:
    """
    Produce a composable finite-key bookkeeping report.

    Explicitly tracks:
    - error correction leakage
    - privacy amplification epsilon penalty term
    - parameter estimation sample size and bounds
    - epsilons combined end-to-end

    Bound used: Hoeffding inequality for QBER estimation.
    """
    if params is None:
        params = FiniteKeyParams()

    result = finite_key_rate_per_pulse(
        n_sent=n_sent,
        n_sifted=n_sifted,
        n_errors=n_errors,
        params=params,
        qber_abort_threshold=qber_abort_threshold,
    )

    return {
        "n_sent": n_sent,
        "n_sifted": n_sifted,
        "n_errors": n_errors,
        "qber_hat": result["qber_hat"],
        "qber_upper": result["qber_upper"],
        "m_pe": result["m_pe"],
        "pe_frac": result["pe_frac"],
        "eps_pe": result["eps_pe"],
        "eps_sec": result["eps_sec"],
        "eps_cor": result["eps_cor"],
        "eps_total": result["eps_total"],
        "leak_ec_bits": result["leak_ec_bits"],
        "privacy_amplification_term_bits": result["delta_eps_bits"],
        "secret_bits": result["ell_bits"],
        "key_rate_per_pulse": result["key_rate_per_pulse"],
        "aborted": result["aborted"],
        "finite_key": result["finite_key"],
        "bound": {
            "name": "Hoeffding",
            "description": "QBER upper bound using Hoeffding inequality.",
        },
    }


def compare_asymptotic_vs_finite(
    n_sent: int,
    n_sifted: int,
    qber_observed: float,
    params: Optional[FiniteKeyParams] = None,
) -> Dict[str, Any]:
    """
    Compare asymptotic and finite-key rates for the same run.

    Useful for understanding the finite-size penalty.

    Parameters
    ----------
    n_sent : int
        Total pulses sent.
    n_sifted : int
        Sifted key bits.
    qber_observed : float
        Observed QBER.
    params : FiniteKeyParams, optional
        Security parameters.

    Returns
    -------
    dict
        Comparison with asymptotic_rate, finite_rate, and penalty_factor.
    """
    if params is None:
        params = FiniteKeyParams()

    n_errors = int(round(qber_observed * n_sifted))

    # Finite-key analysis
    finite = finite_key_rate_per_pulse(n_sent, n_sifted, n_errors, params)

    # Asymptotic rate (for comparison)
    secret_fraction_asymptotic = max(0.0, 1.0 - params.ec_efficiency * h2(qber_observed) - h2(qber_observed))
    sift_prob = n_sifted / n_sent if n_sent > 0 else 0.0
    asymptotic_rate = sift_prob * secret_fraction_asymptotic

    # Penalty factor
    penalty = 1.0 - (finite["key_rate_per_pulse"] / asymptotic_rate) if asymptotic_rate > 0 else 1.0

    return {
        "asymptotic_rate": asymptotic_rate,
        "finite_rate": finite["key_rate_per_pulse"],
        "penalty_factor": penalty,
        "finite_key_bits": finite["l_secret_bits"],
        "n_sent": n_sent,
        "n_sifted": n_sifted,
        "qber_observed": qber_observed,
        "qber_upper": finite["qber_upper"],
    }
