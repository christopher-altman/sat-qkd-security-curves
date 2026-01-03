"""
Entanglement-based QKD (EB-QKD) expected-value simulator.

This module provides a minimal, calibration-ready EB-QKD harness with
passive basis choice at the receiver. It uses analytic expectations
instead of Monte Carlo to keep runs fast and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .detector import DetectorParams, DEFAULT_DETECTOR
from .finite_key import FiniteKeyParams, finite_key_rate_per_pulse


@dataclass(frozen=True)
class EBQKDParams:
    """Parameters for EB-QKD expected-value simulation."""
    loss_db: float = 0.0
    flip_prob: float = 0.0
    qber_abort_threshold: float = 0.11


def _expected_qber(
    p_sig: float,
    p_bg: float,
    flip_prob: float,
) -> float:
    """Expected QBER from signal/background mixing and intrinsic flips."""
    p_click = p_sig + p_bg - p_sig * p_bg
    if p_click <= 0:
        return float("nan")
    qber = (0.5 * p_bg + flip_prob * p_sig) / p_click
    return min(0.5, max(0.0, qber))


def simulate_eb_qkd_expected(
    n_pairs: int,
    params: Optional[EBQKDParams] = None,
    detector: Optional[DetectorParams] = None,
    finite_key: Optional[FiniteKeyParams] = None,
) -> Dict[str, Any]:
    """
    Simulate EB-QKD expected metrics with passive basis choice.

    Parameters
    ----------
    n_pairs : int
        Total entangled pairs generated.
    params : EBQKDParams, optional
        EB-QKD parameters (loss, noise, abort threshold).
    detector : DetectorParams, optional
        Detector parameters for the receiver.
    finite_key : FiniteKeyParams, optional
        Finite-key security parameters.

    Returns
    -------
    dict
        EB-QKD metrics including finite-key bounds.
    """
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive.")

    p = params if params is not None else EBQKDParams()
    det = detector if detector is not None else DEFAULT_DETECTOR
    fk = finite_key if finite_key is not None else FiniteKeyParams()

    eta_eff = det.eta
    p_sig = eta_eff * (10 ** (-p.loss_db / 10.0))
    p_bg = det.p_bg
    p_click = p_sig + p_bg - p_sig * p_bg

    qber_mean = _expected_qber(p_sig, p_bg, p.flip_prob)
    n_received = int(round(n_pairs * p_click))
    n_sifted = int(round(n_received * 0.5))
    n_errors = int(round((0.0 if qber_mean != qber_mean else qber_mean) * n_sifted))

    finite = finite_key_rate_per_pulse(
        n_sent=n_pairs,
        n_sifted=n_sifted,
        n_errors=n_errors,
        params=fk,
        qber_abort_threshold=p.qber_abort_threshold,
    )
    finite_key_info = finite["finite_key"]

    if finite_key_info["status"] == "secure" and n_sifted > 0:
        secret_fraction_finite = finite["ell_bits"] / n_sifted
    else:
        secret_fraction_finite = 0.0

    return {
        "n_pairs": int(n_pairs),
        "n_sent": int(n_pairs),
        "n_received": int(n_received),
        "n_sifted": int(n_sifted),
        "qber_mean": float(qber_mean),
        "qber_upper": float(finite["qber_upper"]),
        "secret_fraction_finite": float(secret_fraction_finite),
        "n_secret_est_finite": float(finite["ell_bits"]) if finite_key_info["status"] == "secure" else 0.0,
        "key_rate_per_pair": float(finite["key_rate_per_pulse"]) if finite_key_info["status"] == "secure" else 0.0,
        "aborted": bool(finite["aborted"]),
        "finite_key": finite_key_info,
        "meta": {
            "loss_db": float(p.loss_db),
            "flip_prob": float(p.flip_prob),
            "qber_abort_threshold": float(p.qber_abort_threshold),
            "eta": float(det.eta),
            "p_bg": float(det.p_bg),
            "eps_pe": float(fk.eps_pe),
            "eps_sec": float(fk.eps_sec),
            "eps_cor": float(fk.eps_cor),
            "pe_frac": float(fk.pe_frac),
        },
    }
