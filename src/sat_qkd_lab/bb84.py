from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from .helpers import h2, RunSummary
from .detector import DetectorParams, DEFAULT_DETECTOR
from .attacks import Attack, AttackConfig, AttackState, apply_attack, apply_time_shift

def simulate_bb84(
    n_pulses: int = 200_000,
    loss_db: float = 0.0,
    flip_prob: float = 0.0,
    attack: Attack = "none",
    sample_frac: float = 0.1,
    qber_abort_threshold: float = 0.11,
    ec_efficiency: float = 1.16,
    seed: Optional[int] = 0,
    detector: Optional[DetectorParams] = None,
    attack_config: Optional[AttackConfig] = None,
) -> RunSummary:
    """
    BB84 Monte Carlo simulator with:
      - channel loss: transmittance = 10^(-loss_db/10)
      - detector model: detection efficiency (eta) and background clicks (p_bg)
      - intrinsic bit-flip noise after measurement (flip_prob)
      - optional toy attacks: intercept-resend, PNS, time-shift, blinding

    Returns sifted length, QBER estimate (from a random sample of sifted bits),
    and an engineering-facing asymptotic secret fraction:

      secret_fraction ≈ max(0, 1 - f_ec*h2(Q) - h2(Q))

    This is not a full finite-key proof; it's a clean security-curve instrument.

    Detection Model
    ---------------
    For each pulse:
      - Signal click probability: p_sig = eta * transmittance
      - Background click probability: p_bg (independent of signal)
      - A click occurs if sig_click OR bg_click
      - If only bg_click (no sig_click): Bob's bit is uniform random
      - If sig_click (with or without bg_click): signal dominates (Bob gets signal bit)

    This models the key operational effect: at high loss, background clicks
    dominate and QBER approaches 0.5.

    Parameters
    ----------
    detector : DetectorParams, optional
        Detector model parameters. If None, uses DEFAULT_DETECTOR (eta=0.2, p_bg=1e-4).
    """
    rng = np.random.default_rng(seed)
    det = detector if detector is not None else DEFAULT_DETECTOR
    config = attack_config if attack_config is not None else AttackConfig(attack=attack)
    attack = config.attack

    # Alice bits and bases: 0=Z, 1=X
    a_bits = rng.integers(0, 2, size=n_pulses, dtype=np.int8)
    a_basis = rng.integers(0, 2, size=n_pulses, dtype=np.int8)

    # Channel transmittance and detection model
    trans = 10 ** (-loss_db / 10.0)
    eta_z = det.eta_z if det.eta_z is not None else det.eta
    eta_x = det.eta_x if det.eta_x is not None else det.eta
    use_basis_efficiency = attack == "time_shift" or eta_z != eta_x

    if use_basis_efficiency:
        eta_z_eff, eta_x_eff = apply_time_shift(eta_z, eta_x, config.timeshift_bias)
        b_basis_full = rng.integers(0, 2, size=n_pulses, dtype=np.int8)
        eta_basis = np.where(b_basis_full == 0, eta_z_eff, eta_x_eff)
        p_sig_click = eta_basis * trans
        sig_click = rng.random(n_pulses) < p_sig_click
    else:
        b_basis_full = None
        p_sig_click = det.eta * trans
        sig_click = rng.random(n_pulses) < p_sig_click

    bg_click = rng.random(n_pulses) < det.p_bg
    click = sig_click | bg_click

    # Track which clicks are background-only (will be random bits)
    bg_only = bg_click & (~sig_click)

    state = apply_attack(
        config,
        state=AttackState(a_bits=a_bits, a_basis=a_basis, click=click, bg_only=bg_only),
        rng=rng,
    )
    click = state.click
    bg_only = state.bg_only
    blinding_forced = state.blinding_forced
    pns_multi = state.meta.get("pns_multi_photon_frac")
    idx = np.where(click)[0]

    if idx.size == 0:
        return RunSummary(
            n_sent=n_pulses,
            n_received=0,
            n_sifted=0,
            qber=float("nan"),
            secret_fraction=0.0,
            n_secret_est=0,
            aborted=True,
            meta=_meta(loss_db, flip_prob, attack, sample_frac, qber_abort_threshold,
                       ec_efficiency, seed, det, config, pns_multi),
        )

    # For signal clicks, determine incoming bits (possibly through Eve)
    # For background-only clicks, we'll assign random bits after sifting

    incoming_bits = state.incoming_bits
    incoming_basis = state.incoming_basis

    # Bob chooses bases
    if b_basis_full is None:
        b_basis = rng.integers(0, 2, size=idx.size, dtype=np.int8)
    else:
        b_basis = b_basis_full[idx]

    # Bob measures
    b_bits = incoming_bits.copy()
    mismatch_b = b_basis != incoming_basis
    b_bits[mismatch_b] = rng.integers(0, 2, size=np.sum(mismatch_b), dtype=np.int8)

    # For background-only clicks, Bob's bit is uniform random (independent of Alice)
    bg_only_idx = bg_only[idx]
    b_bits[bg_only_idx] = rng.integers(0, 2, size=np.sum(bg_only_idx), dtype=np.int8)

    if attack == "blinding" and np.any(blinding_forced):
        forced_idx = blinding_forced[idx]
        if np.any(forced_idx):
            if config.blinding_mode == "stealth":
                b_bits[forced_idx] = a_bits[idx][forced_idx]
            else:
                b_bits[forced_idx] = rng.integers(0, 2, size=np.sum(forced_idx), dtype=np.int8)

    # Intrinsic noise
    if flip_prob > 0:
        flips = rng.random(idx.size) < flip_prob
        b_bits[flips] ^= 1

    # Sifting where Bob basis matches Alice basis
    sift = b_basis == a_basis[idx]
    a_sift = a_bits[idx][sift]
    b_sift = b_bits[sift]
    n_sift = int(a_sift.size)

    if n_sift == 0:
        return RunSummary(
            n_sent=n_pulses,
            n_received=int(idx.size),
            n_sifted=0,
            qber=float("nan"),
            secret_fraction=0.0,
            n_secret_est=0,
            aborted=True,
            meta=_meta(loss_db, flip_prob, attack, sample_frac, qber_abort_threshold,
                       ec_efficiency, seed, det, config, pns_multi),
        )

    # Estimate QBER from a random sample
    n_sample = max(1, int(sample_frac * n_sift))
    sidx = rng.choice(n_sift, size=n_sample, replace=False)
    qber = float(np.mean(a_sift[sidx] != b_sift[sidx]))

    aborted = qber > qber_abort_threshold

    # Compute secret fraction only for non-aborted trials
    # Aborted trials have secret_fraction = 0 (no key extractable)
    if aborted:
        secret_fraction = 0.0
        n_secret_est = 0
    else:
        secret_fraction = max(0.0, 1.0 - ec_efficiency * h2(qber) - h2(qber))
        n_raw_key = n_sift - n_sample
        n_secret_est = int(np.floor(n_raw_key * secret_fraction))

    if attack == "pns" and not aborted:
        privacy_factor = max(0.0, 1.0 - (pns_multi if pns_multi is not None else 0.0))
        secret_fraction *= privacy_factor
        n_raw_key = n_sift - n_sample
        n_secret_est = int(np.floor(n_raw_key * secret_fraction))

    return RunSummary(
        n_sent=n_pulses,
        n_received=int(idx.size),
        n_sifted=n_sift,
        qber=qber,
        secret_fraction=secret_fraction,
        n_secret_est=n_secret_est,
        aborted=aborted,
        meta=_meta(loss_db, flip_prob, attack, sample_frac, qber_abort_threshold,
                   ec_efficiency, seed, det, config, pns_multi),
    )

def _meta(
    loss_db: float,
    flip_prob: float,
    attack: Attack,
    sample_frac: float,
    qber_abort_threshold: float,
    ec_efficiency: float,
    seed: Optional[int],
    detector: DetectorParams,
    config: AttackConfig,
    pns_multi: Optional[float],
) -> Dict[str, Any]:
    return {
        "loss_db": float(loss_db),
        "flip_prob": float(flip_prob),
        "attack": str(attack),
        "sample_frac": float(sample_frac),
        "qber_abort_threshold": float(qber_abort_threshold),
        "ec_efficiency": float(ec_efficiency),
        "seed": seed,
        "eta": float(detector.eta),
        "p_bg": float(detector.p_bg),
        "eta_z": float(detector.eta_z if detector.eta_z is not None else detector.eta),
        "eta_x": float(detector.eta_x if detector.eta_x is not None else detector.eta),
        "attack_mu": float(config.mu),
        "timeshift_bias": float(config.timeshift_bias),
        "blinding_mode": str(config.blinding_mode),
        "blinding_prob": float(config.blinding_prob),
        "pns_multi_photon_frac": float(pns_multi) if pns_multi is not None else None,
    }
