from __future__ import annotations
import numpy as np
from typing import Literal, Optional, Dict, Any
from .helpers import h2, RunSummary

Attack = Literal["none", "intercept_resend"]

def simulate_bb84(
    n_pulses: int = 200_000,
    loss_db: float = 0.0,
    flip_prob: float = 0.0,
    attack: Attack = "none",
    sample_frac: float = 0.1,
    qber_abort_threshold: float = 0.11,
    ec_efficiency: float = 1.16,
    seed: Optional[int] = 0,
) -> RunSummary:
    """
    BB84 Monte Carlo simulator with:
      - channel loss: transmittance = 10^(-loss_db/10)
      - intrinsic bit-flip noise after measurement (flip_prob)
      - optional intercept-resend attack (Eve measures in random basis and resends)

    Returns sifted length, QBER estimate (from a random sample of sifted bits),
    and an engineering-facing asymptotic secret fraction:

      secret_fraction ≈ max(0, 1 - f_ec*h2(Q) - h2(Q))

    This is not a full finite-key proof; it's a clean security-curve instrument.
    """
    rng = np.random.default_rng(seed)

    # Alice bits and bases: 0=Z, 1=X
    a_bits = rng.integers(0, 2, size=n_pulses, dtype=np.int8)
    a_basis = rng.integers(0, 2, size=n_pulses, dtype=np.int8)

    # Loss: Bob gets a detection event with probability trans
    trans = 10 ** (-loss_db / 10.0)
    received = rng.random(n_pulses) < trans
    idx = np.where(received)[0]

    if idx.size == 0:
        return RunSummary(
            n_sent=n_pulses,
            n_received=0,
            n_sifted=0,
            qber=float("nan"),
            secret_fraction=0.0,
            n_secret_est=0,
            aborted=True,
            meta=_meta(loss_db, flip_prob, attack, sample_frac, qber_abort_threshold, ec_efficiency, seed),
        )

    # Eve intercept-resend
    if attack == "intercept_resend":
        e_basis = rng.integers(0, 2, size=idx.size, dtype=np.int8)
        e_bits = a_bits[idx].copy()
        mismatch_e = e_basis != a_basis[idx]
        e_bits[mismatch_e] = rng.integers(0, 2, size=np.sum(mismatch_e), dtype=np.int8)
        incoming_bits = e_bits
        incoming_basis = e_basis
    else:
        incoming_bits = a_bits[idx]
        incoming_basis = a_basis[idx]

    # Bob chooses bases
    b_basis = rng.integers(0, 2, size=idx.size, dtype=np.int8)

    # Bob measures
    b_bits = incoming_bits.copy()
    mismatch_b = b_basis != incoming_basis
    b_bits[mismatch_b] = rng.integers(0, 2, size=np.sum(mismatch_b), dtype=np.int8)

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
            meta=_meta(loss_db, flip_prob, attack, sample_frac, qber_abort_threshold, ec_efficiency, seed),
        )

    # Estimate QBER from a random sample
    n_sample = max(1, int(sample_frac * n_sift))
    sidx = rng.choice(n_sift, size=n_sample, replace=False)
    qber = float(np.mean(a_sift[sidx] != b_sift[sidx]))

    aborted = qber > qber_abort_threshold

    secret_fraction = max(0.0, 1.0 - ec_efficiency * h2(qber) - h2(qber))

    n_raw_key = n_sift - n_sample
    n_secret_est = int(np.floor(n_raw_key * secret_fraction)) if (not aborted) else 0

    return RunSummary(
        n_sent=n_pulses,
        n_received=int(idx.size),
        n_sifted=n_sift,
        qber=qber,
        secret_fraction=secret_fraction,
        n_secret_est=n_secret_est,
        aborted=aborted,
        meta=_meta(loss_db, flip_prob, attack, sample_frac, qber_abort_threshold, ec_efficiency, seed),
    )

def _meta(loss_db: float, flip_prob: float, attack: Attack, sample_frac: float, qber_abort_threshold: float, ec_efficiency: float, seed: Optional[int]) -> Dict[str, Any]:
    return {
        "loss_db": float(loss_db),
        "flip_prob": float(flip_prob),
        "attack": str(attack),
        "sample_frac": float(sample_frac),
        "qber_abort_threshold": float(qber_abort_threshold),
        "ec_efficiency": float(ec_efficiency),
        "seed": seed,
    }
