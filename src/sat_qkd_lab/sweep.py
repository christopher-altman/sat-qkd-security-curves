from __future__ import annotations
from typing import List, Dict, Any, Sequence
from .bb84 import simulate_bb84, Attack
from .link_budget import SatLinkParams, total_channel_loss_db

def sweep_loss(
    loss_db_values: Sequence[float],
    flip_prob: float = 0.0,
    attack: Attack = "none",
    n_pulses: int = 200_000,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, loss_db in enumerate(loss_db_values):
        s = simulate_bb84(
            n_pulses=n_pulses,
            loss_db=float(loss_db),
            flip_prob=float(flip_prob),
            attack=attack,
            seed=seed + i,
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
) -> List[Dict[str, Any]]:
    p = link_params or SatLinkParams()
    out: List[Dict[str, Any]] = []
    for i, el in enumerate(elevation_deg_values):
        loss_db = total_channel_loss_db(float(el), p)
        s = simulate_bb84(
            n_pulses=n_pulses,
            loss_db=float(loss_db),
            flip_prob=float(flip_prob),
            attack=attack,
            seed=seed + i,
        )
        out.append({
            "elevation_deg": float(el),
            "loss_db": float(loss_db),
            "flip_prob": float(flip_prob),
            "attack": str(attack),
            "qber": s.qber,
            "secret_fraction": s.secret_fraction,
            "n_secret_est": s.n_secret_est,
            "n_sifted": s.n_sifted,
            "aborted": bool(s.aborted),
        })
    return out
