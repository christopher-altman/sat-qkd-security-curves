from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Any, Literal
import numpy as np

Attack = Literal["none", "intercept_resend", "pns", "time_shift", "blinding"]
BlindingMode = Literal["loud", "stealth"]


@dataclass(frozen=True)
class AttackConfig:
    attack: Attack = "none"
    mu: float = 0.6
    timeshift_bias: float = 0.0
    blinding_mode: BlindingMode = "loud"
    blinding_prob: float = 0.05
    leakage_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.mu < 0.0:
            raise ValueError(f"mu must be >= 0, got {self.mu}")
        if not 0.0 <= self.timeshift_bias <= 1.0:
            raise ValueError(f"timeshift_bias must be in [0, 1], got {self.timeshift_bias}")
        if self.blinding_mode not in ("loud", "stealth"):
            raise ValueError(f"blinding_mode must be loud or stealth, got {self.blinding_mode}")
        if not 0.0 <= self.blinding_prob <= 1.0:
            raise ValueError(f"blinding_prob must be in [0, 1], got {self.blinding_prob}")
        if not 0.0 <= self.leakage_fraction <= 1.0:
            raise ValueError(f"leakage_fraction must be in [0, 1], got {self.leakage_fraction}")


@dataclass
class AttackState:
    a_bits: np.ndarray
    a_basis: np.ndarray
    click: np.ndarray
    bg_only: np.ndarray


@dataclass
class AttackOutcome:
    incoming_bits: np.ndarray
    incoming_basis: np.ndarray
    click: np.ndarray
    bg_only: np.ndarray
    blinding_forced: np.ndarray
    meta: Dict[str, Any]


def poisson_multi_photon_fraction(mu: float) -> float:
    """Return P(n>=2) for Poisson mean mu."""
    if mu <= 0.0:
        return 0.0
    return 1.0 - (1.0 + mu) * math.exp(-mu)


def apply_time_shift(eta_z: float, eta_x: float, bias: float) -> tuple[float, float]:
    """Bias efficiencies toward the larger basis to model time-shift."""
    if bias <= 0.0 or eta_z == eta_x:
        return eta_z, eta_x
    if eta_z > eta_x:
        eta_z_eff = min(1.0, eta_z * (1.0 + bias))
        eta_x_eff = max(0.0, eta_x * (1.0 - bias))
    else:
        eta_x_eff = min(1.0, eta_x * (1.0 + bias))
        eta_z_eff = max(0.0, eta_z * (1.0 - bias))
    return eta_z_eff, eta_x_eff


def apply_attack(
    config: AttackConfig,
    state: AttackState,
    rng: np.random.Generator,
) -> AttackOutcome:
    click = state.click.copy()
    bg_only = state.bg_only.copy()
    blinding_forced = np.zeros_like(click, dtype=bool)
    meta: Dict[str, Any] = {}

    if config.attack == "blinding" and config.blinding_prob > 0.0:
        forced = rng.random(click.size) < config.blinding_prob
        blinding_forced = forced
        newly_forced = forced & (~click)
        click = click | forced
        bg_only = bg_only | newly_forced
        meta["blinding_forced"] = int(np.sum(forced))

    idx = np.where(click)[0]
    incoming_bits = state.a_bits[idx].copy()
    incoming_basis = state.a_basis[idx].copy()

    if config.attack == "intercept_resend":
        e_basis = rng.integers(0, 2, size=idx.size, dtype=np.int8)
        e_bits = incoming_bits.copy()
        mismatch = e_basis != incoming_basis
        if np.any(mismatch):
            e_bits[mismatch] = rng.integers(0, 2, size=np.sum(mismatch), dtype=np.int8)
        incoming_bits = e_bits
        incoming_basis = e_basis

    if config.attack == "pns":
        meta["pns_multi_photon_frac"] = poisson_multi_photon_fraction(config.mu)
    if config.leakage_fraction > 0.0:
        meta["leakage_fraction"] = float(config.leakage_fraction)

    return AttackOutcome(
        incoming_bits=incoming_bits,
        incoming_basis=incoming_basis,
        click=click,
        bg_only=bg_only,
        blinding_forced=blinding_forced,
        meta=meta,
    )
