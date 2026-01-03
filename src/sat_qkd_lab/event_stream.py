"""
Event-stream generator for detector time tags with gating and detector effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np

from .timetags import TimeTags, merge_time_tags


@dataclass(frozen=True)
class StreamParams:
    duration_s: float
    pair_rate_hz: float
    background_rate_hz: float
    gate_duty_cycle: float = 1.0
    dead_time_s: float = 0.0
    afterpulse_prob: float = 0.0
    afterpulse_window_s: float = 0.0
    afterpulse_decay_s: float = 0.0
    eta_z: float = 1.0
    eta_x: float = 1.0
    misalignment_prob: float = 0.0
    jitter_sigma_s: float = 0.0
    seed: int = 0


def _apply_gate_mask(times: np.ndarray, duty_cycle: float, rng: np.random.Generator) -> np.ndarray:
    if duty_cycle >= 1.0:
        return np.ones(times.size, dtype=bool)
    duty_cycle = max(0.0, min(1.0, duty_cycle))
    return rng.random(times.size) < duty_cycle


def _apply_dead_time_tags(tags: TimeTags, dead_time_s: float) -> TimeTags:
    if dead_time_s <= 0 or tags.times.size == 0:
        return tags
    order = np.argsort(tags.times)
    times = tags.times[order]
    keep_idx = []
    last = -np.inf
    for idx, t in enumerate(times):
        if t - last >= dead_time_s:
            keep_idx.append(idx)
            last = t
    keep_idx = np.array(keep_idx, dtype=int)
    return TimeTags(
        times=times[keep_idx],
        is_signal=tags.is_signal[order][keep_idx],
        basis=tags.basis[order][keep_idx],
        bit=tags.bit[order][keep_idx],
    )


def _afterpulse_tags(
    times: np.ndarray,
    params: StreamParams,
    rng: np.random.Generator,
) -> TimeTags:
    if params.afterpulse_prob <= 0 or params.afterpulse_window_s <= 0 or times.size == 0:
        return TimeTags(
            times=np.array([], dtype=float),
            is_signal=np.array([], dtype=bool),
            basis=np.array([], dtype=np.int8),
            bit=np.array([], dtype=np.int8),
        )
    new_times = []
    for t in times:
        dt = rng.random() * params.afterpulse_window_s
        if params.afterpulse_decay_s > 0.0:
            prob = params.afterpulse_prob * math.exp(-dt / params.afterpulse_decay_s)
        else:
            prob = params.afterpulse_prob
        if rng.random() < prob:
            new_times.append(t + dt)
    if not new_times:
        return TimeTags(
            times=np.array([], dtype=float),
            is_signal=np.array([], dtype=bool),
            basis=np.array([], dtype=np.int8),
            bit=np.array([], dtype=np.int8),
        )
    n = len(new_times)
    return TimeTags(
        times=np.array(new_times, dtype=float),
        is_signal=np.zeros(n, dtype=bool),
        basis=rng.integers(0, 2, size=n, dtype=np.int8),
        bit=rng.integers(0, 2, size=n, dtype=np.int8),
    )


def generate_event_stream(params: StreamParams) -> Tuple[TimeTags, TimeTags]:
    """
    Generate per-detector time tags with gating, dead time, and afterpulsing.
    """
    rng = np.random.default_rng(params.seed)
    n_pairs = rng.poisson(params.pair_rate_hz * params.duration_s)
    emission = rng.random(n_pairs) * params.duration_s
    jitter = rng.normal(0.0, params.jitter_sigma_s, size=n_pairs) if params.jitter_sigma_s > 0.0 else 0.0

    basis_a = rng.integers(0, 2, size=n_pairs, dtype=np.int8)
    basis_b = rng.integers(0, 2, size=n_pairs, dtype=np.int8)
    bits = rng.integers(0, 2, size=n_pairs, dtype=np.int8)

    bit_a = bits.copy()
    bit_b = bits.copy()
    mismatch_basis = basis_a != basis_b
    if np.any(mismatch_basis):
        bit_b[mismatch_basis] = rng.integers(0, 2, size=np.sum(mismatch_basis), dtype=np.int8)

    if params.misalignment_prob > 0.0:
        flip_a = rng.random(n_pairs) < params.misalignment_prob
        flip_b = rng.random(n_pairs) < params.misalignment_prob
        bit_a[flip_a] ^= 1
        bit_b[flip_b] ^= 1

    eta_a = np.where(basis_a == 0, params.eta_z, params.eta_x)
    eta_b = np.where(basis_b == 0, params.eta_z, params.eta_x)
    det_a = rng.random(n_pairs) < eta_a
    det_b = rng.random(n_pairs) < eta_b

    times_a = emission[det_a] + jitter if np.ndim(jitter) else emission[det_a]
    times_b = emission[det_b] + jitter if np.ndim(jitter) else emission[det_b]
    times_a = np.clip(times_a, 0.0, params.duration_s)
    times_b = np.clip(times_b, 0.0, params.duration_s)

    mask_a = _apply_gate_mask(times_a, params.gate_duty_cycle, rng)
    mask_b = _apply_gate_mask(times_b, params.gate_duty_cycle, rng)

    times_a = times_a[mask_a]
    times_b = times_b[mask_b]
    basis_a_det = basis_a[det_a][mask_a]
    basis_b_det = basis_b[det_b][mask_b]
    bit_a_det = bit_a[det_a][mask_a]
    bit_b_det = bit_b[det_b][mask_b]

    tags_a = TimeTags(
        times=times_a,
        is_signal=np.ones(times_a.size, dtype=bool),
        basis=basis_a_det,
        bit=bit_a_det,
    )
    tags_b = TimeTags(
        times=times_b,
        is_signal=np.ones(times_b.size, dtype=bool),
        basis=basis_b_det,
        bit=bit_b_det,
    )

    n_bg_a = rng.poisson(params.background_rate_hz * params.duration_s * params.gate_duty_cycle)
    n_bg_b = rng.poisson(params.background_rate_hz * params.duration_s * params.gate_duty_cycle)
    bg_times_a = rng.random(n_bg_a) * params.duration_s
    bg_times_b = rng.random(n_bg_b) * params.duration_s
    bg_a = TimeTags(
        times=bg_times_a,
        is_signal=np.zeros(bg_times_a.size, dtype=bool),
        basis=rng.integers(0, 2, size=bg_times_a.size, dtype=np.int8),
        bit=rng.integers(0, 2, size=bg_times_a.size, dtype=np.int8),
    )
    bg_b = TimeTags(
        times=bg_times_b,
        is_signal=np.zeros(bg_times_b.size, dtype=bool),
        basis=rng.integers(0, 2, size=bg_times_b.size, dtype=np.int8),
        bit=rng.integers(0, 2, size=bg_times_b.size, dtype=np.int8),
    )

    after_a = _afterpulse_tags(tags_a.times, params, rng)
    after_b = _afterpulse_tags(tags_b.times, params, rng)

    tags_a = _apply_dead_time_tags(merge_time_tags(tags_a, merge_time_tags(bg_a, after_a)), params.dead_time_s)
    tags_b = _apply_dead_time_tags(merge_time_tags(tags_b, merge_time_tags(bg_b, after_b)), params.dead_time_s)

    return tags_a, tags_b
