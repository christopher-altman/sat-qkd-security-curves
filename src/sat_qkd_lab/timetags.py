"""
Deterministic time-tag generator for signal and background events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import math
import numpy as np


@dataclass(frozen=True)
class TimeTags:
    times: np.ndarray
    is_signal: np.ndarray
    basis: np.ndarray
    bit: np.ndarray


def _clip_times(times: np.ndarray, duration_s: float) -> np.ndarray:
    return np.clip(times, 0.0, duration_s)


def _sort_tags(tags: TimeTags) -> TimeTags:
    order = np.argsort(tags.times)
    return TimeTags(
        times=tags.times[order],
        is_signal=tags.is_signal[order],
        basis=tags.basis[order],
        bit=tags.bit[order],
    )


def generate_pair_time_tags(
    n_pairs: int,
    duration_s: float,
    sigma_a: float,
    sigma_b: float,
    seed: int,
) -> Tuple[TimeTags, TimeTags]:
    """
    Generate time tags for entangled pairs with independent jitter per detector.
    """
    rng = np.random.default_rng(seed)
    emission = rng.random(n_pairs) * duration_s
    times_a = _clip_times(emission + rng.normal(0.0, sigma_a, n_pairs), duration_s)
    times_b = _clip_times(emission + rng.normal(0.0, sigma_b, n_pairs), duration_s)

    basis_a = rng.integers(0, 2, size=n_pairs, dtype=np.int8)
    basis_b = rng.integers(0, 2, size=n_pairs, dtype=np.int8)
    bits = rng.integers(0, 2, size=n_pairs, dtype=np.int8)

    bit_a = bits.copy()
    bit_b = bits.copy()
    mismatch = basis_a != basis_b
    bit_b[mismatch] = rng.integers(0, 2, size=np.sum(mismatch), dtype=np.int8)

    tags_a = TimeTags(
        times=times_a,
        is_signal=np.ones(n_pairs, dtype=bool),
        basis=basis_a,
        bit=bit_a,
    )
    tags_b = TimeTags(
        times=times_b,
        is_signal=np.ones(n_pairs, dtype=bool),
        basis=basis_b,
        bit=bit_b,
    )
    return _sort_tags(tags_a), _sort_tags(tags_b)


def generate_background_time_tags(
    rate_hz: float,
    duration_s: float,
    sigma: float,
    seed: int,
) -> TimeTags:
    """
    Generate background time tags with random basis/bit assignments.
    """
    rng = np.random.default_rng(seed)
    n_bg = rng.poisson(rate_hz * duration_s)
    times = rng.random(n_bg) * duration_s
    times = _clip_times(times + rng.normal(0.0, sigma, n_bg), duration_s)
    basis = rng.integers(0, 2, size=n_bg, dtype=np.int8)
    bit = rng.integers(0, 2, size=n_bg, dtype=np.int8)
    tags = TimeTags(
        times=times,
        is_signal=np.zeros(n_bg, dtype=bool),
        basis=basis,
        bit=bit,
    )
    return _sort_tags(tags)


def merge_time_tags(signal: TimeTags, background: TimeTags) -> TimeTags:
    """
    Merge signal and background time tags into one sorted stream.
    """
    times = np.concatenate([signal.times, background.times])
    is_signal = np.concatenate([signal.is_signal, background.is_signal])
    basis = np.concatenate([signal.basis, background.basis])
    bit = np.concatenate([signal.bit, background.bit])
    return _sort_tags(TimeTags(times=times, is_signal=is_signal, basis=basis, bit=bit))


def apply_dead_time(tags: TimeTags, dead_time_s: float) -> TimeTags:
    """
    Apply detector dead time by discarding events within the window.
    """
    if dead_time_s <= 0 or tags.times.size == 0:
        return tags
    keep = []
    last_time = -np.inf
    for idx, t in enumerate(tags.times):
        if t - last_time >= dead_time_s:
            keep.append(idx)
            last_time = t
    keep_idx = np.array(keep, dtype=int)
    return TimeTags(
        times=tags.times[keep_idx],
        is_signal=tags.is_signal[keep_idx],
        basis=tags.basis[keep_idx],
        bit=tags.bit[keep_idx],
    )


def add_afterpulsing(
    tags: TimeTags,
    p_afterpulse: float,
    window_s: float,
    decay: float,
    seed: int,
) -> TimeTags:
    """
    Add afterpulse background events following signal detections.
    """
    if p_afterpulse <= 0 or window_s <= 0 or tags.times.size == 0:
        return tags
    rng = np.random.default_rng(seed)
    new_times = []
    new_basis = []
    new_bits = []
    for t, is_sig in zip(tags.times, tags.is_signal):
        if not is_sig:
            continue
        dt = rng.random() * window_s
        if decay > 0.0:
            prob = p_afterpulse * math.exp(-dt / decay)
        else:
            prob = p_afterpulse
        if rng.random() < prob:
            new_times.append(min(t + dt, window_s + t))
            new_basis.append(rng.integers(0, 2))
            new_bits.append(rng.integers(0, 2))
    if not new_times:
        return tags
    after_tags = TimeTags(
        times=np.array(new_times, dtype=float),
        is_signal=np.zeros(len(new_times), dtype=bool),
        basis=np.array(new_basis, dtype=np.int8),
        bit=np.array(new_bits, dtype=np.int8),
    )
    return merge_time_tags(tags, _sort_tags(after_tags))
