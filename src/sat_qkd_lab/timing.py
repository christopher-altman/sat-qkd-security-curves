"""
Timing model utilities for clock offset, drift, and TDC quantization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .timetags import TimeTags


@dataclass(frozen=True)
class TimingModel:
    delta_t: float = 0.0
    drift_ppm: float = 0.0
    tdc_seconds: float = 0.0
    jitter_sigma_s: float = 0.0


def apply_timing_model(
    tags: TimeTags,
    model: TimingModel,
    rng: Optional[np.random.Generator] = None,
    delta_override: Optional[float] = None,
) -> TimeTags:
    """
    Apply clock offset, drift, jitter, and quantization to time tags.
    """
    if tags.times.size == 0:
        return tags

    delta_t = model.delta_t if delta_override is None else delta_override
    drift = 1.0 + model.drift_ppm * 1e-6
    times = tags.times * drift + delta_t

    if model.jitter_sigma_s > 0.0:
        if rng is None:
            rng = np.random.default_rng(0)
        times = times + rng.normal(0.0, model.jitter_sigma_s, size=times.size)

    if model.tdc_seconds > 0.0:
        times = np.round(times / model.tdc_seconds) * model.tdc_seconds

    return TimeTags(
        times=times,
        is_signal=tags.is_signal,
        basis=tags.basis,
        bit=tags.bit,
    )


def _count_coincidences(
    tags_a: TimeTags,
    tags_b: TimeTags,
    tau_seconds: float,
) -> int:
    i = 0
    j = 0
    count = 0
    times_a = tags_a.times
    times_b = tags_b.times
    while i < len(times_a) and j < len(times_b):
        dt = times_a[i] - times_b[j]
        if abs(dt) <= tau_seconds:
            count += 1
            i += 1
            j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1
    return count


def estimate_clock_offset(
    tags_a: TimeTags,
    tags_b: TimeTags,
    model: TimingModel,
    tau_seconds: float,
    search_window_s: float,
    coarse_step_s: float,
    fine_step_s: float,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Estimate clock offset by scanning for maximum coincidences.
    """
    if search_window_s <= 0:
        return model.delta_t

    base_offsets = np.arange(-search_window_s, search_window_s + coarse_step_s, coarse_step_s)
    best_offset = model.delta_t
    best_count = -1
    for offset in base_offsets:
        tags_b_shift = apply_timing_model(tags_b, model, rng=rng, delta_override=offset)
        count = _count_coincidences(tags_a, tags_b_shift, tau_seconds)
        if count > best_count:
            best_count = count
            best_offset = offset

    fine_offsets = np.arange(best_offset - coarse_step_s, best_offset + coarse_step_s + fine_step_s, fine_step_s)
    for offset in fine_offsets:
        tags_b_shift = apply_timing_model(tags_b, model, rng=rng, delta_override=offset)
        count = _count_coincidences(tags_a, tags_b_shift, tau_seconds)
        if count > best_count:
            best_count = count
            best_offset = offset

    return best_offset
