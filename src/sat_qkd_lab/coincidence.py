"""
Coincidence matching and CAR computation for time-tag streams.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import math

import numpy as np

from .timetags import TimeTags
from .timing import TimingModel, apply_timing_model, estimate_clock_offset


@dataclass(frozen=True)
class CoincidenceResult:
    coincidences: int
    accidentals: int
    car: float
    matrices: Dict[str, List[List[int]]]
    estimated_offset_s: Optional[float] = None


def match_coincidences(
    tags_a: TimeTags,
    tags_b: TimeTags,
    tau_seconds: float,
    timing_model: Optional[TimingModel] = None,
    estimate_offset: bool = False,
) -> CoincidenceResult:
    """
    Match time tags with a two-pointer merge.

    Counts coincidences when |t_a - t_b| <= tau_seconds.
    Accidentals are matched events where at least one event is background.
    """
    if tau_seconds <= 0:
        raise ValueError("tau_seconds must be positive.")

    estimated_offset = None
    if timing_model is not None:
        if estimate_offset:
            search_window = 5.0 * tau_seconds
            coarse_step = max(tau_seconds / 2.0, 1e-12)
            fine_step = max(tau_seconds / 10.0, 1e-12)
            estimated_offset = estimate_clock_offset(
                tags_a=tags_a,
                tags_b=tags_b,
                model=timing_model,
                tau_seconds=tau_seconds,
                search_window_s=search_window,
                coarse_step_s=coarse_step,
                fine_step_s=fine_step,
                rng=np.random.default_rng(0),
            )
            tags_b = apply_timing_model(tags_b, timing_model, rng=np.random.default_rng(1), delta_override=estimated_offset)
        else:
            tags_b = apply_timing_model(tags_b, timing_model, rng=np.random.default_rng(1))

    i = 0
    j = 0
    coincidences = 0
    accidentals = 0
    matrices = {
        "Z": [[0, 0], [0, 0]],
        "X": [[0, 0], [0, 0]],
    }

    times_a = tags_a.times
    times_b = tags_b.times

    while i < len(times_a) and j < len(times_b):
        dt = times_a[i] - times_b[j]
        if abs(dt) <= tau_seconds:
            is_signal = tags_a.is_signal[i] and tags_b.is_signal[j]
            if is_signal:
                coincidences += 1
            else:
                accidentals += 1

            if tags_a.basis[i] == tags_b.basis[j]:
                basis_key = "Z" if tags_a.basis[i] == 0 else "X"
                a_bit = int(tags_a.bit[i])
                b_bit = int(tags_b.bit[j])
                matrices[basis_key][a_bit][b_bit] += 1

            i += 1
            j += 1
        elif dt < 0:
            i += 1
        else:
            j += 1

    if accidentals > 0:
        car = coincidences / accidentals
    else:
        car = math.inf if coincidences > 0 else 0.0

    return CoincidenceResult(
        coincidences=coincidences,
        accidentals=accidentals,
        car=car,
        matrices=matrices,
        estimated_offset_s=estimated_offset,
    )
