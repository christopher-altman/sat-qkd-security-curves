import numpy as np

from sat_qkd_lab.timetags import TimeTags
from sat_qkd_lab.timing import TimingModel, apply_timing_model, estimate_clock_offset
from sat_qkd_lab.coincidence import match_coincidences


def test_offset_increases_accidentals_and_reduces_car():
    tags_a = TimeTags(
        times=np.array([0.0, 0.2]),
        is_signal=np.array([True, False]),
        basis=np.array([0, 0]),
        bit=np.array([0, 1]),
    )
    tags_b = TimeTags(
        times=np.array([0.0, 1.0]),
        is_signal=np.array([True, False]),
        basis=np.array([0, 0]),
        bit=np.array([0, 1]),
    )
    base = match_coincidences(tags_a, tags_b, tau_seconds=0.05)
    timing = TimingModel(delta_t=0.2, drift_ppm=0.0, tdc_seconds=0.0)
    shifted = match_coincidences(tags_a, tags_b, tau_seconds=0.05, timing_model=timing)
    assert base.coincidences == 1
    assert base.accidentals == 0
    assert shifted.coincidences == 0
    assert shifted.accidentals == 1
    assert shifted.car < base.car


def test_estimate_offset_recovers():
    tags_a = TimeTags(
        times=np.array([0.1, 0.4, 0.8]),
        is_signal=np.array([True, True, True]),
        basis=np.array([0, 0, 0]),
        bit=np.array([0, 1, 0]),
    )
    tags_b = TimeTags(
        times=np.array([0.2, 0.5, 0.9]),
        is_signal=np.array([True, True, True]),
        basis=np.array([0, 0, 0]),
        bit=np.array([0, 1, 0]),
    )
    timing = TimingModel(delta_t=0.1, drift_ppm=0.0, tdc_seconds=0.0)
    est = estimate_clock_offset(
        tags_a=tags_a,
        tags_b=tags_b,
        model=timing,
        tau_seconds=0.02,
        search_window_s=0.2,
        coarse_step_s=0.02,
        fine_step_s=0.005,
        rng=np.random.default_rng(0),
    )
    assert abs(est + 0.1) <= 0.01


def test_tdc_quantization_changes_matching():
    tags_a = TimeTags(
        times=np.array([0.49e-9]),
        is_signal=np.array([True]),
        basis=np.array([0]),
        bit=np.array([1]),
    )
    tags_b = TimeTags(
        times=np.array([0.51e-9]),
        is_signal=np.array([True]),
        basis=np.array([0]),
        bit=np.array([1]),
    )
    base = match_coincidences(tags_a, tags_b, tau_seconds=0.05e-9)
    timing = TimingModel(delta_t=0.0, drift_ppm=0.0, tdc_seconds=1e-9)
    quant = match_coincidences(tags_a, tags_b, tau_seconds=0.05e-9, timing_model=timing)
    assert base.coincidences == 1
    assert quant.coincidences == 0
