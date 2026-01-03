import numpy as np

from sat_qkd_lab.timetags import TimeTags, generate_pair_time_tags, generate_background_time_tags
from sat_qkd_lab.coincidence import match_coincidences


def test_timetag_generation_deterministic():
    tags_a, tags_b = generate_pair_time_tags(
        n_pairs=5,
        duration_s=1.0,
        sigma_a=1e-10,
        sigma_b=1e-10,
        seed=7,
    )
    assert len(tags_a.times) == 5
    assert np.all(np.diff(tags_a.times) >= 0)
    assert np.all(tags_a.is_signal)

    bg = generate_background_time_tags(
        rate_hz=10.0,
        duration_s=1.0,
        sigma=1e-10,
        seed=7,
    )
    assert np.all(~bg.is_signal)
    assert np.all(np.diff(bg.times) >= 0)


def test_coincidence_matching_counts():
    tags_a = TimeTags(
        times=np.array([0.1, 0.5, 1.0]),
        is_signal=np.array([True, True, False]),
        basis=np.array([0, 0, 0]),
        bit=np.array([0, 1, 0]),
    )
    tags_b = TimeTags(
        times=np.array([0.12, 0.54, 1.3]),
        is_signal=np.array([True, False, True]),
        basis=np.array([0, 0, 0]),
        bit=np.array([0, 1, 1]),
    )
    result = match_coincidences(tags_a, tags_b, tau_seconds=0.05)
    assert result.coincidences == 1
    assert result.accidentals == 1
    assert result.matrices["Z"] == [[1, 0], [0, 1]]


def test_coincidence_car_infinite_when_no_accidentals():
    tags_a = TimeTags(
        times=np.array([0.1, 0.5]),
        is_signal=np.array([True, True]),
        basis=np.array([1, 1]),
        bit=np.array([0, 1]),
    )
    tags_b = TimeTags(
        times=np.array([0.11, 0.49]),
        is_signal=np.array([True, True]),
        basis=np.array([1, 1]),
        bit=np.array([0, 1]),
    )
    result = match_coincidences(tags_a, tags_b, tau_seconds=0.05)
    assert result.coincidences == 2
    assert result.accidentals == 0
    assert np.isinf(result.car)
