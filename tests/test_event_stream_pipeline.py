import numpy as np

from sat_qkd_lab.event_stream import StreamParams, generate_event_stream
from sat_qkd_lab.coincidence import match_coincidences


def test_dead_time_reduces_detections():
    params = StreamParams(
        duration_s=1.0,
        pair_rate_hz=2000.0,
        background_rate_hz=200.0,
        dead_time_s=0.0,
        seed=1,
    )
    tags_a_base, tags_b_base = generate_event_stream(params)
    params_dead = StreamParams(
        duration_s=1.0,
        pair_rate_hz=2000.0,
        background_rate_hz=200.0,
        dead_time_s=5e-4,
        seed=1,
    )
    tags_a_dead, tags_b_dead = generate_event_stream(params_dead)
    assert tags_a_dead.times.size <= tags_a_base.times.size
    assert tags_b_dead.times.size <= tags_b_base.times.size


def test_afterpulsing_increases_background_like_events():
    params = StreamParams(
        duration_s=1.0,
        pair_rate_hz=2000.0,
        background_rate_hz=200.0,
        afterpulse_prob=0.0,
        seed=2,
    )
    tags_a_base, _ = generate_event_stream(params)
    params_after = StreamParams(
        duration_s=1.0,
        pair_rate_hz=2000.0,
        background_rate_hz=200.0,
        afterpulse_prob=0.3,
        afterpulse_window_s=1e-3,
        seed=2,
    )
    tags_a_after, _ = generate_event_stream(params_after)
    bg_base = np.sum(~tags_a_base.is_signal)
    bg_after = np.sum(~tags_a_after.is_signal)
    assert bg_after >= bg_base


def test_gating_improves_car():
    params = StreamParams(
        duration_s=1.0,
        pair_rate_hz=2000.0,
        background_rate_hz=2000.0,
        gate_duty_cycle=1.0,
        seed=3,
    )
    tags_a_full, tags_b_full = generate_event_stream(params)
    full = match_coincidences(tags_a_full, tags_b_full, tau_seconds=5e-4)

    params_gate = StreamParams(
        duration_s=1.0,
        pair_rate_hz=2000.0,
        background_rate_hz=2000.0,
        gate_duty_cycle=0.2,
        seed=3,
    )
    tags_a_gate, tags_b_gate = generate_event_stream(params_gate)
    gated = match_coincidences(tags_a_gate, tags_b_gate, tau_seconds=5e-4)

    assert gated.car >= full.car
