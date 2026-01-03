from sat_qkd_lab.clock_sync import generate_beacon_times, apply_clock_model, estimate_offset_drift
from sat_qkd_lab.timetags import generate_pair_time_tags, TimeTags
from sat_qkd_lab.coincidence import match_coincidences
from sat_qkd_lab.timing import TimingModel


def test_estimate_offset_and_drift():
    times_a = generate_beacon_times(duration_s=1.0, rate_hz=1000.0, jitter_sigma_s=0.0, seed=1)
    times_b = apply_clock_model(times_a, offset_s=3e-9, drift_ppm=5.0)
    result = estimate_offset_drift(times_a, times_b)
    assert abs(result.offset_s - 3e-9) < 1e-11
    assert abs(result.drift_ppm - 5.0) < 1e-3


def test_sync_improves_coincidences():
    tags_a, tags_b = generate_pair_time_tags(
        n_pairs=200,
        duration_s=1.0,
        sigma_a=1e-10,
        sigma_b=1e-10,
        seed=2,
    )
    skewed_times_b = apply_clock_model(tags_b.times, offset_s=5e-9, drift_ppm=10.0)
    tags_b = TimeTags(
        times=skewed_times_b,
        is_signal=tags_b.is_signal,
        basis=tags_b.basis,
        bit=tags_b.bit,
    )
    tau = 1e-9
    pre = match_coincidences(tags_a, tags_b, tau_seconds=tau)
    correction = TimingModel(delta_t=-5e-9, drift_ppm=-10.0, tdc_seconds=0.0)
    post = match_coincidences(tags_a, tags_b, tau_seconds=tau, timing_model=correction)
    assert post.coincidences >= pre.coincidences
