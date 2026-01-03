from sat_qkd_lab.bb84 import simulate_bb84
from sat_qkd_lab.detector import DetectorParams


def test_dead_time_reduces_detection_rate():
    det_base = DetectorParams(eta=0.9, p_bg=0.0, dead_time_pulses=0)
    det_dead = DetectorParams(eta=0.9, p_bg=0.0, dead_time_pulses=5)

    base = simulate_bb84(n_pulses=2000, loss_db=0.0, seed=1, detector=det_base)
    dead = simulate_bb84(n_pulses=2000, loss_db=0.0, seed=1, detector=det_dead)

    assert dead.n_received <= base.n_received


def test_afterpulsing_increases_qber():
    det_base = DetectorParams(eta=0.4, p_bg=1e-5, p_afterpulse=0.0, afterpulse_window=0)
    det_after = DetectorParams(
        eta=0.4,
        p_bg=1e-5,
        p_afterpulse=0.2,
        afterpulse_window=3,
        afterpulse_decay=0.0,
    )

    base = simulate_bb84(n_pulses=5000, loss_db=5.0, seed=2, detector=det_base)
    after = simulate_bb84(n_pulses=5000, loss_db=5.0, seed=2, detector=det_after)

    assert after.qber >= base.qber
