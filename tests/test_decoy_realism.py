from sat_qkd_lab.decoy_bb84 import DecoyParams, simulate_decoy_bb84
from sat_qkd_lab.detector import DetectorParams


def test_afterpulsing_increases_gain_at_high_loss() -> None:
    decoy = DecoyParams()
    det_base = DetectorParams(eta=0.2, p_bg=1e-5)
    det_after = DetectorParams(
        eta=0.2,
        p_bg=1e-5,
        p_afterpulse=0.05,
        afterpulse_window=5,
    )

    result_base = simulate_decoy_bb84(
        n_pulses=20_000,
        loss_db=45.0,
        decoy=decoy,
        detector=det_base,
        seed=123,
    )
    result_after = simulate_decoy_bb84(
        n_pulses=20_000,
        loss_db=45.0,
        decoy=decoy,
        detector=det_after,
        seed=123,
    )

    assert result_after["Q_signal"] >= result_base["Q_signal"]


def test_dead_time_reduces_detection_gain() -> None:
    decoy = DecoyParams()
    det_base = DetectorParams(eta=0.8, p_bg=1e-6)
    det_dead = DetectorParams(eta=0.8, p_bg=1e-6, dead_time_pulses=5)

    result_base = simulate_decoy_bb84(
        n_pulses=20_000,
        loss_db=5.0,
        decoy=decoy,
        detector=det_base,
        seed=456,
    )
    result_dead = simulate_decoy_bb84(
        n_pulses=20_000,
        loss_db=5.0,
        decoy=decoy,
        detector=det_dead,
        seed=456,
    )

    assert result_dead["Q_signal"] < result_base["Q_signal"]
