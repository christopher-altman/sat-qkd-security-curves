from sat_qkd_lab.pass_model import PassModelParams, compute_pass_records
from sat_qkd_lab.pointing import PointingParams
from sat_qkd_lab.free_space_link import FreeSpaceLinkParams


def test_acquisition_delays_secure_window():
    pass_params = PassModelParams(max_elevation_deg=60.0, min_elevation_deg=10.0, pass_seconds=20.0, dt_seconds=1.0, rep_rate_hz=1e7)
    link = FreeSpaceLinkParams(atm_loss_db_zenith=0.0, system_loss_db=0.0)
    pointing = PointingParams(acq_seconds=5.0, dropout_prob=0.0, relock_seconds=0.0, pointing_jitter_urad=1.0, seed=1)
    records, summary = compute_pass_records(pass_params, link_params=link, pointing=pointing)
    assert summary["secure_window"]["t_start_seconds"] is None or summary["secure_window"]["t_start_seconds"] >= 5.0


def test_dropouts_reduce_total_secure_time():
    pass_params = PassModelParams(max_elevation_deg=60.0, min_elevation_deg=10.0, pass_seconds=20.0, dt_seconds=1.0, rep_rate_hz=1e7)
    link = FreeSpaceLinkParams(atm_loss_db_zenith=0.0, system_loss_db=0.0)
    base_records, base_summary = compute_pass_records(pass_params, link_params=link)

    pointing = PointingParams(acq_seconds=0.0, dropout_prob=0.5, relock_seconds=2.0, pointing_jitter_urad=1.0, seed=2)
    _, drop_summary = compute_pass_records(pass_params, link_params=link, pointing=pointing)

    assert drop_summary["secure_window_seconds_total"] <= base_summary["secure_window_seconds_total"]
