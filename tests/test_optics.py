import numpy as np

from sat_qkd_lab.optics import OpticalParams, background_rate_hz, dark_count_rate_hz
from sat_qkd_lab.pass_model import PassModelParams, compute_pass_records
from sat_qkd_lab.free_space_link import FreeSpaceLinkParams
from sat_qkd_lab.detector import DetectorParams


def test_background_rate_increases_with_bandwidth():
    base = 100.0
    params_low = OpticalParams(filter_bandwidth_nm=0.5, detector_temp_c=20.0, mode="night")
    params_high = OpticalParams(filter_bandwidth_nm=2.0, detector_temp_c=20.0, mode="night")
    assert background_rate_hz(params_high, base) > background_rate_hz(params_low, base)


def test_dark_counts_increase_with_temperature():
    base = 100.0
    params_cold = OpticalParams(filter_bandwidth_nm=1.0, detector_temp_c=0.0, mode="night")
    params_hot = OpticalParams(filter_bandwidth_nm=1.0, detector_temp_c=40.0, mode="night")
    assert dark_count_rate_hz(params_hot, base) > dark_count_rate_hz(params_cold, base)


def test_bandwidth_worsens_qber():
    pass_params = PassModelParams(pass_seconds=10.0, dt_seconds=1.0, rep_rate_hz=1e7)
    link = FreeSpaceLinkParams(atm_loss_db_zenith=0.0, system_loss_db=0.0)
    det_low = DetectorParams(eta=0.6, p_bg=1e-5)
    det_high = DetectorParams(eta=0.6, p_bg=1e-3)
    records_low, _ = compute_pass_records(pass_params, link_params=link, detector=det_low)
    records_high, _ = compute_pass_records(pass_params, link_params=link, detector=det_high)
    q_low = np.mean([r["qber_mean"] for r in records_low])
    q_high = np.mean([r["qber_mean"] for r in records_high])
    assert q_high >= q_low
