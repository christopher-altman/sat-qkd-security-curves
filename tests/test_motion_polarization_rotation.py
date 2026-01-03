import numpy as np

from sat_qkd_lab.pass_model import PassModelParams, compute_pass_records


def test_rotation_increases_qber() -> None:
    params = PassModelParams(pass_seconds=2.0, dt_seconds=1.0, rep_rate_hz=1e6)
    baseline_angle = np.zeros(3, dtype=float)
    rotated_angle = np.full(3, 0.3, dtype=float)

    base_records, _ = compute_pass_records(
        params=params,
        polarization_angle_rad=baseline_angle,
    )
    rot_records, _ = compute_pass_records(
        params=params,
        polarization_angle_rad=rotated_angle,
    )

    assert rot_records[0]["qber_mean"] >= base_records[0]["qber_mean"]
    assert rot_records[0]["qber_z"] >= base_records[0]["qber_z"]
