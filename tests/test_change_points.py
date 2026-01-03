from sat_qkd_lab.change_points import detect_change_points, attribute_incidents


def test_detect_change_points_simple_step():
    time_series = {
        "t_seconds": [0, 1, 2, 3, 4],
        "qber_mean": [0.01, 0.01, 0.01, 0.08, 0.08],
    }
    incidents = detect_change_points(time_series, metrics=("qber_mean",), z_threshold=2.0)
    assert any(incident["index"] == 3 for incident in incidents)


def test_attribute_pointing_dropout():
    time_series = {
        "t_seconds": [0, 1, 2, 3],
        "qber_mean": [0.02, 0.02, 0.05, 0.05],
        "loss_db": [20.0, 20.0, 20.0, 20.0],
        "key_rate_per_pulse": [1e-3, 1e-3, 5e-4, 5e-4],
        "pointing_locked": [1, 1, 0, 0],
    }
    incidents = detect_change_points(time_series, metrics=("qber_mean",), z_threshold=2.0)
    incidents = attribute_incidents(time_series, incidents)
    assert any(incident["attribution"] == "pointing_dropout" for incident in incidents)
