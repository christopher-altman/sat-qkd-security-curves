import math

from sat_qkd_lab.plotting import _key_rate_scale_settings


def test_key_rate_scale_log_when_all_positive():
    scale, params = _key_rate_scale_settings([1e-3, 2e-4, 5e-5])

    assert scale == "log"
    assert math.isclose(params["bottom"], 5e-6)


def test_key_rate_scale_symlog_when_zero_present():
    scale, params = _key_rate_scale_settings([0.0, 1e-9, 2e-8])

    assert scale == "symlog"
    assert math.isclose(params["linthresh"], 1e-10)
    assert params["bottom"] == 0
    assert params["linscale"] == 1.0


def test_key_rate_scale_linear_when_all_zero():
    scale, params = _key_rate_scale_settings([0.0, 0.0])

    assert scale == "linear"
    assert params["bottom"] == 0
