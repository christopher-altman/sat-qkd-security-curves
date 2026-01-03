import pytest

from sat_qkd_lab.scoring import bh_fdr


def test_bh_fdr_adjustment():
    p_values = [0.01, 0.04, 0.03, 0.002]
    q_values = bh_fdr(p_values, alpha=0.1)
    expected = [0.02, 0.04, 0.04, 0.008]
    for q_val, exp in zip(q_values, expected):
        assert q_val == pytest.approx(exp)
