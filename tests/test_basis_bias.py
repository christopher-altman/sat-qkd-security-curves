import numpy as np

from sat_qkd_lab.basis_bias import BasisBiasParams, basis_bias_from_elevation, basis_probs_from_bias


def test_bias_monotonic_with_elevation():
    elevation = np.array([0.0, 30.0, 60.0, 90.0], dtype=float)
    params = BasisBiasParams(max_bias=0.2, rotation_deg_at_zenith=10.0)
    bias, rotation = basis_bias_from_elevation(elevation, params)
    assert bias[0] == 0.0
    assert bias[-1] > bias[1]
    assert rotation[-1] == params.rotation_deg_at_zenith


def test_basis_probs_sum_to_one():
    bias = np.array([-0.2, 0.0, 0.3], dtype=float)
    pz, px = basis_probs_from_bias(bias)
    assert np.allclose(pz + px, 1.0)
