"""Tests for CV-QKD (GG02) scaffold implementation."""

from __future__ import annotations

import pytest
import numpy as np

from sat_qkd_lab.cv.gg02 import (
    GG02Params,
    GG02Result,
    compute_snr,
    compute_mutual_information,
    compute_holevo_bound,
    compute_secret_key_rate,
)


def test_gg02_module_imports():
    """Test that CV-QKD module can be imported."""
    from sat_qkd_lab.cv import gg02
    assert hasattr(gg02, "GG02Params")
    assert hasattr(gg02, "GG02Result")
    assert hasattr(gg02, "compute_snr")
    assert hasattr(gg02, "compute_mutual_information")
    assert hasattr(gg02, "compute_holevo_bound")
    assert hasattr(gg02, "compute_secret_key_rate")


def test_gg02_params_validation():
    """Test GG02Params validation."""
    # Valid params
    params = GG02Params(V_A=10.0, T=0.5, xi=0.01, eta=0.6, v_el=0.01, beta=0.95)
    assert params.V_A == 10.0
    assert params.T == 0.5

    # Invalid V_A
    with pytest.raises(ValueError, match="V_A must be positive"):
        GG02Params(V_A=-1.0, T=0.5, xi=0.01, eta=0.6, v_el=0.01, beta=0.95)

    # Invalid T
    with pytest.raises(ValueError, match="T must be in"):
        GG02Params(V_A=10.0, T=1.5, xi=0.01, eta=0.6, v_el=0.01, beta=0.95)

    # Invalid xi
    with pytest.raises(ValueError, match="xi must be non-negative"):
        GG02Params(V_A=10.0, T=0.5, xi=-0.1, eta=0.6, v_el=0.01, beta=0.95)

    # Invalid eta
    with pytest.raises(ValueError, match="eta must be in"):
        GG02Params(V_A=10.0, T=0.5, xi=0.01, eta=1.5, v_el=0.01, beta=0.95)

    # Invalid v_el
    with pytest.raises(ValueError, match="v_el must be non-negative"):
        GG02Params(V_A=10.0, T=0.5, xi=0.01, eta=0.6, v_el=-0.1, beta=0.95)

    # Invalid beta
    with pytest.raises(ValueError, match="beta must be in"):
        GG02Params(V_A=10.0, T=0.5, xi=0.01, eta=0.6, v_el=0.01, beta=1.5)


def test_compute_snr_basic():
    """Test SNR computation with basic parameters."""
    params = GG02Params(V_A=10.0, T=0.5, xi=0.01, eta=0.6, v_el=0.01, beta=0.95)
    snr = compute_snr(params)
    assert snr >= 0.0
    assert np.isfinite(snr)


def test_compute_snr_zero_transmittance():
    """Test SNR with zero transmittance."""
    params = GG02Params(V_A=10.0, T=0.0, xi=0.01, eta=0.6, v_el=0.01, beta=0.95)
    snr = compute_snr(params)
    assert snr == 0.0


def test_compute_snr_high_transmittance():
    """Test SNR with high transmittance."""
    params = GG02Params(V_A=10.0, T=0.9, xi=0.01, eta=0.6, v_el=0.01, beta=0.95)
    snr = compute_snr(params)
    assert snr > 0.0


def test_compute_mutual_information_basic():
    """Test mutual information computation."""
    params = GG02Params(V_A=10.0, T=0.5, xi=0.01, eta=0.6, v_el=0.01, beta=0.95)
    I_AB = compute_mutual_information(params)
    assert I_AB >= 0.0
    assert np.isfinite(I_AB)


def test_compute_mutual_information_zero_transmittance():
    """Test mutual information with zero transmittance."""
    params = GG02Params(V_A=10.0, T=0.0, xi=0.01, eta=0.6, v_el=0.01, beta=0.95)
    I_AB = compute_mutual_information(params)
    assert I_AB == 0.0


def test_compute_holevo_bound_stubbed():
    """Test that Holevo bound is currently stubbed (returns None)."""
    params = GG02Params(V_A=10.0, T=0.5, xi=0.01, eta=0.6, v_el=0.01, beta=0.95)
    chi_BE = compute_holevo_bound(params)
    assert chi_BE is None


def test_compute_secret_key_rate_structure():
    """Test secret key rate computation returns correct structure."""
    params = GG02Params(V_A=10.0, T=0.5, xi=0.01, eta=0.6, v_el=0.01, beta=0.95)
    result = compute_secret_key_rate(params)

    assert isinstance(result, GG02Result)
    assert result.snr >= 0.0
    assert result.I_AB >= 0.0
    assert result.chi_BE is None  # Currently stubbed
    assert result.secret_key_rate is None  # Cannot compute without chi_BE
    assert result.status in ["toy", "stub", "not_implemented"]


def test_cv_sweep_tiny_params():
    """Test that CV sweep can run with tiny parameters (integration-like test)."""
    # This tests the pattern that would be used in the CLI
    loss_db_vals = np.linspace(0.0, 10.0, 5)
    results = []

    for loss_db in loss_db_vals:
        T = 10 ** (-loss_db / 10.0)
        params = GG02Params(V_A=10.0, T=T, xi=0.01, eta=0.6, v_el=0.01, beta=0.95)
        result = compute_secret_key_rate(params)
        results.append(result)

    # Check that all results are valid
    assert len(results) == 5
    for result in results:
        assert result.snr >= 0.0
        assert result.I_AB >= 0.0


def test_snr_decreases_with_loss():
    """Test that SNR decreases as channel loss increases."""
    loss_db_vals = [0.0, 5.0, 10.0, 15.0]
    snr_vals = []

    for loss_db in loss_db_vals:
        T = 10 ** (-loss_db / 10.0)
        params = GG02Params(V_A=10.0, T=T, xi=0.01, eta=0.6, v_el=0.01, beta=0.95)
        snr = compute_snr(params)
        snr_vals.append(snr)

    # SNR should be monotonically decreasing with loss
    for i in range(len(snr_vals) - 1):
        assert snr_vals[i] >= snr_vals[i + 1]


def test_mutual_info_decreases_with_loss():
    """Test that mutual information decreases as channel loss increases."""
    loss_db_vals = [0.0, 5.0, 10.0, 15.0]
    I_AB_vals = []

    for loss_db in loss_db_vals:
        T = 10 ** (-loss_db / 10.0)
        params = GG02Params(V_A=10.0, T=T, xi=0.01, eta=0.6, v_el=0.01, beta=0.95)
        I_AB = compute_mutual_information(params)
        I_AB_vals.append(I_AB)

    # I(A:B) should be monotonically decreasing with loss
    for i in range(len(I_AB_vals) - 1):
        assert I_AB_vals[i] >= I_AB_vals[i + 1]
