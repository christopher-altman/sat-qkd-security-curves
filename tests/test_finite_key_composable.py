from sat_qkd_lab.finite_key import (
    FiniteKeyParams,
    composable_finite_key_report,
    finite_key_bounds,
)


def test_pe_sample_size_tightens_bound():
    bounds_small = finite_key_bounds(n_sifted=100, n_errors=5, eps_pe=1e-6, m_pe=20)
    bounds_large = finite_key_bounds(n_sifted=100, n_errors=5, eps_pe=1e-6, m_pe=80)
    assert bounds_large["qber_upper"] <= bounds_small["qber_upper"]


def test_higher_eps_sec_reduces_penalty():
    params_tight = FiniteKeyParams(eps_sec=1e-12, eps_cor=1e-15)
    params_loose = FiniteKeyParams(eps_sec=1e-6, eps_cor=1e-15)

    report_tight = composable_finite_key_report(
        n_sent=10000,
        n_sifted=2000,
        n_errors=50,
        params=params_tight,
    )
    report_loose = composable_finite_key_report(
        n_sent=10000,
        n_sifted=2000,
        n_errors=50,
        params=params_loose,
    )

    assert report_loose["privacy_amplification_term_bits"] <= report_tight["privacy_amplification_term_bits"]
