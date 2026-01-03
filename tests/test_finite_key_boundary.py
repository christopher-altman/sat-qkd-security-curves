from sat_qkd_lab.finite_key import finite_key_rate_per_pulse


def test_finite_key_boundary_insecure_small_n():
    result = finite_key_rate_per_pulse(
        n_sent=10_000,
        n_sifted=500,
        n_errors=10,
    )
    finite_key = result["finite_key"]
    assert finite_key["status"] == "insecure"
    assert "NO-SECRET-KEY" in finite_key["reason"]
    assert finite_key["bound"] <= 0.0
    assert result["key_rate_per_pulse"] == 0.0


def test_finite_key_boundary_secure_large_n():
    result = finite_key_rate_per_pulse(
        n_sent=1_000_000,
        n_sifted=250_000,
        n_errors=2_500,
    )
    finite_key = result["finite_key"]
    assert finite_key["status"] == "secure"
    assert finite_key["bound"] > 0.0
    assert result["key_rate_per_pulse"] > 0.0
