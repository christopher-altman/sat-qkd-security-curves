from sat_qkd_lab.finite_key import FiniteKeyParams
from sat_qkd_lab.sweep import sweep_loss_finite_key


def test_finite_key_rate_not_looser_than_asymptotic() -> None:
    loss_values = [5.0, 10.0, 15.0]
    fk_params = FiniteKeyParams(eps_pe=1e-6, eps_sec=1e-6, eps_cor=1e-9)
    results = sweep_loss_finite_key(
        loss_values,
        flip_prob=0.001,
        attack="none",
        n_pulses=2000,
        seed=1,
        finite_key_params=fk_params,
    )
    eps = 1e-12
    for record in results:
        finite = record["key_rate_per_pulse_finite"]
        asymptotic = record["key_rate_per_pulse_asymptotic"]
        assert finite <= asymptotic + eps
        finite_key = record.get("finite_key")
        assert isinstance(finite_key, dict)
        assert finite_key.get("status") in {"secure", "insecure"}
