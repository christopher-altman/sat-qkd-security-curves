from sat_qkd_lab.bb84 import simulate_bb84
from sat_qkd_lab.attacks import AttackConfig


def test_leakage_fraction_reduces_secret_fraction_without_large_qber_change():
    base = simulate_bb84(n_pulses=5000, loss_db=5.0, seed=3)
    leakage_cfg = AttackConfig(attack="none", leakage_fraction=0.2)
    leaked = simulate_bb84(n_pulses=5000, loss_db=5.0, seed=3, attack_config=leakage_cfg)
    assert leaked.secret_fraction < base.secret_fraction
    assert abs(leaked.qber - base.qber) <= 0.01
