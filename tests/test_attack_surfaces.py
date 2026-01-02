from sat_qkd_lab.attacks import AttackConfig
from sat_qkd_lab.bb84 import simulate_bb84
from sat_qkd_lab.detector import DetectorParams
from sat_qkd_lab.run import build_parser


def test_pns_reduces_secret_fraction_without_qber_change():
    det = DetectorParams(eta=0.4, p_bg=0.0)
    base = simulate_bb84(
        n_pulses=20_000,
        loss_db=0.0,
        flip_prob=0.0,
        attack="none",
        seed=1,
        detector=det,
    )
    pns = simulate_bb84(
        n_pulses=20_000,
        loss_db=0.0,
        flip_prob=0.0,
        attack="pns",
        seed=1,
        detector=det,
        attack_config=AttackConfig(attack="pns", mu=0.7),
    )

    assert pns.qber == base.qber
    assert pns.secret_fraction < base.secret_fraction


def test_blinding_increases_clicks():
    det = DetectorParams(eta=0.15, p_bg=0.0)
    base = simulate_bb84(
        n_pulses=5_000,
        loss_db=5.0,
        flip_prob=0.0,
        attack="none",
        seed=2,
        detector=det,
    )
    blinded = simulate_bb84(
        n_pulses=5_000,
        loss_db=5.0,
        flip_prob=0.0,
        attack="blinding",
        seed=2,
        detector=det,
        attack_config=AttackConfig(attack="blinding", blinding_prob=0.4, blinding_mode="loud"),
    )

    assert blinded.n_received >= base.n_received


def test_sweep_parser_accepts_attack_modes(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "sweep",
        "--attack", "pns",
        "--outdir", str(tmp_path),
    ])

    assert args.attack == "pns"
