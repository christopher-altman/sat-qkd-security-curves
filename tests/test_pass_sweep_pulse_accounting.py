from types import SimpleNamespace

from sat_qkd_lab.run import _resolve_pass_pulse_accounting


def _args(**overrides: float | int | None) -> SimpleNamespace:
    defaults = {
        "n_sent": None,
        "rep_rate": None,
        "pass_duration": 300.0,
        "time_step": 5.0,
        "pulses": 200_000,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_pass_pulses_total_is_monotone_with_n_sent() -> None:
    n_steps = 10
    low = _resolve_pass_pulse_accounting(_args(n_sent=100), n_steps)
    high = _resolve_pass_pulse_accounting(_args(n_sent=101), n_steps)

    assert low[1] == 100
    assert high[1] == 101
    assert high[0] >= low[0]


def test_pass_rep_rate_matches_explicit_n_sent() -> None:
    n_steps = 12
    from_rate = _resolve_pass_pulse_accounting(
        _args(rep_rate=10.0, pass_duration=30.0),
        n_steps,
    )
    from_total = _resolve_pass_pulse_accounting(_args(n_sent=300), n_steps)

    assert from_rate == from_total


def test_pass_pulses_is_total_not_per_step() -> None:
    n_steps = 10
    pulses_per_step, total_sent = _resolve_pass_pulse_accounting(
        _args(pulses=100),
        n_steps,
    )

    assert total_sent == 100
    assert pulses_per_step == 10
