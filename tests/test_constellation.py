from sat_qkd_lab.constellation import schedule_passes, simulate_inventory


def test_schedule_deterministic_counts():
    schedule_a = schedule_passes(
        n_sats=2,
        n_stations=2,
        horizon_seconds=1000.0,
        passes_per_sat=2,
        pass_duration_s=100.0,
        seed=7,
    )
    schedule_b = schedule_passes(
        n_sats=2,
        n_stations=2,
        horizon_seconds=1000.0,
        passes_per_sat=2,
        pass_duration_s=100.0,
        seed=7,
    )
    assert len(schedule_a) == 4
    assert [p.t_start_s for p in schedule_a] == [p.t_start_s for p in schedule_b]


def test_inventory_balance():
    schedule = schedule_passes(
        n_sats=1,
        n_stations=1,
        horizon_seconds=1000.0,
        passes_per_sat=2,
        pass_duration_s=100.0,
        seed=1,
    )
    inventory = simulate_inventory(
        schedule=schedule,
        horizon_seconds=1000.0,
        initial_bits=0.0,
        production_bits_per_pass=10.0,
        consumption_bps=0.0,
    )
    assert inventory["produced_bits"][-1] == 20.0
    assert inventory["inventory_bits"][-1] == 20.0

    inventory = simulate_inventory(
        schedule=schedule,
        horizon_seconds=1000.0,
        initial_bits=1000.0,
        production_bits_per_pass=0.0,
        consumption_bps=1.0,
    )
    assert inventory["inventory_bits"][-1] == 0.0
