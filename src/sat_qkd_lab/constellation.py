"""
Constellation pass scheduling and key inventory model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
import random


@dataclass(frozen=True)
class PassEvent:
    satellite_id: str
    station_id: str
    t_start_s: float
    t_end_s: float


def schedule_passes(
    n_sats: int,
    n_stations: int,
    horizon_seconds: float,
    passes_per_sat: int,
    pass_duration_s: float,
    seed: int = 0,
) -> List[PassEvent]:
    """
    Generate a deterministic pass schedule across satellites and ground stations.
    """
    rng = random.Random(seed)
    schedule: List[PassEvent] = []
    if n_sats <= 0 or n_stations <= 0 or passes_per_sat <= 0:
        return schedule

    spacing = horizon_seconds / max(1, passes_per_sat)
    station_ids = [f"GS{idx:02d}" for idx in range(n_stations)]

    for sat_idx in range(n_sats):
        sat_id = f"SAT{sat_idx:02d}"
        for pass_idx in range(passes_per_sat):
            base_start = pass_idx * spacing
            jitter = rng.uniform(-0.1, 0.1) * spacing
            t_start = max(0.0, min(horizon_seconds - pass_duration_s, base_start + jitter))
            t_end = min(horizon_seconds, t_start + pass_duration_s)
            station_id = station_ids[(sat_idx + pass_idx) % n_stations]
            schedule.append(PassEvent(satellite_id=sat_id, station_id=station_id,
                                      t_start_s=t_start, t_end_s=t_end))

    schedule.sort(key=lambda p: (p.t_start_s, p.satellite_id))
    return schedule


def simulate_inventory(
    schedule: Sequence[PassEvent],
    horizon_seconds: float,
    initial_bits: float,
    production_bits_per_pass: float,
    consumption_bps: float,
) -> Dict[str, List[float]]:
    """
    Simulate key inventory with production on pass completion and continuous consumption.
    """
    t_points = [0.0]
    inventory = [float(initial_bits)]
    produced = [0.0]
    consumed = [0.0]
    cumulative_prod = 0.0
    cumulative_cons = 0.0

    current_time = 0.0
    current_inventory = float(initial_bits)

    for event in sorted(schedule, key=lambda p: p.t_end_s):
        if event.t_end_s < current_time:
            continue
        delta_t = event.t_end_s - current_time
        consumption = max(0.0, consumption_bps) * delta_t
        cumulative_cons += consumption
        current_inventory = max(0.0, current_inventory - consumption)
        current_time = event.t_end_s

        production = max(0.0, production_bits_per_pass)
        cumulative_prod += production
        current_inventory += production

        t_points.append(float(current_time))
        inventory.append(float(current_inventory))
        produced.append(float(cumulative_prod))
        consumed.append(float(cumulative_cons))

    if current_time < horizon_seconds:
        delta_t = horizon_seconds - current_time
        consumption = max(0.0, consumption_bps) * delta_t
        cumulative_cons += consumption
        current_inventory = max(0.0, current_inventory - consumption)
        current_time = horizon_seconds
        t_points.append(float(current_time))
        inventory.append(float(current_inventory))
        produced.append(float(cumulative_prod))
        consumed.append(float(cumulative_cons))

    return {
        "t_seconds": t_points,
        "inventory_bits": inventory,
        "produced_bits": produced,
        "consumed_bits": consumed,
    }
