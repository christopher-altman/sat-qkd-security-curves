"""
Deterministic golden checks for asymptotic secret fraction and satellite pass envelope.
"""
from statistics import mean

import pytest

from sat_qkd_lab.detector import DetectorParams
from sat_qkd_lab.pass_model import (
    PassModelParams,
    _asymptotic_key_rate_per_pulse,
    compute_pass_records,
)
from sat_qkd_lab.free_space_link import FreeSpaceLinkParams


def test_golden_theory_secret_fraction_threshold():
    """Secret fraction should fall to zero near the BB84 abort threshold."""
    qber_scan = [0.002, 0.03, 0.06, 0.09, 0.11, 0.14]
    secret_fractions = [
        _asymptotic_key_rate_per_pulse(qber, sifted_fraction=1.0, ec_efficiency=1.16)
        for qber in qber_scan
    ]

    assert all(sf >= 0.0 for sf in secret_fractions)
    assert all(
        secret_fractions[i] >= secret_fractions[i + 1] - 1e-12
        for i in range(len(secret_fractions) - 1)
    )
    assert secret_fractions[-1] == pytest.approx(0.0, abs=1e-7)
    assert secret_fractions[-3] > secret_fractions[-2]


def test_golden_satellite_envelope_viability_monotone():
    """Key rate decreases and QBER increases as equivalent link loss grows."""
    params = PassModelParams(
        max_elevation_deg=60.0,
        min_elevation_deg=10.0,
        pass_seconds=60.0,
        dt_seconds=5.0,
        flip_prob=0.005,
        rep_rate_hz=1e7,
        qber_abort_threshold=0.11,
        background_mode="night",
    )
    detector = DetectorParams(eta=0.3, p_bg=1e-4)
    link_losses = [0.5, 2.0, 5.0, 10.0, 20.0]
    peak_key_rates = []
    mean_qbers = []

    for loss_db in link_losses:
        link = FreeSpaceLinkParams(system_loss_db=loss_db)
        records, summary = compute_pass_records(
            params=params,
            link_params=link,
            detector=detector,
        )
        peak_key_rates.append(summary["peak_key_rate"])
        qbers = [record["qber_mean"] for record in records]
        mean_qbers.append(mean(qbers))

    assert peak_key_rates[0] > 0.0
    assert peak_key_rates[-1] < 1e-8
    assert all(
        peak_key_rates[i] >= peak_key_rates[i + 1] - 1e-12
        for i in range(len(peak_key_rates) - 1)
    )
    assert all(0.0 <= q <= 0.5 for q in mean_qbers)
    assert all(
        mean_qbers[i] <= mean_qbers[i + 1] + 1e-12
        for i in range(len(mean_qbers) - 1)
    )
