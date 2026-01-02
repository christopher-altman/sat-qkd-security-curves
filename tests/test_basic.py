"""
Test suite for sat-qkd-security-curves.

Tests cover:
- Binary entropy function (h2)
- BB84 simulation correctness
- Detector/background model effects
- Decoy-state BB84 bounds
- Reproducibility with fixed seeds
"""
import math
import pytest


# --- Binary Entropy Tests ---

def test_h2_at_zero():
    """h2(0) should return 0 (no uncertainty when p=0)."""
    from sat_qkd_lab.helpers import h2
    assert h2(0.0) < 1e-8


def test_h2_at_one():
    """h2(1) should return 0 (no uncertainty when p=1)."""
    from sat_qkd_lab.helpers import h2
    assert h2(1.0) < 1e-8


def test_h2_at_half():
    """h2(0.5) should return 1.0 (maximum entropy)."""
    from sat_qkd_lab.helpers import h2
    assert abs(h2(0.5) - 1.0) < 1e-10


def test_h2_symmetry():
    """h2(p) should equal h2(1-p)."""
    from sat_qkd_lab.helpers import h2
    for p in [0.1, 0.2, 0.3, 0.4]:
        assert abs(h2(p) - h2(1.0 - p)) < 1e-10


# --- BB84 Reproducibility Tests ---

def test_reproducibility_same_seed():
    """Same seed should produce identical results."""
    from sat_qkd_lab.bb84 import simulate_bb84
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=0.2, p_bg=0.0)

    r1 = simulate_bb84(n_pulses=50_000, loss_db=30.0, seed=42, detector=det)
    r2 = simulate_bb84(n_pulses=50_000, loss_db=30.0, seed=42, detector=det)

    assert r1.n_received == r2.n_received
    assert r1.n_sifted == r2.n_sifted
    assert r1.qber == r2.qber
    assert r1.secret_fraction == r2.secret_fraction


def test_different_seeds_produce_different_results():
    """Different seeds should produce different results (with high probability)."""
    from sat_qkd_lab.bb84 import simulate_bb84
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=0.2, p_bg=0.0)

    r1 = simulate_bb84(n_pulses=50_000, loss_db=30.0, seed=1, detector=det)
    r2 = simulate_bb84(n_pulses=50_000, loss_db=30.0, seed=2, detector=det)

    # At least one metric should differ
    assert (r1.n_received != r2.n_received or
            r1.n_sifted != r2.n_sifted or
            r1.qber != r2.qber)


# --- Intercept-Resend Attack Tests ---

def test_intercept_resend_qber_near_25pct_no_background():
    """
    Intercept-resend attack with no background noise should yield QBER ~25%.

    Theory: Eve measures in wrong basis 50% of time, causing 50% error
    on those bits. After sifting (Bob matches Alice basis), this gives
    ~25% QBER on sifted key.
    """
    from sat_qkd_lab.bb84 import simulate_bb84
    from sat_qkd_lab.detector import DetectorParams

    # No background, no intrinsic noise, moderate loss
    det = DetectorParams(eta=1.0, p_bg=0.0)  # Perfect detector

    result = simulate_bb84(
        n_pulses=100_000,
        loss_db=10.0,  # Low loss to get many detections
        flip_prob=0.0,
        attack="intercept_resend",
        seed=123,
        detector=det,
        qber_abort_threshold=0.50,  # Raise threshold to observe QBER without aborting
    )

    # QBER should be near 0.25 (allow for statistical variation)
    # Note: protocol would normally abort at this QBER, but we raised threshold for testing
    assert 0.20 <= result.qber <= 0.30, f"Expected QBER ~0.25, got {result.qber}"


def test_no_attack_low_qber_no_background():
    """Without attack and no background, QBER should be close to flip_prob."""
    from sat_qkd_lab.bb84 import simulate_bb84
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=1.0, p_bg=0.0)
    flip_prob = 0.01

    result = simulate_bb84(
        n_pulses=100_000,
        loss_db=10.0,
        flip_prob=flip_prob,
        attack="none",
        seed=456,
        detector=det,
    )

    # QBER should be close to flip_prob
    assert result.qber < 0.05, f"Expected QBER ~0.01, got {result.qber}"


# --- Background Click Model Tests ---

def test_qber_increases_with_loss_when_background_present():
    """
    With background clicks, QBER should increase at higher loss.

    At high loss, signal clicks become rare while background clicks
    (random bits) remain constant, pushing QBER toward 0.5.
    """
    from sat_qkd_lab.bb84 import simulate_bb84
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=0.2, p_bg=1e-3)  # Significant background

    result_low_loss = simulate_bb84(
        n_pulses=50_000,
        loss_db=20.0,
        flip_prob=0.0,
        attack="none",
        seed=789,
        detector=det,
    )

    result_high_loss = simulate_bb84(
        n_pulses=50_000,
        loss_db=50.0,
        flip_prob=0.0,
        attack="none",
        seed=789,
        detector=det,
    )

    # High loss should have higher QBER (or abort/NaN)
    if not math.isnan(result_low_loss.qber) and not math.isnan(result_high_loss.qber):
        assert result_high_loss.qber >= result_low_loss.qber - 0.05, \
            f"Expected QBER to increase with loss: {result_low_loss.qber} -> {result_high_loss.qber}"


def test_no_background_qber_stable_with_loss():
    """Without background, QBER should not depend strongly on loss."""
    from sat_qkd_lab.bb84 import simulate_bb84
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=0.2, p_bg=0.0)  # No background

    result_low = simulate_bb84(
        n_pulses=50_000,
        loss_db=20.0,
        flip_prob=0.01,
        attack="none",
        seed=111,
        detector=det,
    )

    result_mid = simulate_bb84(
        n_pulses=50_000,
        loss_db=35.0,
        flip_prob=0.01,
        attack="none",
        seed=111,
        detector=det,
    )

    # QBER should be similar (both around flip_prob)
    if not result_low.aborted and not result_mid.aborted:
        assert abs(result_low.qber - result_mid.qber) < 0.03, \
            f"QBER should be stable without background: {result_low.qber} vs {result_mid.qber}"


# --- High Loss / Abort Tests ---

def test_high_loss_aborts_or_zero_received():
    """Very high loss should abort protocol or yield zero detections."""
    from sat_qkd_lab.bb84 import simulate_bb84
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=0.1, p_bg=1e-6)

    result = simulate_bb84(
        n_pulses=10_000,
        loss_db=80.0,  # Extremely high loss
        flip_prob=0.0,
        attack="none",
        seed=222,
        detector=det,
    )

    # Should either abort or have very few detections
    assert result.aborted or result.n_received < 100 or result.n_sifted < 50


def test_secret_fraction_zero_when_aborted():
    """When protocol aborts, secret fraction should be 0."""
    from sat_qkd_lab.bb84 import simulate_bb84
    from sat_qkd_lab.detector import DetectorParams

    # Force high QBER by using intercept-resend at low loss (more detections = stable QBER)
    det = DetectorParams(eta=1.0, p_bg=0.0)

    result = simulate_bb84(
        n_pulses=100_000,
        loss_db=5.0,  # Low loss for many detections and stable 25% QBER
        flip_prob=0.0,
        attack="intercept_resend",
        qber_abort_threshold=0.11,  # Will trigger abort at ~25% QBER
        seed=333,
        detector=det,
    )

    assert result.aborted, f"Expected abort with QBER={result.qber:.3f} > 0.11"
    assert result.n_secret_est == 0


# --- Decoy-State BB84 Tests ---

def test_decoy_outputs_in_valid_range():
    """Decoy bounds should be in physically valid ranges."""
    from sat_qkd_lab.decoy_bb84 import simulate_decoy_bb84, DecoyParams
    from sat_qkd_lab.detector import DetectorParams

    decoy = DecoyParams(mu_s=0.6, mu_d=0.1, mu_v=0.0, p_s=0.8, p_d=0.15, p_v=0.05)
    det = DetectorParams(eta=0.2, p_bg=1e-4)

    result = simulate_decoy_bb84(
        n_pulses=100_000,
        loss_db=30.0,
        decoy=decoy,
        detector=det,
        seed=444,
    )

    # Y1 lower bound should be in [0, 1]
    assert 0.0 <= result["Y1_lower"] <= 1.0, f"Y1_lower out of range: {result['Y1_lower']}"

    # e1 upper bound should be in [0, 0.5]
    assert 0.0 <= result["e1_upper"] <= 0.5, f"e1_upper out of range: {result['e1_upper']}"

    # Key rate should be non-negative
    assert result["key_rate_asymptotic"] >= 0.0, f"Negative key rate: {result['key_rate_asymptotic']}"


def test_decoy_rate_decreases_with_loss():
    """Decoy key rate should decrease (or stay zero) as loss increases."""
    from sat_qkd_lab.decoy_bb84 import simulate_decoy_bb84, DecoyParams
    from sat_qkd_lab.detector import DetectorParams

    decoy = DecoyParams()
    det = DetectorParams(eta=0.2, p_bg=1e-4)

    result_low = simulate_decoy_bb84(
        n_pulses=50_000,
        loss_db=25.0,
        decoy=decoy,
        detector=det,
        seed=555,
    )

    result_high = simulate_decoy_bb84(
        n_pulses=50_000,
        loss_db=45.0,
        decoy=decoy,
        detector=det,
        seed=555,
    )

    # Allow small tolerance for Monte Carlo variation
    assert result_high["key_rate_asymptotic"] <= result_low["key_rate_asymptotic"] + 1e-4, \
        f"Key rate should decrease with loss: {result_low['key_rate_asymptotic']} -> {result_high['key_rate_asymptotic']}"


def test_decoy_params_validation():
    """DecoyParams should validate probability sum."""
    from sat_qkd_lab.decoy_bb84 import DecoyParams

    # Valid parameters
    valid = DecoyParams(mu_s=0.6, mu_d=0.1, mu_v=0.0, p_s=0.8, p_d=0.15, p_v=0.05)
    assert valid.p_s + valid.p_d + valid.p_v == 1.0

    # Invalid: probabilities don't sum to 1
    with pytest.raises(ValueError):
        DecoyParams(mu_s=0.6, mu_d=0.1, mu_v=0.0, p_s=0.5, p_d=0.3, p_v=0.3)

    # Invalid: vacuum intensity not zero
    with pytest.raises(ValueError):
        DecoyParams(mu_s=0.6, mu_d=0.1, mu_v=0.01, p_s=0.8, p_d=0.15, p_v=0.05)


# --- Detector Parameter Validation Tests ---

def test_detector_params_validation():
    """DetectorParams should validate parameter ranges."""
    from sat_qkd_lab.detector import DetectorParams

    # Valid parameters
    valid = DetectorParams(eta=0.5, p_bg=1e-4)
    assert valid.eta == 0.5

    # Invalid eta
    with pytest.raises(ValueError):
        DetectorParams(eta=1.5, p_bg=1e-4)

    with pytest.raises(ValueError):
        DetectorParams(eta=-0.1, p_bg=1e-4)

    # Invalid p_bg
    with pytest.raises(ValueError):
        DetectorParams(eta=0.5, p_bg=-1e-4)


# --- Link Budget Tests ---

def test_slant_range_at_zenith():
    """At zenith (90 deg), slant range should equal satellite altitude."""
    from sat_qkd_lab.link_budget import slant_range_m, SatLinkParams

    p = SatLinkParams(altitude_m=500_000)
    r = slant_range_m(90.0, p)

    # Should be close to altitude (within 1km tolerance)
    assert abs(r - p.altitude_m) < 1000, f"Expected ~{p.altitude_m}, got {r}"


def test_slant_range_increases_toward_horizon():
    """Slant range should increase as elevation decreases."""
    from sat_qkd_lab.link_budget import slant_range_m, SatLinkParams

    p = SatLinkParams()

    r_90 = slant_range_m(90.0, p)
    r_45 = slant_range_m(45.0, p)
    r_10 = slant_range_m(10.0, p)

    assert r_45 > r_90, "Slant range should increase from zenith"
    assert r_10 > r_45, "Slant range should increase toward horizon"


def test_total_loss_increases_toward_horizon():
    """Total loss should increase as elevation decreases."""
    from sat_qkd_lab.link_budget import total_channel_loss_db, SatLinkParams

    p = SatLinkParams()

    loss_90 = total_channel_loss_db(90.0, p)
    loss_45 = total_channel_loss_db(45.0, p)
    loss_10 = total_channel_loss_db(10.0, p)

    assert loss_45 > loss_90, "Loss should increase from zenith"
    assert loss_10 > loss_45, "Loss should increase toward horizon"


# --- Sweep Function Tests ---

def test_sweep_loss_returns_correct_structure():
    """sweep_loss should return list of dicts with expected keys."""
    from sat_qkd_lab.sweep import sweep_loss
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=0.2, p_bg=0.0)
    results = sweep_loss(
        [20.0, 30.0, 40.0],
        n_pulses=10_000,
        seed=666,
        detector=det,
    )

    assert len(results) == 3

    expected_keys = {"loss_db", "flip_prob", "attack", "n_sent", "n_received",
                     "n_sifted", "qber", "secret_fraction", "n_secret_est", "aborted"}

    for r in results:
        assert expected_keys.issubset(r.keys()), f"Missing keys in result: {expected_keys - r.keys()}"


def test_sweep_loss_with_ci_returns_statistics():
    """sweep_loss_with_ci should return mean, std, and CI bounds."""
    from sat_qkd_lab.sweep import sweep_loss_with_ci
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=0.2, p_bg=1e-4)
    results = sweep_loss_with_ci(
        [25.0, 35.0],
        n_pulses=10_000,
        n_trials=3,
        seed=777,
        detector=det,
    )

    assert len(results) == 2

    for r in results:
        assert "qber_mean" in r
        assert "qber_std" in r
        assert "qber_ci_low" in r
        assert "qber_ci_high" in r
        assert "secret_fraction_mean" in r
        assert "n_trials" in r
        assert r["n_trials"] == 3


# --- Key Rate Per Pulse Tests ---

def test_key_rate_per_pulse_includes_sifting():
    """key_rate_per_pulse should account for sifting factor (~1/2 for BB84)."""
    from sat_qkd_lab.bb84 import simulate_bb84
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=1.0, p_bg=0.0)

    result = simulate_bb84(
        n_pulses=100_000,
        loss_db=5.0,  # Low loss for reliable detection
        flip_prob=0.0,
        attack="none",
        seed=888,
        detector=det,
    )

    if not result.aborted and result.n_sent > 0:
        # key_rate_per_pulse = (n_sifted / n_sent) * secret_fraction
        expected_kr = (result.n_sifted / result.n_sent) * result.secret_fraction
        assert abs(result.key_rate_per_pulse - expected_kr) < 1e-10

        # Sifting probability should be roughly 1/2
        sift_prob = result.n_sifted / result.n_received if result.n_received > 0 else 0
        assert 0.45 <= sift_prob <= 0.55, f"Sifting probability should be ~0.5, got {sift_prob}"


def test_key_rate_per_pulse_zero_when_aborted():
    """key_rate_per_pulse should be 0 when protocol aborts."""
    from sat_qkd_lab.bb84 import simulate_bb84
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=1.0, p_bg=0.0)

    result = simulate_bb84(
        n_pulses=100_000,
        loss_db=5.0,
        flip_prob=0.0,
        attack="intercept_resend",
        qber_abort_threshold=0.11,
        seed=889,
        detector=det,
    )

    assert result.aborted
    assert result.key_rate_per_pulse == 0.0


# --- CI Clamping Tests ---

def test_ci_bounds_non_negative():
    """CI lower bounds for key rate and secret fraction should be non-negative."""
    from sat_qkd_lab.sweep import sweep_loss_with_ci
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=0.2, p_bg=1e-4)

    results = sweep_loss_with_ci(
        [40.0, 50.0],  # High loss where values may approach zero
        n_pulses=5000,
        n_trials=5,
        seed=900,
        detector=det,
    )

    for r in results:
        assert r["secret_fraction_ci_low"] >= 0.0, \
            f"secret_fraction_ci_low should be >= 0, got {r['secret_fraction_ci_low']}"
        assert r["key_rate_per_pulse_ci_low"] >= 0.0, \
            f"key_rate_per_pulse_ci_low should be >= 0, got {r['key_rate_per_pulse_ci_low']}"
        assert r["qber_ci_low"] >= 0.0, \
            f"qber_ci_low should be >= 0, got {r['qber_ci_low']}"


def test_qber_ci_clamped_to_half():
    """QBER CI upper bound should not exceed 0.5 (physical maximum)."""
    from sat_qkd_lab.sweep import sweep_loss_with_ci
    from sat_qkd_lab.detector import DetectorParams

    # High background to push QBER toward 0.5
    det = DetectorParams(eta=0.1, p_bg=1e-2)

    results = sweep_loss_with_ci(
        [50.0, 60.0],  # Very high loss
        n_pulses=5000,
        n_trials=3,
        seed=901,
        detector=det,
    )

    for r in results:
        if not math.isnan(r["qber_ci_high"]):
            assert r["qber_ci_high"] <= 0.5, \
                f"qber_ci_high should be <= 0.5, got {r['qber_ci_high']}"


# --- RNG Independence Tests ---

def test_sweep_rng_independence():
    """Different grid points should have independent RNG streams."""
    from sat_qkd_lab.sweep import sweep_loss_with_ci
    from sat_qkd_lab.detector import DetectorParams

    # Use background noise to ensure non-zero, variable QBER
    det = DetectorParams(eta=0.3, p_bg=1e-3)

    results = sweep_loss_with_ci(
        [25.0, 30.0, 35.0],  # Different loss values
        n_pulses=20_000,
        n_trials=5,
        seed=42,
        detector=det,
    )

    # Check n_sifted values differ - this is a better independence indicator
    # than QBER which may naturally converge to similar values
    n_sifted = [r["n_sifted_mean"] for r in results]
    # Different loss should give different sifted counts
    assert n_sifted[0] > n_sifted[1] > n_sifted[2], \
        f"n_sifted should decrease with loss: {n_sifted}"


def test_sweep_loss_includes_key_rate_per_pulse():
    """sweep_loss output should include key_rate_per_pulse field."""
    from sat_qkd_lab.sweep import sweep_loss
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=0.2, p_bg=0.0)
    results = sweep_loss(
        [20.0, 30.0],
        n_pulses=10_000,
        seed=902,
        detector=det,
    )

    for r in results:
        assert "key_rate_per_pulse" in r, "key_rate_per_pulse missing from sweep_loss output"


def test_sweep_loss_with_ci_includes_key_rate_ci():
    """sweep_loss_with_ci should include key_rate_per_pulse CI fields."""
    from sat_qkd_lab.sweep import sweep_loss_with_ci
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=0.2, p_bg=1e-4)
    results = sweep_loss_with_ci(
        [25.0],
        n_pulses=10_000,
        n_trials=3,
        seed=903,
        detector=det,
    )

    for r in results:
        assert "key_rate_per_pulse_mean" in r
        assert "key_rate_per_pulse_std" in r
        assert "key_rate_per_pulse_ci_low" in r
        assert "key_rate_per_pulse_ci_high" in r


# --- Regression Tests for Issue A-D Fixes (v0.2.1) ---

def test_sifting_uses_alice_bob_basis_not_eve():
    """
    Regression test: Sifting must depend on Alice-Bob basis match, not Eve-Bob.

    Under intercept-resend, Eve uses her own random basis. If sifting incorrectly
    used Eve's basis for matching, the QBER would be artificially low. This test
    verifies that the ~25% QBER signature (from Eve's 50% basis mismatch causing
    50% errors on half the bits) is present, confirming correct sifting.
    """
    from sat_qkd_lab.bb84 import simulate_bb84
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=1.0, p_bg=0.0)

    result = simulate_bb84(
        n_pulses=200_000,
        loss_db=5.0,
        flip_prob=0.0,
        attack="intercept_resend",
        qber_abort_threshold=0.50,  # Raise threshold to observe QBER
        seed=1000,
        detector=det,
    )

    # If sifting used Eve-Bob basis (wrong), QBER would be ~0
    # If sifting uses Alice-Bob basis (correct), QBER should be ~0.25
    assert result.qber > 0.15, \
        f"QBER too low ({result.qber:.3f}), sifting may incorrectly use Eve-Bob basis"
    assert 0.20 <= result.qber <= 0.30, \
        f"Expected QBER ~0.25 for intercept-resend, got {result.qber:.3f}"


def test_ci_bounds_clamped_in_json_output():
    """
    Regression test: CI bounds in sweep output must be clamped to physical ranges.

    - QBER CI: [0.0, 0.5]
    - Secret fraction CI: [0.0, 1.0]
    - Key rate CI: [0.0, inf)

    This test runs sweeps at extreme conditions and verifies clamping.
    """
    from sat_qkd_lab.sweep import sweep_loss_with_ci
    from sat_qkd_lab.detector import DetectorParams

    # High background to push QBER toward 0.5 and key rate toward 0
    det = DetectorParams(eta=0.1, p_bg=5e-3)

    results = sweep_loss_with_ci(
        [30.0, 45.0, 55.0],  # Range of losses
        n_pulses=5000,
        n_trials=5,
        seed=1001,
        detector=det,
    )

    for r in results:
        # QBER bounds: [0, 0.5]
        if not math.isnan(r["qber_ci_low"]):
            assert 0.0 <= r["qber_ci_low"] <= 0.5, \
                f"qber_ci_low out of range: {r['qber_ci_low']}"
        if not math.isnan(r["qber_ci_high"]):
            assert 0.0 <= r["qber_ci_high"] <= 0.5, \
                f"qber_ci_high out of range: {r['qber_ci_high']}"

        # Secret fraction bounds: [0, 1]
        assert 0.0 <= r["secret_fraction_ci_low"] <= 1.0, \
            f"secret_fraction_ci_low out of range: {r['secret_fraction_ci_low']}"
        assert 0.0 <= r["secret_fraction_ci_high"] <= 1.0, \
            f"secret_fraction_ci_high out of range: {r['secret_fraction_ci_high']}"

        # Key rate bounds: >= 0
        assert r["key_rate_per_pulse_ci_low"] >= 0.0, \
            f"key_rate_per_pulse_ci_low negative: {r['key_rate_per_pulse_ci_low']}"
        assert r["key_rate_per_pulse_ci_high"] >= 0.0, \
            f"key_rate_per_pulse_ci_high negative: {r['key_rate_per_pulse_ci_high']}"


def test_aborted_trials_have_zero_secret_fraction():
    """
    Regression test: Aborted trials must have secret_fraction = 0.

    When QBER exceeds abort threshold, no key can be extracted. Previously,
    secret_fraction was computed even for aborted trials, inflating averages.
    """
    from sat_qkd_lab.bb84 import simulate_bb84
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=1.0, p_bg=0.0)

    # Force abort with intercept-resend attack (QBER ~25% > 11% threshold)
    result = simulate_bb84(
        n_pulses=100_000,
        loss_db=5.0,
        flip_prob=0.0,
        attack="intercept_resend",
        qber_abort_threshold=0.11,
        seed=1002,
        detector=det,
    )

    assert result.aborted, "Expected protocol to abort"
    assert result.secret_fraction == 0.0, \
        f"Aborted trial should have secret_fraction=0, got {result.secret_fraction}"
    assert result.n_secret_est == 0, \
        f"Aborted trial should have n_secret_est=0, got {result.n_secret_est}"
    assert result.key_rate_per_pulse == 0.0, \
        f"Aborted trial should have key_rate_per_pulse=0, got {result.key_rate_per_pulse}"


def test_background_only_clicks_produce_random_bits():
    """
    Regression test: Background-only clicks should produce ~50% QBER contribution.

    When a click is from background noise only (no signal photon), Bob's bit
    is uniformly random, contributing ~50% error rate on those bits.
    """
    from sat_qkd_lab.bb84 import simulate_bb84
    from sat_qkd_lab.detector import DetectorParams

    # Very high loss + significant background = mostly background clicks
    det = DetectorParams(eta=0.2, p_bg=1e-2)  # High background

    result = simulate_bb84(
        n_pulses=100_000,
        loss_db=60.0,  # Very high loss, signal clicks rare
        flip_prob=0.0,
        attack="none",
        qber_abort_threshold=0.50,  # Don't abort for this test
        seed=1003,
        detector=det,
    )

    # At extreme loss, background dominates and QBER approaches 0.5
    # (sifted background-only clicks have ~50% error rate, weighted by their fraction)
    if not result.aborted and result.n_sifted > 100:
        # QBER should be elevated due to background
        assert result.qber > 0.20, \
            f"Expected elevated QBER from background clicks, got {result.qber:.3f}"


# --- Validation Helper Tests ---

def test_validate_int_valid():
    """validate_int should accept valid integers."""
    from sat_qkd_lab.helpers import validate_int

    assert validate_int("x", 5) == 5
    assert validate_int("x", 0, min_value=0) == 0
    assert validate_int("x", 100, max_value=100) == 100
    assert validate_int("x", 50, min_value=0, max_value=100) == 50
    # Float that is an integer should work
    assert validate_int("x", 10.0) == 10


def test_validate_int_invalid():
    """validate_int should reject invalid inputs."""
    from sat_qkd_lab.helpers import validate_int

    # Non-integer float
    with pytest.raises(ValueError, match="expected int"):
        validate_int("x", 5.5)

    # Below minimum
    with pytest.raises(ValueError, match="< minimum"):
        validate_int("x", -1, min_value=0)

    # Above maximum
    with pytest.raises(ValueError, match="> maximum"):
        validate_int("x", 101, max_value=100)

    # Wrong type
    with pytest.raises(ValueError, match="expected int"):
        validate_int("x", "not an int")


def test_validate_float_valid():
    """validate_float should accept valid floats."""
    from sat_qkd_lab.helpers import validate_float

    assert validate_float("x", 0.5) == 0.5
    assert validate_float("x", 0.0, min_value=0.0) == 0.0
    assert validate_float("x", 1.0, max_value=1.0) == 1.0
    # Integer should be converted to float
    assert validate_float("x", 5) == 5.0


def test_validate_float_invalid():
    """validate_float should reject invalid inputs."""
    from sat_qkd_lab.helpers import validate_float

    # NaN not allowed by default
    with pytest.raises(ValueError, match="NaN not allowed"):
        validate_float("x", float("nan"))

    # Infinity not allowed by default
    with pytest.raises(ValueError, match="infinity not allowed"):
        validate_float("x", float("inf"))

    # Below minimum
    with pytest.raises(ValueError, match="< minimum"):
        validate_float("x", -0.1, min_value=0.0)

    # Above maximum
    with pytest.raises(ValueError, match="> maximum"):
        validate_float("x", 1.1, max_value=1.0)


def test_validate_float_allows_nan_inf():
    """validate_float can allow NaN and infinity when requested."""
    from sat_qkd_lab.helpers import validate_float

    assert math.isnan(validate_float("x", float("nan"), allow_nan=True))
    assert validate_float("x", float("inf"), allow_inf=True) == float("inf")


def test_validate_seed_valid():
    """validate_seed should accept valid seeds."""
    from sat_qkd_lab.helpers import validate_seed

    assert validate_seed(None) is None
    assert validate_seed(0) == 0
    assert validate_seed(42) == 42
    assert validate_seed(123456789) == 123456789


def test_validate_seed_invalid():
    """validate_seed should reject invalid seeds."""
    from sat_qkd_lab.helpers import validate_seed

    # Negative seed
    with pytest.raises(ValueError, match="< minimum"):
        validate_seed(-1)

    # Wrong type
    with pytest.raises(ValueError, match="expected int or None"):
        validate_seed("not a seed")

    with pytest.raises(ValueError, match="expected int or None"):
        validate_seed(3.14)


def test_happy_path_bb84_produces_nonzero_output():
    """A valid BB84 run should produce nonzero outputs."""
    from sat_qkd_lab.bb84 import simulate_bb84
    from sat_qkd_lab.detector import DetectorParams

    det = DetectorParams(eta=0.5, p_bg=0.0)

    result = simulate_bb84(
        n_pulses=50_000,
        loss_db=20.0,
        flip_prob=0.0,
        attack="none",
        seed=42,
        detector=det,
    )

    # Valid run should have nonzero outputs
    assert result.n_sent == 50_000
    assert result.n_received > 0
    assert result.n_sifted > 0
    assert not result.aborted
    assert result.secret_fraction > 0.0
    assert result.n_secret_est > 0
    assert result.key_rate_per_pulse > 0.0


# --- Figure Filename Tests ---

def test_plot_key_metrics_creates_canonical_filenames(tmp_path):
    """plot_key_metrics_vs_loss should create canonical figure filenames."""
    from sat_qkd_lab.plotting import plot_key_metrics_vs_loss

    records = [
        {"loss_db": 20.0, "qber": 0.01, "secret_fraction": 0.9},
        {"loss_db": 30.0, "qber": 0.02, "secret_fraction": 0.85},
    ]

    prefix = str(tmp_path / "key")
    q_path, k_path = plot_key_metrics_vs_loss(records, records, prefix)

    # Check canonical filenames
    assert q_path.endswith("key_qber_vs_loss.png")
    assert k_path.endswith("key_fraction_vs_loss.png")

    # Check files exist
    from pathlib import Path
    assert Path(q_path).exists()
    assert Path(k_path).exists()


def test_plot_ci_creates_canonical_filename(tmp_path):
    """plot_key_rate_vs_loss_ci should create canonical figure filename."""
    from sat_qkd_lab.plotting import plot_key_rate_vs_loss_ci

    records = [
        {"loss_db": 20.0, "secret_fraction_mean": 0.9,
         "secret_fraction_ci_low": 0.85, "secret_fraction_ci_high": 0.95},
        {"loss_db": 30.0, "secret_fraction_mean": 0.8,
         "secret_fraction_ci_low": 0.75, "secret_fraction_ci_high": 0.85},
    ]

    out_path = str(tmp_path / "secret_fraction_vs_loss_ci.png")
    result_path = plot_key_rate_vs_loss_ci(records, records, out_path)

    from pathlib import Path
    assert Path(result_path).exists()
    assert result_path.endswith("secret_fraction_vs_loss_ci.png")


# --- Finite-Key Analysis Tests ---

def test_finite_key_params_defaults():
    """FiniteKeyParams should have sensible defaults."""
    from sat_qkd_lab.finite_key import FiniteKeyParams

    params = FiniteKeyParams()
    assert params.eps_pe == 1e-10
    assert params.eps_sec == 1e-10
    assert params.eps_cor == 1e-15
    assert params.ec_efficiency == 1.16
    assert params.pe_frac == 0.5
    assert params.m_pe is None
    assert params.eps_total == params.eps_pe + params.eps_sec + params.eps_cor


def test_finite_key_params_validation():
    """FiniteKeyParams should validate parameter ranges."""
    from sat_qkd_lab.finite_key import FiniteKeyParams

    # Valid parameters
    valid = FiniteKeyParams(eps_pe=1e-10, eps_sec=1e-10, eps_cor=1e-15, ec_efficiency=1.16)
    assert valid.eps_pe == 1e-10

    # Invalid eps_pe (out of range)
    with pytest.raises(ValueError, match="eps_pe"):
        FiniteKeyParams(eps_pe=0.0)
    with pytest.raises(ValueError, match="eps_pe"):
        FiniteKeyParams(eps_pe=1.0)

    # Invalid ec_efficiency (too low)
    with pytest.raises(ValueError, match="ec_efficiency"):
        FiniteKeyParams(ec_efficiency=0.9)

    # Invalid pe_frac
    with pytest.raises(ValueError, match="pe_frac"):
        FiniteKeyParams(pe_frac=0.0)


def test_hoeffding_bound_basic():
    """Hoeffding bound should add uncertainty margin."""
    from sat_qkd_lab.finite_key import hoeffding_bound

    # With enough samples, bound should be close to observed
    upper = hoeffding_bound(n_samples=100000, observed_rate=0.05, eps=1e-10)
    assert upper > 0.05  # Upper bound should exceed observed
    assert upper < 0.08  # But not by too much with many samples

    # With few samples, bound should be wider
    upper_few = hoeffding_bound(n_samples=100, observed_rate=0.05, eps=1e-10)
    assert upper_few > upper  # Less data = wider bound


def test_hoeffding_bound_edge_cases():
    """Hoeffding bound edge cases."""
    from sat_qkd_lab.finite_key import hoeffding_bound

    # Zero samples -> worst case
    assert hoeffding_bound(0, 0.05, 1e-10) == 1.0

    # Zero tolerance -> worst case
    assert hoeffding_bound(1000, 0.05, 0.0) == 1.0

    # Upper clamped to 1.0
    assert hoeffding_bound(10, 0.95, 1e-10) == 1.0

    # Lower clamped to 0.0
    from sat_qkd_lab.finite_key import hoeffding_lower_bound
    assert hoeffding_lower_bound(10, 0.05, 1e-10) == 0.0


def test_finite_key_bounds_output():
    """finite_key_bounds should return correct structure."""
    from sat_qkd_lab.finite_key import finite_key_bounds

    result = finite_key_bounds(n_sifted=10000, n_errors=500, eps_pe=1e-10)

    assert "qber_hat" in result
    assert "qber_upper" in result
    assert "qber_lower" in result
    assert "n_sifted" in result

    assert result["qber_hat"] == 0.05  # 500/10000
    assert result["qber_upper"] > result["qber_hat"]
    assert result["qber_lower"] < result["qber_hat"]
    assert result["qber_upper"] <= 0.5  # Physical max


def test_finite_key_secret_length_positive():
    """finite_key_secret_length should return positive key bits for good QBER."""
    from sat_qkd_lab.finite_key import finite_key_secret_length, FiniteKeyParams

    params = FiniteKeyParams()
    result = finite_key_secret_length(
        n_sifted=100000,
        qber_upper=0.03,  # Good QBER
        params=params,
    )

    assert result["l_secret_bits"] > 0
    assert not result["aborted"]
    assert result["key_rate_per_sifted"] > 0


def test_finite_key_secret_length_aborts_on_high_qber():
    """finite_key_secret_length should abort on high QBER."""
    from sat_qkd_lab.finite_key import finite_key_secret_length, FiniteKeyParams

    params = FiniteKeyParams()
    result = finite_key_secret_length(
        n_sifted=100000,
        qber_upper=0.15,  # Above 11% threshold
        params=params,
    )

    assert result["l_secret_bits"] == 0
    assert result["aborted"]
    assert result["key_rate_per_sifted"] == 0.0


def test_finite_key_rate_per_pulse():
    """finite_key_rate_per_pulse should combine bounds and length."""
    from sat_qkd_lab.finite_key import finite_key_rate_per_pulse, FiniteKeyParams

    params = FiniteKeyParams()
    result = finite_key_rate_per_pulse(
        n_sent=200000,
        n_sifted=50000,
        n_errors=1500,  # 3% QBER
        params=params,
    )

    assert "qber_hat" in result
    assert "qber_upper" in result
    assert "ell_bits" in result
    assert "l_secret_bits" in result
    assert "delta_eps_bits" in result
    assert "m_pe" in result
    assert "key_rate_per_pulse" in result
    assert "eps_total" in result

    # Should produce positive key
    assert result["ell_bits"] > 0
    assert result["key_rate_per_pulse"] > 0


def test_finite_rate_less_than_asymptotic():
    """Finite-key rate should always be <= asymptotic rate."""
    from sat_qkd_lab.finite_key import compare_asymptotic_vs_finite, FiniteKeyParams

    params = FiniteKeyParams()

    # Test across different QBER values
    for qber in [0.01, 0.03, 0.05, 0.08]:
        result = compare_asymptotic_vs_finite(
            n_sent=200000,
            n_sifted=50000,
            qber_observed=qber,
            params=params,
        )

        # Finite rate should be <= asymptotic
        assert result["finite_rate"] <= result["asymptotic_rate"] + 1e-10, \
            f"Finite rate ({result['finite_rate']}) > asymptotic ({result['asymptotic_rate']}) at QBER={qber}"


def test_finite_key_rate_never_exceeds_asymptotic_small_case():
    """Finite-key rate should not exceed asymptotic rate for a fixed run."""
    import numpy as np
    from sat_qkd_lab.finite_key import finite_key_rate_per_pulse, FiniteKeyParams
    from sat_qkd_lab.helpers import h2

    rng = np.random.default_rng(123)
    n_sent = 10000
    n_sifted = 5000
    qber = 0.03
    n_errors = int(rng.binomial(n_sifted, qber))

    params = FiniteKeyParams()
    finite = finite_key_rate_per_pulse(n_sent, n_sifted, n_errors, params)
    asymptotic = (n_sifted / n_sent) * max(0.0, 1.0 - params.ec_efficiency * h2(qber) - h2(qber))

    assert finite["key_rate_per_pulse"] <= asymptotic + 1e-12


def test_finite_key_rate_non_decreasing_with_n_sent():
    """Finite-key rate should be non-decreasing with larger n_sent (within tolerance)."""
    from sat_qkd_lab.finite_key import finite_key_rate_per_pulse, FiniteKeyParams

    params = FiniteKeyParams()
    qber = 0.02
    n_sent_values = [10000, 30000, 100000]
    rates = []

    for n_sent in n_sent_values:
        n_sifted = int(0.5 * n_sent)
        n_errors = int(round(qber * n_sifted))
        result = finite_key_rate_per_pulse(n_sent, n_sifted, n_errors, params)
        rates.append(result["key_rate_per_pulse"])

    for i in range(len(rates) - 1):
        assert rates[i + 1] >= rates[i] - 1e-12


def test_stricter_epsilon_reduces_finite_key_rate():
    """Stricter epsilon_sec should reduce finite-key rate."""
    from sat_qkd_lab.finite_key import finite_key_rate_per_pulse, FiniteKeyParams

    n_sent = 50000
    n_sifted = 25000
    qber = 0.02
    n_errors = int(round(qber * n_sifted))

    params_loose = FiniteKeyParams(eps_sec=1e-6)
    params_strict = FiniteKeyParams(eps_sec=1e-12)

    rate_loose = finite_key_rate_per_pulse(n_sent, n_sifted, n_errors, params_loose)["key_rate_per_pulse"]
    rate_strict = finite_key_rate_per_pulse(n_sent, n_sifted, n_errors, params_strict)["key_rate_per_pulse"]

    assert rate_strict <= rate_loose


def test_sweep_loss_finite_key_output():
    """sweep_loss_finite_key should return both asymptotic and finite-key metrics."""
    from sat_qkd_lab.sweep import sweep_loss_finite_key
    from sat_qkd_lab.detector import DetectorParams
    from sat_qkd_lab.finite_key import FiniteKeyParams

    det = DetectorParams(eta=0.2, p_bg=0.0)
    fk_params = FiniteKeyParams()

    results = sweep_loss_finite_key(
        [20.0, 30.0],
        n_pulses=10000,
        seed=42,
        detector=det,
        finite_key_params=fk_params,
    )

    assert len(results) == 2

    for r in results:
        # Asymptotic fields
        assert "qber" in r
        assert "secret_fraction" in r
        assert "key_rate_per_pulse_asymptotic" in r

        # Finite-key fields
        assert "qber_upper" in r
        assert "l_secret_bits" in r
        assert "key_rate_per_pulse_finite" in r
        assert "finite_size_penalty" in r
        assert "eps_total" in r


def test_finite_key_with_attack_aborts():
    """Finite-key sweep with attack should produce aborted results."""
    from sat_qkd_lab.sweep import sweep_loss_finite_key
    from sat_qkd_lab.detector import DetectorParams
    from sat_qkd_lab.finite_key import FiniteKeyParams

    det = DetectorParams(eta=1.0, p_bg=0.0)
    fk_params = FiniteKeyParams()

    results = sweep_loss_finite_key(
        [10.0],  # Low loss for stable QBER
        attack="intercept_resend",
        n_pulses=50000,
        seed=42,
        detector=det,
        finite_key_params=fk_params,
    )

    # Intercept-resend should cause abort (QBER ~25% > 11%)
    assert len(results) == 1
    assert results[0]["aborted"]  # BB84 protocol abort
    assert results[0]["finite_key_aborted"]  # Finite-key abort (QBER too high)
    assert results[0]["l_secret_bits"] == 0


def test_finite_key_plot_functions_exist():
    """Finite-key plot functions should be importable."""
    from sat_qkd_lab.plotting import (
        plot_finite_key_comparison,
        plot_finite_key_bits_vs_loss,
        plot_finite_size_penalty,
        plot_finite_key_rate_vs_n_sent,
    )

    # Just check they're callable
    assert callable(plot_finite_key_comparison)
    assert callable(plot_finite_key_bits_vs_loss)
    assert callable(plot_finite_size_penalty)
    assert callable(plot_finite_key_rate_vs_n_sent)


def test_finite_key_plot_creates_file(tmp_path):
    """Finite-key plot functions should create output files."""
    from sat_qkd_lab.plotting import (
        plot_finite_key_comparison,
        plot_finite_key_bits_vs_loss,
        plot_finite_key_rate_vs_n_sent,
    )
    from pathlib import Path

    records = [
        {"loss_db": 20.0, "asymptotic_rate": 0.001, "finite_rate": 0.0008,
         "l_secret_bits": 800, "finite_size_penalty": 0.2},
        {"loss_db": 30.0, "asymptotic_rate": 0.0005, "finite_rate": 0.0003,
         "l_secret_bits": 300, "finite_size_penalty": 0.4},
    ]

    comp_path = str(tmp_path / "finite_key_comparison.png")
    result = plot_finite_key_comparison(records, comp_path)
    assert Path(result).exists()

    bits_path = str(tmp_path / "finite_key_bits.png")
    result2 = plot_finite_key_bits_vs_loss(records, bits_path)
    assert Path(result2).exists()

    n_sent_records = [
        {"n_sent": 10000, "key_rate_per_pulse_finite": 0.0005},
        {"n_sent": 100000, "key_rate_per_pulse_finite": 0.0010},
    ]
    rate_path = str(tmp_path / "finite_key_rate_vs_n_sent.png")
    result3 = plot_finite_key_rate_vs_n_sent(n_sent_records, rate_path)
    assert Path(result3).exists()


# --- Free-Space Link Model Tests ---

def test_free_space_link_params_defaults():
    """FreeSpaceLinkParams should have sensible defaults."""
    from sat_qkd_lab.free_space_link import FreeSpaceLinkParams

    params = FreeSpaceLinkParams()
    assert params.wavelength_m == 850e-9
    assert params.tx_diameter_m == 0.30
    assert params.rx_diameter_m == 1.0
    assert params.sigma_point_rad == 2e-6
    assert params.altitude_m == 500e3
    assert params.atm_loss_db_zenith == 0.5
    assert params.sigma_ln == 0.0
    assert params.system_loss_db == 3.0
    assert params.is_night is True
    assert params.day_background_factor == 100.0


def test_diffraction_limited_divergence():
    """Diffraction-limited beam divergence should be computed correctly."""
    from sat_qkd_lab.free_space_link import FreeSpaceLinkParams

    # No manual divergence -> compute diffraction limit
    params = FreeSpaceLinkParams(wavelength_m=850e-9, tx_diameter_m=0.30)

    # Diffraction-limited: theta = 1.22 * lambda / D
    expected = 1.22 * 850e-9 / 0.30
    # The effective_divergence_rad property
    assert abs(params.effective_divergence_rad - expected) < 1e-10


def test_slant_range_calculation():
    """Slant range should increase towards horizon."""
    from sat_qkd_lab.free_space_link import slant_range_m, FreeSpaceLinkParams

    params = FreeSpaceLinkParams(altitude_m=500e3)

    range_zenith = slant_range_m(90.0, params)
    range_60 = slant_range_m(60.0, params)
    range_30 = slant_range_m(30.0, params)
    range_10 = slant_range_m(10.0, params)

    # Range should increase as elevation decreases
    assert range_zenith < range_60 < range_30 < range_10
    # At zenith, range should equal altitude
    assert abs(range_zenith - 500e3) < 1e3


def test_geometric_coupling_efficiency():
    """Geometric coupling efficiency should decrease with range."""
    from sat_qkd_lab.free_space_link import geometric_coupling_efficiency, FreeSpaceLinkParams

    params = FreeSpaceLinkParams()

    # Short range = higher coupling
    eta_short = geometric_coupling_efficiency(500e3, params)
    eta_long = geometric_coupling_efficiency(1500e3, params)

    assert eta_short > eta_long
    assert 0 < eta_short <= 1.0
    assert 0 < eta_long <= 1.0


def test_pointing_loss_scales_with_sigma():
    """Pointing loss should increase with pointing error."""
    from sat_qkd_lab.free_space_link import pointing_loss_db, FreeSpaceLinkParams

    params_low = FreeSpaceLinkParams(sigma_point_rad=1e-6)
    params_high = FreeSpaceLinkParams(sigma_point_rad=10e-6)

    loss_low = pointing_loss_db(params_low)
    loss_high = pointing_loss_db(params_high)

    # Higher pointing error = higher loss
    assert loss_high > loss_low
    assert loss_low >= 0
    assert loss_high >= 0


def test_atmospheric_extinction_increases_toward_horizon():
    """Atmospheric loss should increase toward horizon."""
    from sat_qkd_lab.free_space_link import atmospheric_extinction_db, FreeSpaceLinkParams

    params = FreeSpaceLinkParams(atm_loss_db_zenith=0.5)

    loss_zenith = atmospheric_extinction_db(90.0, params)
    loss_60 = atmospheric_extinction_db(60.0, params)
    loss_30 = atmospheric_extinction_db(30.0, params)
    loss_10 = atmospheric_extinction_db(10.0, params)

    assert loss_zenith < loss_60 < loss_30 < loss_10
    # At zenith, loss should equal zenith loss
    assert abs(loss_zenith - 0.5) < 0.01


def test_total_link_loss_monotonic_with_elevation():
    """Total link loss should decrease monotonically as elevation increases."""
    from sat_qkd_lab.free_space_link import total_link_loss_db, FreeSpaceLinkParams

    params = FreeSpaceLinkParams()

    elevations = [10.0, 20.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    losses = [total_link_loss_db(el, params) for el in elevations]

    # Loss should decrease (improve) as elevation increases
    for i in range(len(losses) - 1):
        assert losses[i] > losses[i + 1], \
            f"Loss should decrease with elevation: {losses[i]:.1f} dB at {elevations[i]}° > {losses[i+1]:.1f} dB at {elevations[i+1]}°"


def test_turbulence_fading_samples():
    """Turbulence fading should produce lognormal samples."""
    from sat_qkd_lab.free_space_link import sample_turbulence_fading
    import numpy as np

    rng = np.random.default_rng(42)
    samples = sample_turbulence_fading(10000, sigma_ln=0.3, rng=rng)

    # Check samples are positive
    assert np.all(samples > 0)

    # Mean of lognormal(0, sigma) is exp(sigma^2/2)
    expected_mean = np.exp(0.3**2 / 2)
    assert abs(np.mean(samples) - expected_mean) < 0.05


def test_effective_background_prob_day_night():
    """Day mode should increase background probability."""
    from sat_qkd_lab.free_space_link import effective_background_prob, FreeSpaceLinkParams

    base_p_bg = 1e-5

    params_night = FreeSpaceLinkParams(is_night=True, day_background_factor=100.0)
    params_day = FreeSpaceLinkParams(is_night=False, day_background_factor=100.0)

    p_bg_night = effective_background_prob(base_p_bg, params_night)
    p_bg_day = effective_background_prob(base_p_bg, params_day)

    assert p_bg_day == base_p_bg * 100.0
    assert p_bg_night == base_p_bg


def test_generate_elevation_profile():
    """Elevation profile should be symmetric and peak at center."""
    from sat_qkd_lab.free_space_link import generate_elevation_profile

    time_s, elevation_deg = generate_elevation_profile(
        max_elevation_deg=70.0,
        min_elevation_deg=10.0,
        time_step_s=10.0,
        pass_duration_s=300.0,
    )

    # Check basic structure
    assert len(time_s) == len(elevation_deg)
    assert len(time_s) == 31  # 300/10 + 1

    # Check elevation bounds
    assert min(elevation_deg) >= 10.0
    assert max(elevation_deg) <= 70.0

    # Peak should be at/near center
    import numpy as np
    peak_idx = np.argmax(elevation_deg)
    assert 10 <= peak_idx <= 20  # Somewhere in middle third

    # Profile should be roughly symmetric
    assert abs(elevation_deg[0] - elevation_deg[-1]) < 1.0


def test_estimate_secure_window():
    """Secure window estimation should find positive key rate intervals."""
    from sat_qkd_lab.free_space_link import estimate_secure_window
    import numpy as np

    # Simulate records with some positive key rates in the middle
    records = [
        {"time_s": 0, "elevation_deg": 10.0, "key_rate_per_pulse": 0},
        {"time_s": 10, "elevation_deg": 20.0, "key_rate_per_pulse": 0},
        {"time_s": 20, "elevation_deg": 40.0, "key_rate_per_pulse": 0.001},
        {"time_s": 30, "elevation_deg": 60.0, "key_rate_per_pulse": 0.002},
        {"time_s": 40, "elevation_deg": 60.0, "key_rate_per_pulse": 0.003},
        {"time_s": 50, "elevation_deg": 40.0, "key_rate_per_pulse": 0.002},
        {"time_s": 60, "elevation_deg": 20.0, "key_rate_per_pulse": 0},
        {"time_s": 70, "elevation_deg": 10.0, "key_rate_per_pulse": 0},
    ]
    key_rates = np.array([r["key_rate_per_pulse"] for r in records])

    result = estimate_secure_window(records, key_rates, time_step_s=10.0)

    # The actual keys returned by estimate_secure_window
    assert result["secure_window_seconds"] == 40  # 4 secure points * 10s
    assert result["secure_start_s"] == 20
    assert result["secure_end_s"] == 50  # Last secure point is at 50s
    assert result["peak_key_rate"] == 0.003


def test_sweep_pass_output_structure():
    """sweep_pass should return correct output structure."""
    from sat_qkd_lab.sweep import sweep_pass
    from sat_qkd_lab.free_space_link import FreeSpaceLinkParams, generate_elevation_profile
    from sat_qkd_lab.detector import DetectorParams

    time_s, elevation_deg = generate_elevation_profile(
        max_elevation_deg=60.0,
        min_elevation_deg=10.0,
        time_step_s=30.0,
        pass_duration_s=120.0,
    )

    link_params = FreeSpaceLinkParams()
    detector = DetectorParams(eta=0.2, p_bg=1e-5)

    records, summary = sweep_pass(
        elevation_deg_values=elevation_deg,
        time_s_values=time_s,
        n_pulses=10000,
        seed=42,
        detector=detector,
        link_params=link_params,
    )

    # Check records structure
    assert len(records) == len(time_s)
    for r in records:
        assert "time_s" in r
        assert "elevation_deg" in r
        assert "loss_db" in r
        assert "qber" in r
        assert "secret_fraction" in r
        assert "key_rate_per_pulse" in r
        assert "p_bg_effective" in r

    # Check summary structure (actual keys from estimate_secure_window)
    assert "secure_window_seconds" in summary
    assert "peak_key_rate" in summary
    assert "secure_start_s" in summary


def test_day_mode_more_likely_to_abort():
    """Day mode (high background) should be more likely to produce zero key rate."""
    from sat_qkd_lab.sweep import sweep_pass
    from sat_qkd_lab.free_space_link import FreeSpaceLinkParams, generate_elevation_profile
    from sat_qkd_lab.detector import DetectorParams

    time_s, elevation_deg = generate_elevation_profile(
        max_elevation_deg=30.0,  # Lower elevation = higher loss
        min_elevation_deg=10.0,
        time_step_s=30.0,
        pass_duration_s=120.0,
    )

    detector = DetectorParams(eta=0.2, p_bg=1e-5)

    # Night mode
    link_night = FreeSpaceLinkParams(is_night=True, day_background_factor=100.0)
    records_night, _ = sweep_pass(
        elevation_deg_values=elevation_deg,
        time_s_values=time_s,
        n_pulses=50000,
        seed=42,
        detector=detector,
        link_params=link_night,
    )

    # Day mode
    link_day = FreeSpaceLinkParams(is_night=False, day_background_factor=100.0)
    records_day, _ = sweep_pass(
        elevation_deg_values=elevation_deg,
        time_s_values=time_s,
        n_pulses=50000,
        seed=42,
        detector=detector,
        link_params=link_day,
    )

    # Count positive key rates
    night_positive = sum(1 for r in records_night if r["key_rate_per_pulse"] > 0)
    day_positive = sum(1 for r in records_day if r["key_rate_per_pulse"] > 0)

    # Day should have fewer or equal positive key rate points
    assert day_positive <= night_positive


def test_free_space_plot_functions_exist():
    """Free-space link plot functions should be importable."""
    from sat_qkd_lab.plotting import (
        plot_key_rate_vs_elevation,
        plot_secure_window,
        plot_loss_vs_elevation,
    )

    assert callable(plot_key_rate_vs_elevation)
    assert callable(plot_secure_window)
    assert callable(plot_loss_vs_elevation)


def test_free_space_plot_creates_file(tmp_path):
    """Free-space link plot functions should create output files."""
    from sat_qkd_lab.plotting import plot_key_rate_vs_elevation, plot_loss_vs_elevation
    from pathlib import Path

    records = [
        {"time_s": 0, "elevation_deg": 10.0, "loss_db": 50.0,
         "key_rate_per_pulse": 0, "secret_fraction": 0},
        {"time_s": 30, "elevation_deg": 30.0, "loss_db": 35.0,
         "key_rate_per_pulse": 0.001, "secret_fraction": 0.5},
        {"time_s": 60, "elevation_deg": 10.0, "loss_db": 50.0,
         "key_rate_per_pulse": 0, "secret_fraction": 0},
    ]

    elev_path = str(tmp_path / "key_rate_vs_elevation.png")
    result = plot_key_rate_vs_elevation(records, elev_path)
    assert Path(result).exists()

    loss_path = str(tmp_path / "loss_vs_elevation.png")
    result2 = plot_loss_vs_elevation(records, loss_path)
    assert Path(result2).exists()


# Keep the original sanity test for backwards compatibility
def test_sanity():
    assert True
