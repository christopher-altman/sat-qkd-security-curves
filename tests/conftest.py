"""Pytest configuration and shared fixtures for sat-qkd-security-curves tests."""
import pytest
import numpy as np
import random
from sat_qkd_lab.detector import DetectorParams, DEFAULT_DETECTOR
from sat_qkd_lab.attacks import AttackConfig
from sat_qkd_lab.timing import TimingModel
from sat_qkd_lab.timetags import TimeTags
from sat_qkd_lab.link_budget import SatLinkParams
from sat_qkd_lab.finite_key import FiniteKeyParams
from sat_qkd_lab.decoy_bb84 import DecoyParams
from sat_qkd_lab.experiment import ExperimentParams


# ============================================================================
# NUMPY & RANDOM FIXTURES
# ============================================================================

@pytest.fixture
def rng_seed():
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def np_rng(rng_seed):
    """NumPy random generator with fixed seed."""
    return np.random.default_rng(rng_seed)


@pytest.fixture
def py_rng(rng_seed):
    """Python random generator with fixed seed."""
    rng = random.Random(rng_seed)
    return rng


# ============================================================================
# DETECTOR FIXTURES
# ============================================================================

@pytest.fixture
def default_detector():
    """Default detector parameters."""
    return DEFAULT_DETECTOR


@pytest.fixture
def perfect_detector():
    """Perfect detector (eta=1.0, p_bg=0.0)."""
    return DetectorParams(eta=1.0, p_bg=0.0)


@pytest.fixture
def realistic_detector():
    """Realistic detector parameters (eta=0.2, p_bg=1e-4)."""
    return DetectorParams(eta=0.2, p_bg=1e-4)


@pytest.fixture
def low_efficiency_detector():
    """Low efficiency detector (eta=0.05, p_bg=1e-4)."""
    return DetectorParams(eta=0.05, p_bg=1e-4)


@pytest.fixture
def high_background_detector():
    """High background detector (eta=0.2, p_bg=0.01)."""
    return DetectorParams(eta=0.2, p_bg=0.01)


# ============================================================================
# ATTACK FIXTURES
# ============================================================================

@pytest.fixture
def no_attack_config():
    """No attack configuration."""
    return AttackConfig(attack="none")


@pytest.fixture
def intercept_resend_config():
    """Intercept-resend attack configuration."""
    return AttackConfig(attack="intercept_resend")


@pytest.fixture
def pns_attack_config():
    """PNS attack configuration."""
    return AttackConfig(attack="pns", mu=0.6)


@pytest.fixture
def time_shift_attack_config():
    """Time-shift attack configuration."""
    return AttackConfig(attack="time_shift", timeshift_bias=0.5)


@pytest.fixture
def blinding_attack_config():
    """Blinding attack configuration."""
    return AttackConfig(attack="blinding", blinding_prob=0.05, blinding_mode="loud")


# ============================================================================
# TIMING FIXTURES
# ============================================================================

@pytest.fixture
def null_timing_model():
    """Null timing model (no offset, drift, jitter, or quantization)."""
    return TimingModel()


@pytest.fixture
def offset_timing_model():
    """Timing model with offset."""
    return TimingModel(delta_t=1e-6)


@pytest.fixture
def drift_timing_model():
    """Timing model with drift."""
    return TimingModel(drift_ppm=100.0)


@pytest.fixture
def jitter_timing_model():
    """Timing model with jitter."""
    return TimingModel(jitter_sigma_s=1e-9)


@pytest.fixture
def tdc_timing_model():
    """Timing model with TDC quantization."""
    return TimingModel(tdc_seconds=1e-12)


# ============================================================================
# TIME TAG FIXTURES
# ============================================================================

@pytest.fixture
def simple_time_tags():
    """Simple time tags for testing."""
    return TimeTags(
        times=np.array([1.0, 2.0, 3.0], dtype=float),
        is_signal=np.ones(3, dtype=bool),
        basis=np.zeros(3, dtype=np.int8),
        bit=np.zeros(3, dtype=np.int8),
    )


@pytest.fixture
def empty_time_tags():
    """Empty time tags."""
    return TimeTags(
        times=np.array([], dtype=float),
        is_signal=np.array([], dtype=bool),
        basis=np.array([], dtype=np.int8),
        bit=np.array([], dtype=np.int8),
    )


@pytest.fixture
def mixed_signal_tags():
    """Time tags with signal and background events."""
    return TimeTags(
        times=np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        is_signal=np.array([True, True, False, False]),
        basis=np.array([0, 1, 0, 1], dtype=np.int8),
        bit=np.array([0, 1, 1, 0], dtype=np.int8),
    )


# ============================================================================
# LINK BUDGET FIXTURES
# ============================================================================

@pytest.fixture
def default_sat_link_params():
    """Default satellite link parameters."""
    return SatLinkParams()


@pytest.fixture
def leo_sat_link_params():
    """LEO-specific satellite link parameters."""
    return SatLinkParams(
        wavelength_m=850e-9,
        altitude_m=550e3,
        atmospheric_loss_db_zenith=0.5,
        atmospheric_loss_db_horizon=8.0,
    )


@pytest.fixture
def vsat_link_params():
    """VSAT-style link parameters."""
    return SatLinkParams(
        wavelength_m=1550e-9,
        altitude_m=400e3,
        atmospheric_loss_db_zenith=1.0,
    )


# ============================================================================
# SECURITY ANALYSIS FIXTURES
# ============================================================================

@pytest.fixture
def default_finite_key_params():
    """Default finite-key parameters."""
    return FiniteKeyParams()


@pytest.fixture
def tight_finite_key_params():
    """Tight finite-key parameters (high confidence)."""
    return FiniteKeyParams(eps_pe=1e-15, eps_sec=1e-15)


@pytest.fixture
def loose_finite_key_params():
    """Loose finite-key parameters (lower confidence)."""
    return FiniteKeyParams(eps_pe=1e-5, eps_sec=1e-5)


@pytest.fixture
def default_decoy_params():
    """Default decoy-state parameters."""
    return DecoyParams()


@pytest.fixture
def custom_decoy_params():
    """Custom decoy-state parameters."""
    return DecoyParams(mu_s=0.8, mu_d=0.15, p_s=0.7, p_d=0.2, p_v=0.1)


# ============================================================================
# EXPERIMENT FIXTURES
# ============================================================================

@pytest.fixture
def default_experiment_params():
    """Default experiment parameters."""
    return ExperimentParams()


@pytest.fixture
def short_experiment_params():
    """Short experiment (few blocks)."""
    return ExperimentParams(n_blocks=5, block_seconds=10.0)


@pytest.fixture
def long_experiment_params():
    """Long experiment (many blocks)."""
    return ExperimentParams(n_blocks=100, block_seconds=30.0)


@pytest.fixture
def high_noise_experiment_params():
    """Experiment with high QBER noise."""
    return ExperimentParams(base_qber=0.08, qber_jitter=0.01)


# ============================================================================
# NUMERICAL PARAMETER FIXTURES
# ============================================================================

@pytest.fixture
def typical_loss_values():
    """Typical channel loss values (dB)."""
    return [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]


@pytest.fixture
def typical_qber_values():
    """Typical QBER values."""
    return [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]


@pytest.fixture
def typical_elevation_angles():
    """Typical satellite elevation angles (degrees)."""
    return [5, 15, 30, 45, 60, 75, 90]


@pytest.fixture
def extreme_parameters():
    """Extreme parameter combinations for stress testing."""
    return {
        "zero": 0.0,
        "tiny": 1e-12,
        "small": 1e-6,
        "large": 1e6,
        "huge": 1e12,
    }


# ============================================================================
# PARAMETRIZATION HELPERS
# ============================================================================

def pytest_generate_tests(metafunc):
    """Generate parametrized tests for common scenarios."""
    # Can be extended to auto-parametrize tests based on fixtures
    pass
