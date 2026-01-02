from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union


# --- Input Validation Helpers ---

def validate_int(
    name: str,
    value: Union[int, float],
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    """
    Validate that value is an integer within optional bounds.

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    value : int or float
        Value to validate.
    min_value : int, optional
        Minimum allowed value (inclusive).
    max_value : int, optional
        Maximum allowed value (inclusive).

    Returns
    -------
    int
        The validated integer value.

    Raises
    ------
    ValueError
        If value is not a valid integer or out of bounds.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name}: expected int, got {type(value).__name__} ({value!r})")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name}: expected int, got float {value}")
    int_val = int(value)
    if min_value is not None and int_val < min_value:
        raise ValueError(f"{name}: {int_val} < minimum {min_value}")
    if max_value is not None and int_val > max_value:
        raise ValueError(f"{name}: {int_val} > maximum {max_value}")
    return int_val


def validate_float(
    name: str,
    value: Union[int, float],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> float:
    """
    Validate that value is a float within optional bounds.

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    value : int or float
        Value to validate.
    min_value : float, optional
        Minimum allowed value (inclusive).
    max_value : float, optional
        Maximum allowed value (inclusive).
    allow_nan : bool
        Whether to allow NaN values.
    allow_inf : bool
        Whether to allow infinite values.

    Returns
    -------
    float
        The validated float value.

    Raises
    ------
    ValueError
        If value is not a valid float or out of bounds.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name}: expected float, got {type(value).__name__} ({value!r})")
    float_val = float(value)
    if math.isnan(float_val) and not allow_nan:
        raise ValueError(f"{name}: NaN not allowed")
    if math.isinf(float_val) and not allow_inf:
        raise ValueError(f"{name}: infinity not allowed")
    if min_value is not None and float_val < min_value:
        raise ValueError(f"{name}: {float_val} < minimum {min_value}")
    if max_value is not None and float_val > max_value:
        raise ValueError(f"{name}: {float_val} > maximum {max_value}")
    return float_val


def validate_seed(seed: Optional[int]) -> Optional[int]:
    """
    Validate that seed is None or a valid integer for numpy RNG.

    Parameters
    ----------
    seed : int or None
        Random seed value.

    Returns
    -------
    int or None
        The validated seed.

    Raises
    ------
    ValueError
        If seed is not None or a valid non-negative integer.
    """
    if seed is None:
        return None
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed: expected int or None, got {type(seed).__name__} ({seed!r})")
    if seed < 0:
        raise ValueError(f"seed: {seed} < minimum 0")
    # numpy accepts seeds up to 2^32 - 1 for legacy RandomState,
    # but default_rng accepts larger values. We'll allow non-negative ints.
    return seed


# --- Binary Entropy ---

def h2(p: float) -> float:
    """Binary entropy in bits. Defined as 0 at p=0 or p=1."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

@dataclass(frozen=True)
class RunSummary:
    n_sent: int
    n_received: int
    n_sifted: int
    qber: float
    secret_fraction: float
    n_secret_est: int
    aborted: bool
    meta: Dict[str, Any]

    @property
    def key_rate_per_pulse(self) -> float:
        """
        Asymptotic key rate per sent pulse.

        This combines:
        - Detection probability (n_received / n_sent)
        - Sifting probability (n_sifted / n_received) ≈ 1/2 for BB84
        - Secret fraction (bits of key per sifted bit)

        Returns 0.0 if aborted or no pulses sent.
        """
        if self.aborted or self.n_sent == 0:
            return 0.0
        return (self.n_sifted / self.n_sent) * self.secret_fraction
