"""CV-QKD (Continuous-Variable Quantum Key Distribution) scaffold module."""

from .gg02 import (
    GG02Params,
    GG02Result,
    compute_snr,
    compute_mutual_information,
    compute_holevo_bound,
)

__all__ = [
    "GG02Params",
    "GG02Result",
    "compute_snr",
    "compute_mutual_information",
    "compute_holevo_bound",
]
