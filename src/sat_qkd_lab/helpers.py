from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any

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
