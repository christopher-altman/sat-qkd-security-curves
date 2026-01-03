"""
Blinded window generation utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import random

from .experiment import generate_schedule


@dataclass(frozen=True)
class WindowSpec:
    window_id: str
    t_start_seconds: float
    t_end_seconds: float


def generate_windows(n_blocks: int, block_seconds: float, seed: int) -> List[WindowSpec]:
    """Generate sequential windows for a run."""
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive.")
    if block_seconds <= 0:
        raise ValueError("block_seconds must be positive.")
    windows = []
    for idx in range(n_blocks):
        start = idx * block_seconds
        end = (idx + 1) * block_seconds
        windows.append(WindowSpec(
            window_id=f"W{idx:03d}",
            t_start_seconds=float(start),
            t_end_seconds=float(end),
        ))
    return windows


def assign_groups_blinded(windows: List[WindowSpec], seed: int) -> Dict[str, Dict[str, str]]:
    """
    Assign control/intervention labels with blinding.

    Returns mapping with blinded schedule entries and labels per window.
    """
    labels = generate_schedule(len(windows), seed)
    rng = random.Random(seed + 1)
    codes = [f"blk-{rng.getrandbits(32):08x}" for _ in windows]

    schedule_blinded = [
        {
            "window_id": window.window_id,
            "assignment_code": code,
        }
        for window, code in zip(windows, codes)
    ]
    labels_by_window = {
        window.window_id: label
        for window, label in zip(windows, labels)
    }
    schedule_unblinded = [
        {
            "window_id": window.window_id,
            "assignment_code": code,
            "label": label,
        }
        for window, code, label in zip(windows, codes, labels)
    ]

    return {
        "labels_by_window": labels_by_window,
        "schedule_blinded": schedule_blinded,
        "schedule_unblinded": schedule_unblinded,
    }
