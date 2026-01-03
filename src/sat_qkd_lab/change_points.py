"""
Change-point detection and anomaly attribution utilities.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence
import numpy as np


def detect_change_points(
    time_series: Dict[str, Sequence[float]],
    metrics: Iterable[str],
    z_threshold: float = 3.5,
    min_delta: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Detect change points based on robust z-scores of first differences.
    """
    incidents: List[Dict[str, Any]] = []
    t_vals = time_series.get("t_seconds")
    for metric in metrics:
        values = time_series.get(metric)
        if values is None or len(values) < 2:
            continue
        diffs = np.diff(np.array(values, dtype=float))
        median = float(np.median(diffs))
        mad = float(np.median(np.abs(diffs - median)))
        scale = mad * 1.4826 if mad > 0 else float(np.std(diffs))
        if scale <= 0:
            continue
        for idx, diff in enumerate(diffs, start=1):
            z = (diff - median) / scale
            if abs(z) < z_threshold or abs(diff) < min_delta:
                continue
            incidents.append({
                "index": int(idx),
                "t_seconds": float(t_vals[idx]) if t_vals is not None else None,
                "metric": metric,
                "delta": float(diff),
                "z_score": float(z),
                "value_before": float(values[idx - 1]),
                "value_after": float(values[idx]),
            })
    return incidents


def attribute_incidents(
    time_series: Dict[str, Sequence[float]],
    incidents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Apply heuristic attribution labels to incidents.
    """
    loss_db = time_series.get("loss_db")
    qber = time_series.get("qber_mean")
    key_rate = time_series.get("key_rate_per_pulse")
    lock_state = time_series.get("pointing_locked")

    for incident in incidents:
        idx = incident["index"]
        attribution = "unattributed"
        evidence: Dict[str, Any] = {}

        if lock_state is not None and idx < len(lock_state):
            if idx > 0 and bool(lock_state[idx - 1]) and not bool(lock_state[idx]):
                attribution = "pointing_dropout"
                evidence["pointing_locked"] = [bool(lock_state[idx - 1]), bool(lock_state[idx])]
        if attribution == "unattributed" and qber is not None and loss_db is not None:
            if idx > 0:
                loss_delta = float(loss_db[idx] - loss_db[idx - 1])
                qber_delta = float(qber[idx] - qber[idx - 1])
                evidence["loss_delta_db"] = loss_delta
                evidence["qber_delta"] = qber_delta
                if abs(loss_delta) < 0.2 and qber_delta > 0.01:
                    if key_rate is not None and idx < len(key_rate):
                        kr_delta = float(key_rate[idx] - key_rate[idx - 1])
                        evidence["key_rate_delta"] = kr_delta
                        if kr_delta < 0:
                            attribution = "background_burst"
                        else:
                            attribution = "clock_drift"
                    else:
                        attribution = "clock_drift"
        incident["attribution"] = attribution
        if evidence:
            incident["evidence"] = evidence
    return incidents
