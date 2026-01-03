"""
Telemetry ingestion utilities for calibration fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import csv
import json
from pathlib import Path


@dataclass(frozen=True)
class TelemetryRecord:
    loss_db: float
    qber_mean: float
    n_sent: Optional[int] = None
    timestamp_utc: Optional[str] = None
    coincidence_histogram: Optional[List[float]] = None
    coincidence_bin_seconds: Optional[float] = None
    car_series: Optional[List[float]] = None
    lock_state_series: Optional[List[int]] = None
    background_rate: Optional[float] = None
    off_window_counts: Optional[float] = None
    off_window_seconds: Optional[float] = None
    transmittance_series: Optional[List[float]] = None
    meta: Optional[Dict[str, Any]] = None


def _maybe_parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _parse_record(row: Dict[str, Any]) -> TelemetryRecord:
    loss_db = float(row.get("loss_db"))
    qber = float(row.get("qber_mean"))
    n_sent = row.get("n_sent")
    timestamp = row.get("timestamp_utc") or row.get("timestamp")
    coincidence_histogram = _maybe_parse_json(row.get("coincidence_histogram"))
    car_series = _maybe_parse_json(row.get("car_series"))
    lock_state_series = _maybe_parse_json(row.get("lock_state_series"))
    transmittance_series = _maybe_parse_json(row.get("transmittance_series"))
    coincidence_bin = row.get("coincidence_bin_seconds") or row.get("coincidence_bin_s")
    background_rate = row.get("background_rate")
    off_window_counts = row.get("off_window_counts")
    off_window_seconds = row.get("off_window_seconds")
    return TelemetryRecord(
        loss_db=loss_db,
        qber_mean=qber,
        n_sent=int(n_sent) if n_sent is not None else None,
        timestamp_utc=timestamp,
        coincidence_histogram=(
            [float(v) for v in coincidence_histogram]
            if isinstance(coincidence_histogram, list) else None
        ),
        coincidence_bin_seconds=float(coincidence_bin) if coincidence_bin is not None else None,
        car_series=(
            [float(v) for v in car_series] if isinstance(car_series, list) else None
        ),
        lock_state_series=(
            [int(v) for v in lock_state_series]
            if isinstance(lock_state_series, list) else None
        ),
        background_rate=float(background_rate) if background_rate is not None else None,
        off_window_counts=float(off_window_counts) if off_window_counts is not None else None,
        off_window_seconds=float(off_window_seconds) if off_window_seconds is not None else None,
        transmittance_series=(
            [float(v) for v in transmittance_series]
            if isinstance(transmittance_series, list) else None
        ),
        meta=row.get("meta"),
    )


def load_telemetry(path: str) -> List[TelemetryRecord]:
    """
    Load telemetry from JSON or CSV into canonical records.

    JSON expects a list or {"telemetry": [...]} with loss_db and qber_mean.
    CSV expects headers with loss_db and qber_mean columns.
    """
    telemetry_path = Path(path)
    if not telemetry_path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {path}")

    if telemetry_path.suffix.lower() == ".csv":
        with open(telemetry_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return [_parse_record(row) for row in reader]

    with open(telemetry_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        rows = data.get("telemetry")
        if rows is None:
            raise ValueError("JSON telemetry file must be a list or contain 'telemetry'.")
    else:
        rows = data

    if not isinstance(rows, list):
        raise ValueError("JSON telemetry file must contain a list of records.")

    return [_parse_record(row) for row in rows]
