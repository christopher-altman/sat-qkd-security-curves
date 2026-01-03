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
    meta: Optional[Dict[str, Any]] = None


def _parse_record(row: Dict[str, Any]) -> TelemetryRecord:
    loss_db = float(row.get("loss_db"))
    qber = float(row.get("qber_mean"))
    n_sent = row.get("n_sent")
    timestamp = row.get("timestamp_utc") or row.get("timestamp")
    return TelemetryRecord(
        loss_db=loss_db,
        qber_mean=qber,
        n_sent=int(n_sent) if n_sent is not None else None,
        timestamp_utc=timestamp,
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
