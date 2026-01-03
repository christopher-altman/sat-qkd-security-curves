"""
Forecast ingestion utilities for MPI-style forecast harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import csv
import json
from pathlib import Path


ALLOWED_OPERATORS = {">=", "<=", "==", "increases", "decreases"}


@dataclass(frozen=True)
class Forecast:
    forecast_id: str
    timestamp_utc: Optional[str]
    window_id: str
    metric_name: str
    operator: str
    value: Optional[float]
    notes: Optional[str] = None


def _parse_forecast_row(row: Dict[str, Any]) -> Forecast:
    forecast_id = row.get("forecast_id") or row.get("id") or row.get("forecast")
    if forecast_id is None:
        raise ValueError("Forecast missing id/forecast_id field.")

    metric = row.get("metric_name") or row.get("metric")
    if metric is None:
        raise ValueError(f"Forecast {forecast_id} missing metric_name.")

    operator = row.get("operator")
    if operator not in ALLOWED_OPERATORS:
        raise ValueError(f"Forecast {forecast_id} has unsupported operator: {operator}")

    value_raw = row.get("value")
    if operator in (">=", "<=", "=="):
        if value_raw is None:
            raise ValueError(f"Forecast {forecast_id} requires a numeric value.")
        value = float(value_raw)
    else:
        value = None

    return Forecast(
        forecast_id=str(forecast_id),
        timestamp_utc=row.get("timestamp_utc") or row.get("timestamp"),
        window_id=str(row.get("window_id")),
        metric_name=str(metric),
        operator=str(operator),
        value=value,
        notes=row.get("notes"),
    )


def load_forecasts(path: str) -> List[Forecast]:
    """
    Load forecasts from JSON or CSV.

    JSON expects a list of forecast objects or {"forecasts": [...]}.
    CSV expects headers matching Forecast fields.
    """
    forecast_path = Path(path)
    if not forecast_path.exists():
        raise FileNotFoundError(f"Forecast file not found: {path}")

    if forecast_path.suffix.lower() == ".csv":
        with open(forecast_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return [_parse_forecast_row(row) for row in reader]

    with open(forecast_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        rows = data.get("forecasts")
        if rows is None:
            raise ValueError("JSON forecast file must be a list or contain 'forecasts'.")
    else:
        rows = data

    if not isinstance(rows, list):
        raise ValueError("JSON forecast file must contain a list of forecasts.")

    return [_parse_forecast_row(row) for row in rows]
