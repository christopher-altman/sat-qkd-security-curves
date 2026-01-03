"""
Optional calibration hooks for adjusting simulated outputs with empirical data.

This module allows users to apply real hardware/system calibration to simulated
curves without changing default behavior. Calibration is opt-in via --calibration-file.

Supported calibration methods:
1. Affine transformation: metric_calibrated = metric * scale + offset
2. Piecewise lookup table: linear interpolation from empirical data points

Example calibration JSON:
{
  "qber_mean": {
    "method": "affine",
    "scale": 1.05,
    "offset": 0.002
  },
  "key_rate_per_pulse": {
    "method": "piecewise",
    "loss_db": [0, 10, 20, 30],
    "values": [0.5, 0.4, 0.2, 0.05]
  }
}
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np


@dataclass
class CalibrationSpec:
    """Specification for calibrating a single metric.

    Attributes:
        method: "affine" or "piecewise"
        scale: Multiplicative factor for affine method (default 1.0)
        offset: Additive offset for affine method (default 0.0)
        loss_db: Loss values (dB) for piecewise lookup (required for piecewise)
        values: Calibrated values at each loss_db point (required for piecewise)
    """
    method: str
    scale: float = 1.0
    offset: float = 0.0
    loss_db: Optional[List[float]] = None
    values: Optional[List[float]] = None

    def __post_init__(self):
        """Validate calibration spec."""
        if self.method not in ("affine", "piecewise"):
            raise ValueError(f"Unknown calibration method: {self.method}")

        if self.method == "piecewise":
            if self.loss_db is None or self.values is None:
                raise ValueError("Piecewise method requires 'loss_db' and 'values'")
            if len(self.loss_db) != len(self.values):
                raise ValueError("loss_db and values must have same length")
            if len(self.loss_db) < 2:
                raise ValueError("Piecewise method requires at least 2 data points")
            # Ensure loss_db is sorted
            if self.loss_db != sorted(self.loss_db):
                raise ValueError("loss_db must be sorted in ascending order")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationSpec":
        """Create CalibrationSpec from dictionary."""
        return cls(
            method=d["method"],
            scale=d.get("scale", 1.0),
            offset=d.get("offset", 0.0),
            loss_db=d.get("loss_db"),
            values=d.get("values"),
        )


@dataclass
class CalibrationModel:
    """Model for applying calibrations to sweep outputs.

    Attributes:
        calibrations: Dict mapping metric name to CalibrationSpec
        metadata: Original JSON metadata for reporting
    """
    calibrations: Dict[str, CalibrationSpec] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> "CalibrationModel":
        """Load calibration model from JSON file.

        Args:
            path: Path to calibration JSON file

        Returns:
            CalibrationModel instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid or specs are malformed
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        calibrations = {}
        for metric_name, spec_dict in data.items():
            calibrations[metric_name] = CalibrationSpec.from_dict(spec_dict)

        return cls(calibrations=calibrations, metadata=data)

    def apply(
        self,
        metric_name: str,
        value: float,
        loss_db: Optional[float] = None,
    ) -> float:
        """Apply calibration to a metric value.

        Args:
            metric_name: Name of metric (e.g., "qber_mean", "key_rate_per_pulse")
            value: Uncalibrated value from simulation
            loss_db: Current loss in dB (required for piecewise method)

        Returns:
            Calibrated value

        Raises:
            ValueError: If piecewise method requires loss_db but it's not provided
        """
        if metric_name not in self.calibrations:
            # No calibration for this metric, return unchanged
            return value

        spec = self.calibrations[metric_name]

        if spec.method == "affine":
            return value * spec.scale + spec.offset

        elif spec.method == "piecewise":
            if loss_db is None:
                raise ValueError(
                    f"Piecewise calibration for '{metric_name}' requires loss_db"
                )

            # Linear interpolation
            calibrated = np.interp(
                loss_db,
                spec.loss_db,
                spec.values,
            )
            return float(calibrated)

        # Should never reach here due to validation in CalibrationSpec
        raise ValueError(f"Unknown calibration method: {spec.method}")

    def apply_to_record(
        self,
        record: Dict[str, Any],
        loss_db: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Apply all applicable calibrations to a sweep record.

        Args:
            record: Sweep output record (contains metrics like qber_mean, etc.)
            loss_db: Current loss in dB (extracted from record if not provided)

        Returns:
            New record with calibrated values (original record unchanged)
        """
        # Extract loss_db from record if not provided
        if loss_db is None:
            loss_db = record.get("loss_db")

        # Create a copy to avoid modifying original
        calibrated = record.copy()

        # Apply calibrations to all matching metrics
        for metric_name in self.calibrations.keys():
            if metric_name in calibrated:
                original_value = calibrated[metric_name]
                # Skip if value is not numeric (e.g., "inf", None)
                if isinstance(original_value, (int, float)) and not np.isnan(original_value):
                    calibrated[metric_name] = self.apply(
                        metric_name,
                        original_value,
                        loss_db=loss_db,
                    )

        return calibrated

    def get_metadata(self) -> Dict[str, Any]:
        """Get calibration metadata for JSON reports.

        Returns:
            Dictionary with calibration info suitable for JSON serialization
        """
        return {
            "calibration_applied": True,
            "calibrated_metrics": list(self.calibrations.keys()),
            "calibration_specs": self.metadata,
        }
