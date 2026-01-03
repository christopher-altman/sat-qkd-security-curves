"""
Tests for calibration hooks functionality.
"""

import json
import math
import pytest
import tempfile
from pathlib import Path
from sat_qkd_lab.calibration import CalibrationSpec, CalibrationModel


class TestCalibrationSpec:
    """Test CalibrationSpec validation and creation."""

    def test_affine_spec_valid(self):
        """Test valid affine calibration spec."""
        spec = CalibrationSpec(method="affine", scale=1.05, offset=0.002)
        assert spec.method == "affine"
        assert spec.scale == 1.05
        assert spec.offset == 0.002

    def test_affine_spec_defaults(self):
        """Test affine spec with default scale and offset."""
        spec = CalibrationSpec(method="affine")
        assert spec.scale == 1.0
        assert spec.offset == 0.0

    def test_piecewise_spec_valid(self):
        """Test valid piecewise calibration spec."""
        spec = CalibrationSpec(
            method="piecewise",
            loss_db=[0.0, 10.0, 20.0],
            values=[0.5, 0.4, 0.2],
        )
        assert spec.method == "piecewise"
        assert spec.loss_db == [0.0, 10.0, 20.0]
        assert spec.values == [0.5, 0.4, 0.2]

    def test_piecewise_requires_loss_db(self):
        """Test that piecewise method requires loss_db."""
        with pytest.raises(ValueError, match="requires 'loss_db' and 'values'"):
            CalibrationSpec(method="piecewise", values=[0.5, 0.4])

    def test_piecewise_requires_values(self):
        """Test that piecewise method requires values."""
        with pytest.raises(ValueError, match="requires 'loss_db' and 'values'"):
            CalibrationSpec(method="piecewise", loss_db=[0.0, 10.0])

    def test_piecewise_length_mismatch(self):
        """Test that loss_db and values must have same length."""
        with pytest.raises(ValueError, match="must have same length"):
            CalibrationSpec(
                method="piecewise",
                loss_db=[0.0, 10.0, 20.0],
                values=[0.5, 0.4],
            )

    def test_piecewise_minimum_points(self):
        """Test that piecewise requires at least 2 points."""
        with pytest.raises(ValueError, match="at least 2 data points"):
            CalibrationSpec(
                method="piecewise",
                loss_db=[10.0],
                values=[0.5],
            )

    def test_piecewise_unsorted_loss_db(self):
        """Test that loss_db must be sorted."""
        with pytest.raises(ValueError, match="must be sorted"):
            CalibrationSpec(
                method="piecewise",
                loss_db=[20.0, 10.0, 0.0],
                values=[0.2, 0.4, 0.5],
            )

    def test_unknown_method(self):
        """Test that unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown calibration method"):
            CalibrationSpec(method="unknown")

    def test_from_dict_affine(self):
        """Test creating CalibrationSpec from dict (affine)."""
        d = {"method": "affine", "scale": 1.1, "offset": 0.01}
        spec = CalibrationSpec.from_dict(d)
        assert spec.method == "affine"
        assert spec.scale == 1.1
        assert spec.offset == 0.01

    def test_from_dict_piecewise(self):
        """Test creating CalibrationSpec from dict (piecewise)."""
        d = {
            "method": "piecewise",
            "loss_db": [0.0, 10.0, 20.0],
            "values": [0.5, 0.4, 0.2],
        }
        spec = CalibrationSpec.from_dict(d)
        assert spec.method == "piecewise"
        assert spec.loss_db == [0.0, 10.0, 20.0]
        assert spec.values == [0.5, 0.4, 0.2]


class TestCalibrationModel:
    """Test CalibrationModel functionality."""

    def test_from_file_affine(self):
        """Test loading calibration model from file (affine)."""
        data = {
            "qber_mean": {"method": "affine", "scale": 1.05, "offset": 0.002}
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            model = CalibrationModel.from_file(path)
            assert "qber_mean" in model.calibrations
            assert model.calibrations["qber_mean"].method == "affine"
            assert model.calibrations["qber_mean"].scale == 1.05
            assert model.calibrations["qber_mean"].offset == 0.002
        finally:
            Path(path).unlink()

    def test_from_file_piecewise(self):
        """Test loading calibration model from file (piecewise)."""
        data = {
            "key_rate_per_pulse": {
                "method": "piecewise",
                "loss_db": [0.0, 10.0, 20.0, 30.0],
                "values": [0.5, 0.4, 0.2, 0.05],
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            model = CalibrationModel.from_file(path)
            assert "key_rate_per_pulse" in model.calibrations
            assert model.calibrations["key_rate_per_pulse"].method == "piecewise"
            assert model.calibrations["key_rate_per_pulse"].loss_db == [0.0, 10.0, 20.0, 30.0]
            assert model.calibrations["key_rate_per_pulse"].values == [0.5, 0.4, 0.2, 0.05]
        finally:
            Path(path).unlink()

    def test_from_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            CalibrationModel.from_file("/nonexistent/file.json")

    def test_apply_affine(self):
        """Test applying affine calibration."""
        model = CalibrationModel(
            calibrations={
                "qber_mean": CalibrationSpec(method="affine", scale=1.05, offset=0.002)
            }
        )
        result = model.apply("qber_mean", 0.05)
        # 0.05 * 1.05 + 0.002 = 0.0525 + 0.002 = 0.0545
        assert result == pytest.approx(0.0545)

    def test_apply_affine_scale_only(self):
        """Test applying affine with scale only."""
        model = CalibrationModel(
            calibrations={
                "key_rate_per_pulse": CalibrationSpec(method="affine", scale=0.9, offset=0.0)
            }
        )
        result = model.apply("key_rate_per_pulse", 0.5)
        assert result == pytest.approx(0.45)

    def test_apply_affine_offset_only(self):
        """Test applying affine with offset only."""
        model = CalibrationModel(
            calibrations={
                "qber_mean": CalibrationSpec(method="affine", scale=1.0, offset=0.01)
            }
        )
        result = model.apply("qber_mean", 0.05)
        assert result == pytest.approx(0.06)

    def test_apply_piecewise_exact_point(self):
        """Test piecewise calibration at exact data point."""
        model = CalibrationModel(
            calibrations={
                "key_rate_per_pulse": CalibrationSpec(
                    method="piecewise",
                    loss_db=[0.0, 10.0, 20.0, 30.0],
                    values=[0.5, 0.4, 0.2, 0.05],
                )
            }
        )
        result = model.apply("key_rate_per_pulse", 0.3, loss_db=10.0)
        # Should return exact value at loss_db=10.0
        assert result == pytest.approx(0.4)

    def test_apply_piecewise_interpolation(self):
        """Test piecewise calibration with linear interpolation."""
        model = CalibrationModel(
            calibrations={
                "key_rate_per_pulse": CalibrationSpec(
                    method="piecewise",
                    loss_db=[0.0, 10.0, 20.0],
                    values=[0.5, 0.4, 0.2],
                )
            }
        )
        result = model.apply("key_rate_per_pulse", 0.0, loss_db=5.0)
        # Linear interpolation between (0.0, 0.5) and (10.0, 0.4)
        # At loss_db=5.0: 0.5 + (0.4 - 0.5) * (5.0 / 10.0) = 0.5 - 0.05 = 0.45
        assert result == pytest.approx(0.45)

    def test_apply_piecewise_extrapolation_low(self):
        """Test piecewise calibration clamps below range."""
        model = CalibrationModel(
            calibrations={
                "key_rate_per_pulse": CalibrationSpec(
                    method="piecewise",
                    loss_db=[10.0, 20.0, 30.0],
                    values=[0.4, 0.2, 0.05],
                )
            }
        )
        result = model.apply("key_rate_per_pulse", 0.0, loss_db=5.0)
        # np.interp clamps to boundary value (first point)
        assert result == pytest.approx(0.4)

    def test_apply_piecewise_extrapolation_high(self):
        """Test piecewise calibration clamps above range."""
        model = CalibrationModel(
            calibrations={
                "key_rate_per_pulse": CalibrationSpec(
                    method="piecewise",
                    loss_db=[0.0, 10.0, 20.0],
                    values=[0.5, 0.4, 0.2],
                )
            }
        )
        result = model.apply("key_rate_per_pulse", 0.0, loss_db=30.0)
        # np.interp clamps to boundary value (last point)
        assert result == pytest.approx(0.2)

    def test_apply_piecewise_requires_loss_db(self):
        """Test that piecewise calibration requires loss_db parameter."""
        model = CalibrationModel(
            calibrations={
                "key_rate_per_pulse": CalibrationSpec(
                    method="piecewise",
                    loss_db=[0.0, 10.0, 20.0],
                    values=[0.5, 0.4, 0.2],
                )
            }
        )
        with pytest.raises(ValueError, match="requires loss_db"):
            model.apply("key_rate_per_pulse", 0.3)

    def test_apply_no_calibration_for_metric(self):
        """Test that uncalibrated metrics are returned unchanged."""
        model = CalibrationModel(
            calibrations={
                "qber_mean": CalibrationSpec(method="affine", scale=1.05, offset=0.002)
            }
        )
        result = model.apply("key_rate_per_pulse", 0.5)
        assert result == 0.5

    def test_apply_to_record_single_metric(self):
        """Test applying calibration to a record with single metric."""
        model = CalibrationModel(
            calibrations={
                "qber_mean": CalibrationSpec(method="affine", scale=1.05, offset=0.002)
            }
        )
        record = {"loss_db": 25.0, "qber_mean": 0.05, "key_rate_per_pulse": 0.3}
        calibrated = model.apply_to_record(record)

        # Original record should be unchanged
        assert record["qber_mean"] == 0.05

        # Calibrated record should have adjusted qber_mean
        assert calibrated["qber_mean"] == pytest.approx(0.0545)
        # Uncalibrated metrics should remain unchanged
        assert calibrated["key_rate_per_pulse"] == 0.3
        assert calibrated["loss_db"] == 25.0

    def test_apply_to_record_multiple_metrics(self):
        """Test applying calibration to a record with multiple metrics."""
        model = CalibrationModel(
            calibrations={
                "qber_mean": CalibrationSpec(method="affine", scale=1.05, offset=0.002),
                "key_rate_per_pulse": CalibrationSpec(method="affine", scale=0.9, offset=0.0),
            }
        )
        record = {"loss_db": 25.0, "qber_mean": 0.05, "key_rate_per_pulse": 0.3}
        calibrated = model.apply_to_record(record)

        assert calibrated["qber_mean"] == pytest.approx(0.0545)
        assert calibrated["key_rate_per_pulse"] == pytest.approx(0.27)

    def test_apply_to_record_piecewise(self):
        """Test applying piecewise calibration to a record."""
        model = CalibrationModel(
            calibrations={
                "key_rate_per_pulse": CalibrationSpec(
                    method="piecewise",
                    loss_db=[0.0, 10.0, 20.0, 30.0],
                    values=[0.5, 0.4, 0.2, 0.05],
                )
            }
        )
        record = {"loss_db": 15.0, "qber_mean": 0.05, "key_rate_per_pulse": 0.3}
        calibrated = model.apply_to_record(record)

        # Piecewise should interpolate at loss_db=15.0
        # Between (10.0, 0.4) and (20.0, 0.2): 0.4 + (0.2 - 0.4) * (5.0 / 10.0) = 0.3
        assert calibrated["key_rate_per_pulse"] == pytest.approx(0.3)

    def test_apply_to_record_extracts_loss_db(self):
        """Test that apply_to_record extracts loss_db from record."""
        model = CalibrationModel(
            calibrations={
                "key_rate_per_pulse": CalibrationSpec(
                    method="piecewise",
                    loss_db=[0.0, 10.0, 20.0],
                    values=[0.5, 0.4, 0.2],
                )
            }
        )
        record = {"loss_db": 10.0, "key_rate_per_pulse": 0.3}
        calibrated = model.apply_to_record(record)
        assert calibrated["key_rate_per_pulse"] == pytest.approx(0.4)

    def test_apply_to_record_skips_non_numeric(self):
        """Test that non-numeric values are skipped."""
        model = CalibrationModel(
            calibrations={
                "required_rep_rate_hz": CalibrationSpec(method="affine", scale=1.1, offset=0.0)
            }
        )
        record = {"loss_db": 25.0, "required_rep_rate_hz": "inf"}
        calibrated = model.apply_to_record(record)
        # Should remain "inf" (not calibrated)
        assert calibrated["required_rep_rate_hz"] == "inf"

    def test_apply_to_record_skips_nan(self):
        """Test that NaN values are skipped."""
        model = CalibrationModel(
            calibrations={
                "qber_mean": CalibrationSpec(method="affine", scale=1.05, offset=0.002)
            }
        )
        record = {"loss_db": 25.0, "qber_mean": float("nan")}
        calibrated = model.apply_to_record(record)
        # Should remain NaN (not calibrated)
        assert math.isnan(calibrated["qber_mean"])

    def test_get_metadata(self):
        """Test getting calibration metadata for JSON reports."""
        data = {
            "qber_mean": {"method": "affine", "scale": 1.05, "offset": 0.002},
            "key_rate_per_pulse": {
                "method": "piecewise",
                "loss_db": [0.0, 10.0, 20.0],
                "values": [0.5, 0.4, 0.2],
            },
        }
        model = CalibrationModel(
            calibrations={
                "qber_mean": CalibrationSpec.from_dict(data["qber_mean"]),
                "key_rate_per_pulse": CalibrationSpec.from_dict(data["key_rate_per_pulse"]),
            },
            metadata=data,
        )

        metadata = model.get_metadata()
        assert metadata["calibration_applied"] is True
        assert set(metadata["calibrated_metrics"]) == {"qber_mean", "key_rate_per_pulse"}
        assert metadata["calibration_specs"] == data

    def test_empty_calibration_model(self):
        """Test that empty calibration model works correctly."""
        model = CalibrationModel()
        result = model.apply("qber_mean", 0.05)
        assert result == 0.05

        record = {"qber_mean": 0.05, "key_rate_per_pulse": 0.3}
        calibrated = model.apply_to_record(record)
        assert calibrated == record
