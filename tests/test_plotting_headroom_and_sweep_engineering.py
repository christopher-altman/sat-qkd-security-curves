"""
Unit tests for QBER headroom plotting and sweep engineering outputs.

Tests cover:
1. plot_qber_headroom_vs_loss with single trial (no CI)
2. plot_qber_headroom_vs_loss with multiple trials and confidence intervals
3. plot_qber_headroom_vs_loss handling negative headroom cases
4. _run_sweep computing key_rate_bps and total_secret_bits
5. _run_sweep computing required_rep_rate_hz and handling impossible targets
"""
import pytest
import json
import numpy as np
from pathlib import Path
import tempfile
import os

from sat_qkd_lab.plotting import plot_qber_headroom_vs_loss
from sat_qkd_lab.run import main


class TestQberHeadroomPlotting:
    """Test QBER headroom plotting function."""

    def test_single_trial_no_ci(self, tmp_path):
        """Test plot_qber_headroom_vs_loss with single trial (no CI data)."""
        # Create mock records for a single trial
        records = [
            {"loss_db": 20.0, "qber": 0.03},
            {"loss_db": 30.0, "qber": 0.05},
            {"loss_db": 40.0, "qber": 0.08},
            {"loss_db": 50.0, "qber": 0.10},
        ]
        
        out_path = str(tmp_path / "headroom_single.png")
        result_path = plot_qber_headroom_vs_loss(
            records=records,
            out_path=out_path,
            qber_abort=0.11,
            show_ci=False,
        )
        
        # Verify plot was created
        assert os.path.exists(result_path)
        assert result_path == out_path
        
        # Verify file is not empty
        assert os.path.getsize(result_path) > 0

    def test_multiple_trials_with_ci(self, tmp_path):
        """Test plot_qber_headroom_vs_loss with multiple trials and confidence intervals."""
        # Create mock records with CI data
        records = [
            {
                "loss_db": 20.0,
                "qber_mean": 0.03,
                "qber_ci_low": 0.025,
                "qber_ci_high": 0.035,
            },
            {
                "loss_db": 30.0,
                "qber_mean": 0.05,
                "qber_ci_low": 0.045,
                "qber_ci_high": 0.055,
            },
            {
                "loss_db": 40.0,
                "qber_mean": 0.08,
                "qber_ci_low": 0.075,
                "qber_ci_high": 0.085,
            },
        ]
        
        out_path = str(tmp_path / "headroom_ci.png")
        result_path = plot_qber_headroom_vs_loss(
            records=records,
            out_path=out_path,
            qber_abort=0.11,
            show_ci=True,
        )
        
        # Verify plot was created
        assert os.path.exists(result_path)
        assert result_path == out_path
        
        # Verify file is not empty
        assert os.path.getsize(result_path) > 0

    def test_negative_headroom(self, tmp_path):
        """Test plot_qber_headroom_vs_loss handles cases where headroom is negative."""
        # Create records where QBER exceeds abort threshold
        records = [
            {"loss_db": 20.0, "qber": 0.05},
            {"loss_db": 30.0, "qber": 0.09},
            {"loss_db": 40.0, "qber": 0.12},  # Exceeds abort threshold of 0.11
            {"loss_db": 50.0, "qber": 0.15},  # Negative headroom
        ]
        
        out_path = str(tmp_path / "headroom_negative.png")
        result_path = plot_qber_headroom_vs_loss(
            records=records,
            out_path=out_path,
            qber_abort=0.11,
            show_ci=False,
        )
        
        # Verify plot was created (should handle negative values gracefully)
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0

    def test_negative_headroom_with_ci(self, tmp_path):
        """Test negative headroom with confidence intervals."""
        # Create records with CI data where some values exceed threshold
        records = [
            {
                "loss_db": 20.0,
                "qber_mean": 0.05,
                "qber_ci_low": 0.04,
                "qber_ci_high": 0.06,
            },
            {
                "loss_db": 40.0,
                "qber_mean": 0.12,  # Above threshold
                "qber_ci_low": 0.11,
                "qber_ci_high": 0.13,
            },
            {
                "loss_db": 50.0,
                "qber_mean": 0.18,  # Well above threshold
                "qber_ci_low": 0.17,
                "qber_ci_high": 0.19,
            },
        ]
        
        out_path = str(tmp_path / "headroom_negative_ci.png")
        result_path = plot_qber_headroom_vs_loss(
            records=records,
            out_path=out_path,
            qber_abort=0.11,
            show_ci=True,
        )
        
        # Verify plot was created
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0

    def test_custom_abort_threshold(self, tmp_path):
        """Test plot with custom abort threshold."""
        records = [
            {"loss_db": 20.0, "qber": 0.03},
            {"loss_db": 30.0, "qber": 0.07},
            {"loss_db": 40.0, "qber": 0.10},
        ]
        
        out_path = str(tmp_path / "headroom_custom_abort.png")
        result_path = plot_qber_headroom_vs_loss(
            records=records,
            out_path=out_path,
            qber_abort=0.08,  # Custom threshold
            show_ci=False,
        )
        
        # Verify plot was created
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0


class TestRunSweepEngineeringOutputs:
    """Test _run_sweep function for engineering outputs."""

    def test_sweep_with_rep_rate_and_pass_seconds(self, tmp_path, monkeypatch):
        """Test that _run_sweep computes key_rate_bps and total_secret_bits."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        # Prepare arguments for sweep command
        test_args = [
            "sat-qkd-security-curves",
            "sweep",
            "--loss-min", "20.0",
            "--loss-max", "30.0",
            "--steps", "3",
            "--pulses", "10000",  # Small for fast test
            "--rep-rate-hz", "1e6",  # 1 MHz
            "--pass-seconds", "100.0",  # 100 seconds
            "--outdir", str(tmp_path),
            "--seed", "42",
        ]
        
        # Mock sys.argv
        monkeypatch.setattr("sys.argv", test_args)
        
        # Run the sweep
        main()
        
        # Load the generated report
        report_path = tmp_path / "reports" / "latest.json"
        assert report_path.exists(), "Report file should be generated"
        
        with open(report_path) as f:
            report = json.load(f)
        
        # Verify report structure
        assert "loss_sweep" in report
        assert "no_attack" in report["loss_sweep"]
        
        # Check that engineering outputs are present in records
        no_attack_records = report["loss_sweep"]["no_attack"]
        assert len(no_attack_records) > 0
        
        for record in no_attack_records:
            # Verify key_rate_bps is computed
            assert "key_rate_bps" in record, "key_rate_bps should be in record"
            assert isinstance(record["key_rate_bps"], (int, float))
            
            # Verify total_secret_bits is computed
            assert "total_secret_bits" in record, "total_secret_bits should be in record"
            assert isinstance(record["total_secret_bits"], (int, float))
            
            # Verify mathematical relationship
            # total_secret_bits = key_rate_bps * pass_seconds
            expected_total = record["key_rate_bps"] * 100.0
            assert record["total_secret_bits"] == pytest.approx(expected_total, rel=1e-6)
            
            # Verify key_rate_bps = key_rate_per_pulse * rep_rate_hz
            expected_bps = record["key_rate_per_pulse"] * 1e6
            assert record["key_rate_bps"] == pytest.approx(expected_bps, rel=1e-6)

    def test_sweep_with_target_bits_and_pass_seconds(self, tmp_path, monkeypatch):
        """Test that _run_sweep computes required_rep_rate_hz."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        # Prepare arguments for sweep command
        test_args = [
            "sat-qkd-security-curves",
            "sweep",
            "--loss-min", "20.0",
            "--loss-max", "25.0",
            "--steps", "2",
            "--pulses", "10000",
            "--target-bits", "1000000",  # Target 1 Mbits
            "--pass-seconds", "100.0",
            "--outdir", str(tmp_path),
            "--seed", "42",
        ]
        
        # Mock sys.argv
        monkeypatch.setattr("sys.argv", test_args)
        
        # Run the sweep
        main()
        
        # Load the generated report
        report_path = tmp_path / "reports" / "latest.json"
        assert report_path.exists()
        
        with open(report_path) as f:
            report = json.load(f)
        
        # Check that required_rep_rate_hz is present
        no_attack_records = report["loss_sweep"]["no_attack"]
        assert len(no_attack_records) > 0
        
        for record in no_attack_records:
            assert "required_rep_rate_hz" in record
            
            # Verify the calculation
            # required_rep_rate_hz = target_bits / (key_rate_per_pulse * pass_seconds)
            if record["key_rate_per_pulse"] > 0:
                expected_rep_rate = 1000000 / (record["key_rate_per_pulse"] * 100.0)
                assert record["required_rep_rate_hz"] == pytest.approx(expected_rep_rate, rel=1e-6)
            else:
                # If key rate is zero or negative, should be "inf"
                assert record["required_rep_rate_hz"] == "inf"

    def test_sweep_with_impossible_target(self, tmp_path, monkeypatch):
        """Test that _run_sweep handles impossible targets (zero key rate) correctly."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        # Use very high loss to get zero key rate
        test_args = [
            "sat-qkd-security-curves",
            "sweep",
            "--loss-min", "70.0",  # Very high loss
            "--loss-max", "80.0",
            "--steps", "2",
            "--pulses", "10000",
            "--target-bits", "1000000",
            "--pass-seconds", "100.0",
            "--outdir", str(tmp_path),
            "--seed", "42",
        ]
        
        # Mock sys.argv
        monkeypatch.setattr("sys.argv", test_args)
        
        # Run the sweep
        main()
        
        # Load the generated report
        report_path = tmp_path / "reports" / "latest.json"
        assert report_path.exists()
        
        with open(report_path) as f:
            report = json.load(f)
        
        # Check records
        no_attack_records = report["loss_sweep"]["no_attack"]
        
        # At high loss, key rates should be zero or aborted
        for record in no_attack_records:
            assert "required_rep_rate_hz" in record
            
            # If key rate is zero/negative or aborted, required rep rate should be "inf"
            if record["key_rate_per_pulse"] <= 0 or record.get("aborted", False):
                assert record["required_rep_rate_hz"] == "inf"
                assert "required_rep_rate_note" in record

    def test_sweep_parameters_recorded_in_report(self, tmp_path, monkeypatch):
        """Test that rep_rate_hz and target_bits are recorded in report parameters."""
        monkeypatch.chdir(tmp_path)
        
        test_args = [
            "sat-qkd-security-curves",
            "sweep",
            "--loss-min", "20.0",
            "--loss-max", "25.0",
            "--steps", "2",
            "--pulses", "10000",
            "--rep-rate-hz", "5e6",
            "--target-bits", "500000",
            "--pass-seconds", "50.0",
            "--outdir", str(tmp_path),
            "--seed", "42",
        ]
        
        monkeypatch.setattr("sys.argv", test_args)
        main()
        
        report_path = tmp_path / "reports" / "latest.json"
        with open(report_path) as f:
            report = json.load(f)
        
        # Verify parameters are recorded
        assert "parameters" in report
        assert "rep_rate_hz" in report["parameters"]
        assert report["parameters"]["rep_rate_hz"] == 5e6
        assert "target_bits" in report["parameters"]
        assert report["parameters"]["target_bits"] == 500000
        assert "pass_seconds" in report["parameters"]
        assert report["parameters"]["pass_seconds"] == 50.0

    def test_sweep_with_ci_computes_engineering_outputs(self, tmp_path, monkeypatch):
        """Test that sweep with multiple trials (CI mode) also computes engineering outputs."""
        monkeypatch.chdir(tmp_path)
        
        test_args = [
            "sat-qkd-security-curves",
            "sweep",
            "--loss-min", "20.0",
            "--loss-max", "25.0",
            "--steps", "2",
            "--pulses", "5000",
            "--trials", "3",  # Enable CI mode
            "--rep-rate-hz", "1e6",
            "--pass-seconds", "100.0",
            "--outdir", str(tmp_path),
            "--seed", "42",
        ]
        
        monkeypatch.setattr("sys.argv", test_args)
        main()
        
        report_path = tmp_path / "reports" / "latest.json"
        with open(report_path) as f:
            report = json.load(f)
        
        # In CI mode, the key is "loss_sweep_ci"
        assert "loss_sweep_ci" in report
        no_attack_records = report["loss_sweep_ci"]["no_attack"]
        
        for record in no_attack_records:
            # Verify engineering outputs are computed
            assert "key_rate_bps" in record
            assert "total_secret_bits" in record
            
            # In CI mode, key_rate_per_pulse is replaced by key_rate_per_pulse_mean
            expected_bps = record["key_rate_per_pulse_mean"] * 1e6
            assert record["key_rate_bps"] == pytest.approx(expected_bps, rel=1e-6)

    def test_sweep_without_engineering_params_no_extra_outputs(self, tmp_path, monkeypatch):
        """Test that sweep without engineering params doesn't add extra outputs."""
        monkeypatch.chdir(tmp_path)
        
        test_args = [
            "sat-qkd-security-curves",
            "sweep",
            "--loss-min", "20.0",
            "--loss-max", "25.0",
            "--steps", "2",
            "--pulses", "5000",
            "--outdir", str(tmp_path),
            "--seed", "42",
        ]
        
        monkeypatch.setattr("sys.argv", test_args)
        main()
        
        report_path = tmp_path / "reports" / "latest.json"
        with open(report_path) as f:
            report = json.load(f)
        
        no_attack_records = report["loss_sweep"]["no_attack"]
        
        for record in no_attack_records:
            # These fields should NOT be present
            assert "key_rate_bps" not in record
            assert "total_secret_bits" not in record
            assert "required_rep_rate_hz" not in record
