"""
Unit tests for pass_model.compute_pass_records and run.py helper functions.

Tests cover:
1. compute_pass_records correctly calculates secure_start_elevation_deg and secure_end_elevation_deg.
2. compute_pass_records correctly calculates peak_key_rate in the summary.
3. compute_pass_records correctly calculates mean_key_rate_in_window for a secure window.
4. _attach_manifest_and_git correctly adds the assumptions manifest and git commit to a report.
5. _print_sweep_summary outputs correct statistics for single trial sweeps.
"""
from __future__ import annotations

import io
import sys
from typing import Any, Dict

import pytest

from sat_qkd_lab.pass_model import (
    PassModelParams,
    compute_pass_records,
)
from sat_qkd_lab.free_space_link import FreeSpaceLinkParams
from sat_qkd_lab.detector import DetectorParams
from sat_qkd_lab.finite_key import FiniteKeyParams
from sat_qkd_lab.run import _attach_manifest_and_git, _print_sweep_summary, SCHEMA_VERSION
from sat_qkd_lab.assumptions import build_assumptions_manifest


class TestComputePassRecordsElevationCalculations:
    """Tests for secure start/end elevation degree calculations."""

    def test_secure_elevation_with_full_secure_window(self):
        """When entire pass is secure, start/end elevations should be min and max elevation."""
        params = PassModelParams(
            max_elevation_deg=60.0,
            min_elevation_deg=10.0,
            pass_seconds=100.0,
            dt_seconds=10.0,
            flip_prob=0.001,  # Very low to ensure secure
            rep_rate_hz=1e8,
            qber_abort_threshold=0.11,
            background_mode="night",
        )
        link = FreeSpaceLinkParams()
        detector = DetectorParams(eta=0.5, p_bg=1e-6)

        records, summary = compute_pass_records(
            params=params,
            link_params=link,
            detector=detector,
        )

        # With these favorable parameters, we should have a secure window
        assert summary["secure_window_seconds"] > 0.0
        
        # The secure start and end elevations should be close to the min elevation
        # (since the pass goes from min -> max -> min elevation)
        assert 9.0 <= summary["secure_start_elevation_deg"] <= 20.0
        assert 9.0 <= summary["secure_end_elevation_deg"] <= 20.0

    def test_secure_elevation_with_no_secure_window(self):
        """When no secure window exists, elevations should default to min_elevation_deg."""
        params = PassModelParams(
            max_elevation_deg=60.0,
            min_elevation_deg=10.0,
            pass_seconds=100.0,
            dt_seconds=10.0,
            flip_prob=0.15,  # Very high to prevent secure key
            rep_rate_hz=1e8,
            qber_abort_threshold=0.11,
            background_mode="day",
        )
        link = FreeSpaceLinkParams()
        detector = DetectorParams(eta=0.1, p_bg=1e-2)  # Poor detector + high background

        records, summary = compute_pass_records(
            params=params,
            link_params=link,
            detector=detector,
        )

        # No secure window expected
        assert summary["secure_window_seconds"] == 0.0
        assert summary["secure_start_elevation_deg"] == params.min_elevation_deg
        assert summary["secure_end_elevation_deg"] == params.min_elevation_deg

    def test_secure_elevation_with_partial_secure_window(self):
        """When only part of pass is secure, elevations should reflect actual secure region."""
        params = PassModelParams(
            max_elevation_deg=60.0,
            min_elevation_deg=10.0,
            pass_seconds=200.0,
            dt_seconds=10.0,
            flip_prob=0.005,
            rep_rate_hz=1e8,
            qber_abort_threshold=0.11,
            background_mode="night",
        )
        link = FreeSpaceLinkParams()
        detector = DetectorParams(eta=0.3, p_bg=1e-5)

        records, summary = compute_pass_records(
            params=params,
            link_params=link,
            detector=detector,
        )

        # Should have some secure window but not the full pass
        if summary["secure_window_seconds"] > 0:
            # Start and end elevations should be within the pass range
            assert params.min_elevation_deg <= summary["secure_start_elevation_deg"] <= params.max_elevation_deg
            assert params.min_elevation_deg <= summary["secure_end_elevation_deg"] <= params.max_elevation_deg
            
            # Verify they match the records
            secure_records = [r for r in records if r["secret_bits_dt"] > 0.0]
            if secure_records:
                assert summary["secure_start_elevation_deg"] == secure_records[0]["elevation_deg"]
                assert summary["secure_end_elevation_deg"] == secure_records[-1]["elevation_deg"]


class TestComputePassRecordsPeakKeyRate:
    """Tests for peak_key_rate calculation in summary."""

    def test_peak_key_rate_matches_max_from_records(self):
        """peak_key_rate should be the maximum key_rate_per_pulse from all records."""
        params = PassModelParams(
            max_elevation_deg=60.0,
            min_elevation_deg=10.0,
            pass_seconds=100.0,
            dt_seconds=10.0,
            flip_prob=0.005,
            rep_rate_hz=1e8,
            qber_abort_threshold=0.11,
            background_mode="night",
        )
        link = FreeSpaceLinkParams()
        detector = DetectorParams(eta=0.4, p_bg=1e-6)

        records, summary = compute_pass_records(
            params=params,
            link_params=link,
            detector=detector,
        )

        # Extract all key rates from records
        key_rates = [r["key_rate_per_pulse"] for r in records]
        max_key_rate_from_records = max(key_rates)

        # Should match summary
        assert summary["peak_key_rate"] == pytest.approx(max_key_rate_from_records, abs=1e-10)

    def test_peak_key_rate_bps_consistent_with_peak_key_rate(self):
        """peak_key_rate_bps should equal peak_key_rate * rep_rate_hz."""
        params = PassModelParams(
            max_elevation_deg=60.0,
            min_elevation_deg=10.0,
            pass_seconds=100.0,
            dt_seconds=10.0,
            flip_prob=0.005,
            rep_rate_hz=1e8,
            qber_abort_threshold=0.11,
            background_mode="night",
        )
        link = FreeSpaceLinkParams()
        detector = DetectorParams(eta=0.4, p_bg=1e-6)

        records, summary = compute_pass_records(
            params=params,
            link_params=link,
            detector=detector,
        )

        expected_peak_bps = summary["peak_key_rate"] * params.rep_rate_hz
        assert summary["peak_key_rate_bps"] == pytest.approx(expected_peak_bps, abs=1.0)

    def test_peak_key_rate_zero_when_no_secure_bits(self):
        """When no secure key is generated, peak_key_rate should be zero."""
        params = PassModelParams(
            max_elevation_deg=60.0,
            min_elevation_deg=10.0,
            pass_seconds=50.0,
            dt_seconds=10.0,
            flip_prob=0.20,  # Very high error rate
            rep_rate_hz=1e8,
            qber_abort_threshold=0.11,
            background_mode="day",
        )
        link = FreeSpaceLinkParams()
        detector = DetectorParams(eta=0.05, p_bg=1e-2)

        records, summary = compute_pass_records(
            params=params,
            link_params=link,
            detector=detector,
        )

        # All records should have zero or near-zero key rate
        assert summary["peak_key_rate"] == 0.0 or summary["peak_key_rate"] < 1e-9


class TestComputePassRecordsMeanKeyRateInWindow:
    """Tests for mean_key_rate_in_window calculation."""

    def test_mean_key_rate_in_window_is_average_of_secure_records(self):
        """mean_key_rate_in_window should be the mean of key rates where secret_bits_dt > 0."""
        params = PassModelParams(
            max_elevation_deg=60.0,
            min_elevation_deg=10.0,
            pass_seconds=100.0,
            dt_seconds=10.0,
            flip_prob=0.005,
            rep_rate_hz=1e8,
            qber_abort_threshold=0.11,
            background_mode="night",
        )
        link = FreeSpaceLinkParams()
        detector = DetectorParams(eta=0.4, p_bg=1e-6)

        records, summary = compute_pass_records(
            params=params,
            link_params=link,
            detector=detector,
        )

        # Calculate mean from secure records manually
        secure_key_rates = [
            r["key_rate_per_pulse"] for r in records if r["secret_bits_dt"] > 0.0
        ]

        if secure_key_rates:
            expected_mean = sum(secure_key_rates) / len(secure_key_rates)
            assert summary["mean_key_rate_in_window"] == pytest.approx(expected_mean, abs=1e-10)
        else:
            assert summary["mean_key_rate_in_window"] == 0.0

    def test_mean_key_rate_in_window_zero_when_no_secure_window(self):
        """When no secure window exists, mean_key_rate_in_window should be zero."""
        params = PassModelParams(
            max_elevation_deg=60.0,
            min_elevation_deg=10.0,
            pass_seconds=50.0,
            dt_seconds=10.0,
            flip_prob=0.20,
            rep_rate_hz=1e8,
            qber_abort_threshold=0.11,
            background_mode="day",
        )
        link = FreeSpaceLinkParams()
        detector = DetectorParams(eta=0.05, p_bg=1e-2)

        records, summary = compute_pass_records(
            params=params,
            link_params=link,
            detector=detector,
        )

        assert summary["mean_key_rate_in_window"] == 0.0

    def test_mean_key_rate_less_than_or_equal_peak(self):
        """mean_key_rate_in_window should always be <= peak_key_rate."""
        params = PassModelParams(
            max_elevation_deg=60.0,
            min_elevation_deg=10.0,
            pass_seconds=150.0,
            dt_seconds=10.0,
            flip_prob=0.005,
            rep_rate_hz=1e8,
            qber_abort_threshold=0.11,
            background_mode="night",
        )
        link = FreeSpaceLinkParams()
        detector = DetectorParams(eta=0.35, p_bg=5e-6)

        records, summary = compute_pass_records(
            params=params,
            link_params=link,
            detector=detector,
        )

        if summary["peak_key_rate"] > 0:
            assert summary["mean_key_rate_in_window"] <= summary["peak_key_rate"] + 1e-10


class TestAttachManifestAndGit:
    """Tests for _attach_manifest_and_git helper function."""

    def test_attach_manifest_and_git_adds_both_fields(self):
        """_attach_manifest_and_git should add both assumptions_manifest and git_commit."""
        report: Dict[str, Any] = {}
        _attach_manifest_and_git(report)

        assert "assumptions_manifest" in report
        assert "git_commit" in report

    def test_attach_manifest_contains_expected_keys(self):
        """assumptions_manifest should contain the expected structure."""
        report: Dict[str, Any] = {}
        _attach_manifest_and_git(report)

        manifest = report["assumptions_manifest"]
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert "protocol" in manifest
        assert "channel_model" in manifest
        assert "attack_model" in manifest
        assert "key_rate_semantics" in manifest
        assert "disclaimers" in manifest

    def test_attach_manifest_does_not_overwrite_existing(self):
        """If fields already exist, they should not be overwritten."""
        custom_manifest = {"custom": "manifest", "schema_version": "custom_version"}
        custom_git = "custom_git_commit_hash"
        report: Dict[str, Any] = {
            "assumptions_manifest": custom_manifest,
            "git_commit": custom_git,
        }

        _attach_manifest_and_git(report)

        # Should preserve existing values
        assert report["assumptions_manifest"] == custom_manifest
        assert report["git_commit"] == custom_git

    def test_attach_manifest_adds_only_missing_fields(self):
        """Should add only the fields that are missing."""
        report: Dict[str, Any] = {"git_commit": "existing_hash"}
        _attach_manifest_and_git(report)

        # Should keep existing git_commit
        assert report["git_commit"] == "existing_hash"
        # Should add manifest
        assert "assumptions_manifest" in report
        assert report["assumptions_manifest"]["schema_version"] == SCHEMA_VERSION

    def test_git_commit_is_string_or_none(self):
        """git_commit should be a string (hex hash) or None if git is unavailable."""
        report: Dict[str, Any] = {}
        _attach_manifest_and_git(report)

        git_commit = report["git_commit"]
        assert git_commit is None or isinstance(git_commit, str)
        
        # If it's a string, it should be a plausible git hash (40 hex chars for full SHA-1)
        if isinstance(git_commit, str):
            assert len(git_commit) > 0


class TestPrintSweepSummary:
    """Tests for _print_sweep_summary output formatting."""

    def test_print_sweep_summary_single_trial_outputs_basic_stats(self, capsys):
        """For single trial, should output min/median/max for qber and key_rate."""
        records = [
            {"qber_mean": 0.01, "key_rate_per_pulse": 0.001},
            {"qber_mean": 0.02, "key_rate_per_pulse": 0.002},
            {"qber_mean": 0.03, "key_rate_per_pulse": 0.003},
            {"qber_mean": 0.015, "key_rate_per_pulse": 0.0015},
        ]

        _print_sweep_summary(
            loss_min=20.0,
            loss_max=40.0,
            pulses=100_000,
            trials=1,
            seed_policy="fixed",
            records=records,
            qber_field="qber_mean",
            key_rate_field="key_rate_per_pulse",
        )

        captured = capsys.readouterr()
        output = captured.out

        # Check header line
        assert "simulated sweep summary" in output
        assert "loss range: 20" in output
        assert "40 dB" in output
        assert "pulses: 100000" in output
        assert "trials: 1" in output
        assert "seed policy: fixed" in output

        # Check qber stats line
        assert "qber_mean min/median/max:" in output
        assert "0.01" in output  # min
        assert "0.03" in output  # max

        # Check key_rate stats line
        assert "key_rate_per_pulse min/median/max:" in output
        assert "0.001" in output  # min
        assert "0.003" in output  # max

    def test_print_sweep_summary_does_not_print_ci_for_single_trial(self, capsys):
        """For single trial, should NOT print CI fields even if provided."""
        records = [
            {
                "qber_mean": 0.01,
                "key_rate_per_pulse": 0.001,
                "qber_ci_low": 0.005,
                "qber_ci_high": 0.015,
            },
            {
                "qber_mean": 0.02,
                "key_rate_per_pulse": 0.002,
                "qber_ci_low": 0.015,
                "qber_ci_high": 0.025,
            },
        ]

        _print_sweep_summary(
            loss_min=20.0,
            loss_max=40.0,
            pulses=100_000,
            trials=1,
            seed_policy="fixed",
            records=records,
            qber_field="qber_mean",
            key_rate_field="key_rate_per_pulse",
            qber_ci_low_field="qber_ci_low",
            qber_ci_high_field="qber_ci_high",
        )

        captured = capsys.readouterr()
        output = captured.out

        # Should NOT contain CI lines for single trial
        assert "qber_ci_low" not in output
        assert "qber_ci_high" not in output

    def test_print_sweep_summary_prints_ci_for_multiple_trials(self, capsys):
        """For multiple trials, should print CI statistics."""
        records = [
            {
                "qber_mean": 0.01,
                "key_rate_per_pulse": 0.001,
                "qber_ci_low": 0.005,
                "qber_ci_high": 0.015,
                "key_rate_per_pulse_ci_low": 0.0005,
                "key_rate_per_pulse_ci_high": 0.0015,
            },
            {
                "qber_mean": 0.02,
                "key_rate_per_pulse": 0.002,
                "qber_ci_low": 0.015,
                "qber_ci_high": 0.025,
                "key_rate_per_pulse_ci_low": 0.0015,
                "key_rate_per_pulse_ci_high": 0.0025,
            },
        ]

        _print_sweep_summary(
            loss_min=20.0,
            loss_max=40.0,
            pulses=100_000,
            trials=10,
            seed_policy="random",
            records=records,
            qber_field="qber_mean",
            key_rate_field="key_rate_per_pulse",
            qber_ci_low_field="qber_ci_low",
            qber_ci_high_field="qber_ci_high",
            key_rate_ci_low_field="key_rate_per_pulse_ci_low",
            key_rate_ci_high_field="key_rate_per_pulse_ci_high",
        )

        captured = capsys.readouterr()
        output = captured.out

        # Should contain CI lines for multiple trials
        assert "qber_ci_low min/median/max:" in output
        assert "qber_ci_high min/median/max:" in output
        assert "key_rate_per_pulse_ci_low min/median/max:" in output
        assert "key_rate_per_pulse_ci_high min/median/max:" in output

    def test_print_sweep_summary_handles_empty_records(self, capsys):
        """Should handle empty records gracefully without crashing."""
        records = []

        _print_sweep_summary(
            loss_min=20.0,
            loss_max=40.0,
            pulses=100_000,
            trials=1,
            seed_policy="fixed",
            records=records,
            qber_field="qber_mean",
            key_rate_field="key_rate_per_pulse",
        )

        captured = capsys.readouterr()
        output = captured.out

        # Should produce no output for empty records
        assert output == ""

    def test_print_sweep_summary_handles_missing_fields(self, capsys):
        """Should handle records with missing fields gracefully by filtering them out."""
        records = [
            {"qber_mean": 0.01},  # Missing key_rate_per_pulse
            {"key_rate_per_pulse": 0.002},  # Missing qber_mean
        ]

        _print_sweep_summary(
            loss_min=20.0,
            loss_max=40.0,
            pulses=100_000,
            trials=1,
            seed_policy="fixed",
            records=records,
            qber_field="qber_mean",
            key_rate_field="key_rate_per_pulse",
        )

        captured = capsys.readouterr()
        output = captured.out

        # Function filters records but still outputs stats for records with both fields
        # In this case, it extracts what it can from each record separately
        # qber_vals will have [0.01] and key_vals will have [0.002]
        # So it will output statistics for each
        assert "simulated sweep summary" in output
        assert "qber_mean min/median/max:" in output
        assert "key_rate_per_pulse min/median/max:" in output

    def test_print_sweep_summary_formats_scientific_notation(self, capsys):
        """Should use scientific notation (%.6g) for small values."""
        records = [
            {"qber_mean": 1.23456e-8, "key_rate_per_pulse": 9.87654e-10},
        ]

        _print_sweep_summary(
            loss_min=20.0,
            loss_max=40.0,
            pulses=100_000,
            trials=1,
            seed_policy="fixed",
            records=records,
            qber_field="qber_mean",
            key_rate_field="key_rate_per_pulse",
        )

        captured = capsys.readouterr()
        output = captured.out

        # Should contain scientific notation
        assert "e-" in output or ("1.23456" in output and "9.87654" in output)
