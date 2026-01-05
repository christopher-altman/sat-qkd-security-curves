"""
Tests for dashboard export packet functionality.

These tests validate the pure helper functions without requiring Streamlit runtime.
"""
from __future__ import annotations

from pathlib import Path
import json

from sat_qkd_lab.dashboard_helpers import (
    create_export_packet,
    get_glossary_entry,
    format_glossary_markdown,
    GLOSSARY,
)
from sat_qkd_lab.dashboard_presets import (
    PRESETS,
    get_preset_by_name,
    list_preset_names,
)


def test_glossary_entries():
    """Test that glossary entries are well-formed."""
    assert "QBER" in GLOSSARY
    assert "Secret Fraction" in GLOSSARY
    assert "Key Rate" in GLOSSARY

    qber_entry = get_glossary_entry("QBER")
    assert "short" in qber_entry
    assert "definition" in qber_entry
    assert "interpretation" in qber_entry
    assert len(qber_entry["definition"]) > 10


def test_glossary_markdown_format():
    """Test that glossary markdown is non-empty and well-formed."""
    md = format_glossary_markdown()
    assert "## Glossary" in md
    assert "QBER" in md
    assert "Secret Fraction" in md


def test_preset_catalog():
    """Test that preset catalog is populated and accessible."""
    assert len(PRESETS) >= 4
    names = list_preset_names()
    assert len(names) >= 4
    assert "Night / Low Background / Low Turbulence" in names


def test_get_preset_by_name():
    """Test preset retrieval by name."""
    preset = get_preset_by_name("Night / Low Background / Low Turbulence")
    assert preset.name == "Night / Low Background / Low Turbulence"
    assert "sweep_params" in vars(preset)
    assert "pass_params" in vars(preset)
    assert preset.sweep_params["loss_min"] == 20.0


def test_create_export_packet_minimal(tmp_path):
    """Test export packet creation with minimal setup (no existing reports)."""
    outdir = tmp_path / "test_export"
    outdir.mkdir()
    (outdir / "reports").mkdir()
    (outdir / "figures").mkdir()

    export_dir = create_export_packet(outdir, preset_name="Test Preset", overrides={"eta": 0.3})

    # Verify export directory structure
    assert export_dir.exists()
    assert (export_dir / "assumptions.json").exists()
    assert (export_dir / "summary.md").exists()

    # Verify assumptions.json content
    with open(export_dir / "assumptions.json", "r") as f:
        assumptions = json.load(f)
    assert assumptions["preset"] == "Test Preset"
    assert "overrides" in assumptions
    assert assumptions["export_type"] == "dashboard_export"

    # Verify summary.md content
    with open(export_dir / "summary.md", "r") as f:
        summary = f.read()
    assert "Test Preset" in summary
    assert "QBER" in summary
    assert "Interpretation Guide" in summary


def test_create_export_packet_with_reports(tmp_path):
    """Test export packet creation when reports and plots exist."""
    outdir = tmp_path / "test_export_full"
    outdir.mkdir()
    reports_dir = outdir / "reports"
    reports_dir.mkdir()
    figures_dir = outdir / "figures"
    figures_dir.mkdir()

    # Create dummy report files
    dummy_report = {"schema_version": "1.0", "data": [1, 2, 3]}
    with open(reports_dir / "latest.json", "w") as f:
        json.dump(dummy_report, f)

    with open(reports_dir / "latest_pass.json", "w") as f:
        json.dump({"mode": "pass"}, f)

    # Create dummy plot files
    (figures_dir / "key_qber_vs_loss.png").write_bytes(b"fake_png_data")
    (figures_dir / "key_fraction_vs_loss.png").write_bytes(b"fake_png_data2")

    export_dir = create_export_packet(outdir, preset_name=None, overrides=None)

    # Verify all files were copied
    assert (export_dir / "latest.json").exists()
    assert (export_dir / "latest_pass.json").exists()
    assert (export_dir / "key_qber_vs_loss.png").exists()
    assert (export_dir / "key_fraction_vs_loss.png").exists()
    assert (export_dir / "assumptions.json").exists()
    assert (export_dir / "summary.md").exists()

    # Verify copied report content
    with open(export_dir / "latest.json", "r") as f:
        copied_report = json.load(f)
    assert copied_report["data"] == [1, 2, 3]


def test_export_packet_missing_plots_ok(tmp_path):
    """Test that export packet succeeds even if plots are missing."""
    outdir = tmp_path / "test_export_no_plots"
    outdir.mkdir()
    (outdir / "reports").mkdir()
    (outdir / "figures").mkdir()

    # Create export with no plots present
    export_dir = create_export_packet(outdir)

    # Should still create core files
    assert (export_dir / "assumptions.json").exists()
    assert (export_dir / "summary.md").exists()

    # But no plot files should be present
    plot_files = list(export_dir.glob("*.png"))
    assert len(plot_files) == 0
