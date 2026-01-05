"""
Helper functions for dashboard export packets and glossary.

Pure functions that can be tested without Streamlit runtime.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional


# Glossary definitions for operator console
GLOSSARY = {
    "QBER": {
        "short": "Quantum Bit Error Rate",
        "definition": "Fraction of mismatched bits after sifting (range: 0 to 0.5). High QBER indicates high noise or potential attack. Abort threshold: typically 11% for BB84.",
        "interpretation": "Lower is better. If QBER > 11%, privacy amplification cannot extract a secure key.",
    },
    "Secret Fraction": {
        "short": "Secret fraction",
        "definition": "Fraction of sifted bits remaining after privacy amplification (range: 0 to 1). Zero means no extractable key.",
        "interpretation": "Scales as 1 - 2*h(QBER) for BB84. Drops to zero when QBER exceeds abort threshold.",
    },
    "Key Rate": {
        "short": "Key rate per pulse",
        "definition": "Secret bits produced per sent pulse (units: bits/pulse). Accounts for BB84 sifting factor (~0.5) and secret fraction.",
        "interpretation": "Multiply by rep rate (Hz) to get bits per second. Zero key rate means no secure key despite bits being exchanged.",
    },
    "Finite-Key Penalty": {
        "short": "Finite-key penalty",
        "definition": "Gap between asymptotic and finite-key rates due to statistical uncertainty (larger blocks reduce penalty).",
        "interpretation": "Finite-key rate is always <= asymptotic rate. Penalty quantifies cost of limited sample size.",
    },
    "Headroom": {
        "short": "Abort headroom",
        "definition": "Distance to abort threshold (qber_abort - qber_mean). Positive headroom indicates margin for noise fluctuations.",
        "interpretation": "Headroom < 0.02 (2%) is risky. Headroom > 0.05 (5%) provides operational buffer.",
    },
    "CAR": {
        "short": "Coincidence-to-accidental ratio",
        "definition": "Ratio of true coincidences to accidental background coincidences in entanglement-based QKD.",
        "interpretation": "CAR >> 1 indicates good signal quality. CAR ~ 1 means signal dominated by background.",
    },
}


def get_glossary_entry(key: str) -> Dict[str, str]:
    """Retrieve glossary entry for a metric."""
    return GLOSSARY.get(key, {"short": key, "definition": "No definition available.", "interpretation": ""})


def format_glossary_markdown() -> str:
    """Format full glossary as markdown."""
    lines = ["## Glossary\n"]
    for metric, entry in GLOSSARY.items():
        lines.append(f"### {metric}")
        lines.append(f"**{entry['short']}**\n")
        lines.append(f"{entry['definition']}\n")
        if entry.get("interpretation"):
            lines.append(f"*Interpretation:* {entry['interpretation']}\n")
        lines.append("")
    return "\n".join(lines)


def create_export_packet(
    outdir: Path,
    preset_name: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Create a timestamped export packet directory with:
    - latest.json report
    - assumptions.json snapshot
    - key plots (if present)
    - summary.md

    Returns the path to the export directory.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_dir = outdir / "reports" / "exports" / f"export_{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Copy latest.json if it exists
    latest_json = outdir / "reports" / "latest.json"
    if latest_json.exists():
        shutil.copy(latest_json, export_dir / "latest.json")

    # Copy latest_pass.json if it exists
    latest_pass_json = outdir / "reports" / "latest_pass.json"
    if latest_pass_json.exists():
        shutil.copy(latest_pass_json, export_dir / "latest_pass.json")

    # Copy dashboard reports if they exist
    dashboard_sweep = outdir / "reports" / "dashboard_sweep.json"
    if dashboard_sweep.exists():
        shutil.copy(dashboard_sweep, export_dir / "dashboard_sweep.json")

    dashboard_pass = outdir / "reports" / "dashboard_pass.json"
    if dashboard_pass.exists():
        shutil.copy(dashboard_pass, export_dir / "dashboard_pass.json")

    # Create assumptions snapshot
    assumptions_snapshot = _build_assumptions_snapshot(preset_name, overrides)
    with open(export_dir / "assumptions.json", "w") as f:
        json.dump(assumptions_snapshot, f, indent=2)
        f.write("\n")

    # Copy key plots
    plot_files = [
        "key_qber_vs_loss.png",
        "key_fraction_vs_loss.png",
        "key_rate_vs_elevation.png",
        "secure_window_per_pass.png",
        "loss_vs_elevation.png",
        "qber_headroom_vs_loss.png",
    ]

    for plot_file in plot_files:
        src = outdir / "figures" / plot_file
        if src.exists():
            shutil.copy(src, export_dir / plot_file)

    # Create summary.md
    summary_md = _build_summary_markdown(preset_name, overrides, export_dir)
    with open(export_dir / "summary.md", "w") as f:
        f.write(summary_md)

    return export_dir


def _build_assumptions_snapshot(
    preset_name: Optional[str],
    overrides: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build assumptions snapshot for export packet."""
    snapshot = {
        "schema_version": "1.0",
        "export_type": "dashboard_export",
        "timestamp_utc": datetime.now(timezone.utc).isoformat() + "Z",
        "preset": preset_name or "custom",
        "overrides": overrides or {},
    }
    return snapshot


def _build_summary_markdown(
    preset_name: Optional[str],
    overrides: Optional[Dict[str, Any]],
    export_dir: Path,
) -> str:
    """Build one-page summary markdown for export packet."""
    lines = [
        "# QKD Simulation Export Summary\n",
        f"**Export timestamp:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n",
        f"**Preset:** {preset_name or 'Custom configuration'}\n",
    ]

    if overrides:
        lines.append("\n## Parameter Overrides\n")
        for key, value in overrides.items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")

    lines.append("\n## Included Files\n")
    files = list(export_dir.glob("*"))
    for f in sorted(files, key=lambda x: x.name):
        if f.is_file() and f.name != "summary.md":
            lines.append(f"- `{f.name}`")
    lines.append("")

    lines.append("\n## Interpretation Guide\n")
    lines.append("\n### QBER (Quantum Bit Error Rate)\n")
    lines.append("Fraction of bit errors after sifting. High QBER (>11%) means no secure key can be extracted.\n")
    lines.append("**What to look for:** QBER should stay well below 11% abort threshold. If QBER approaches 10%, you are near the security cliff.\n")

    lines.append("\n### Secret Fraction\n")
    lines.append("Fraction of sifted bits remaining after privacy amplification. Zero means no key.\n")
    lines.append("**What to look for:** Secret fraction drops sharply as QBER increases. At QBER=11%, secret fraction hits zero (abort).\n")

    lines.append("\n### Key Rate Per Pulse\n")
    lines.append("Secret bits per sent pulse. Multiply by repetition rate to get bits/second.\n")
    lines.append("**What to look for:** Positive key rate means viable QKD. Zero key rate despite detections means bits without secrecy (security cliff).\n")

    lines.append("\n### Headroom\n")
    lines.append("Distance to abort threshold. Headroom = qber_abort - qber_mean.\n")
    lines.append("**What to look for:** Headroom > 5% is safe margin. Headroom < 2% is operationally risky.\n")

    lines.append("\n---\n")
    lines.append("*For full glossary and technical details, see repository documentation.*\n")

    return "\n".join(lines)
