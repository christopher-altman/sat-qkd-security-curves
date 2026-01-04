"""Assumptions manifest for BB84 security curve outputs."""

from __future__ import annotations

from typing import Any, Dict


def build_assumptions_manifest(schema_version: str) -> Dict[str, Any]:
    """Return a stable, machine-readable assumptions manifest."""
    return {
        "schema_version": schema_version,
        "schema_version_note": (
            "Primary reports/latest.json schema stays at 0.4 until breaking changes. "
            "Bump only with changelog updates."
        ),
        "protocol": {
            "name": "BB84",
            "mode": "prepare-and-measure",
            "basis_choice": "IID random basis selection",
            "sifting": {
                "basis_match_probability": 0.5,
                "notes": "Key-rate calculations assume 1/2 sifting unless explicitly overridden.",
            },
        },
        "channel_model": {
            "loss_db": "Applied as link loss derived from elevation-to-loss model.",
            "flip_prob": "Applied as intrinsic bit-flip probability in the BB84 model.",
            "p_bg": "Background/dark count probability per pulse; day/night scaling may apply.",
            "eta": "Detector efficiency applied to detection probability.",
        },
        "attack_model": {
            "default": "none",
            "supported": ["none", "intercept_resend", "pns", "time_shift", "blinding"],
        },
        "key_rate_semantics": {
            "secret_fraction": "Asymptotic secret fraction from 1 - f*h(Q) - h(Q).",
            "key_rate_per_pulse": "Secret bits per emitted pulse (after sifting and privacy).",
            "finite_key": "Hoeffding-bound finite-key adjustments when enabled.",
        },
        "disclaimers": {
            "toy_model": (
                "Educational simulator; outputs are not validated against real hardware or missions."
            ),
            "physical_accuracy": "Not a validated optical link budget or composable security proof.",
        },
    }
