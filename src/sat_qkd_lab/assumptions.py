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
        "cv_qkd": {
            "protocol": "GG02 (Gaussian-modulated coherent states)",
            "status": "scaffold",
            "threat_model": "placeholder; not a security guarantee",
            "computed": {
                "snr": "Signal-to-noise ratio estimate from channel model",
                "I_AB": "Mutual information I(A:B) using standard Gaussian channel formula",
                "chi_BE": "Holevo bound χ(B:E) - NOT YET IMPLEMENTED (returns None)",
                "secret_key_rate": "NOT YET VALIDATED - requires full Holevo bound computation",
            },
            "not_computed": [
                "Full composable security proof",
                "Optimal collective attack analysis",
                "Covariance matrix symplectic eigenvalue analysis",
                "Reverse reconciliation key rate (only toy direct reconciliation stub)",
            ],
            "loss_to_transmittance_mapping": "T = 10^(-loss_db/10)",
            "notes": [
                "This is a SCAFFOLD implementation for structural demonstration only.",
                "SNR and I(A:B) use standard formulas but are not validated against experiments.",
                "Holevo bound computation is stubbed; secret key rate is not yet available.",
                "Do NOT use for production security claims or deployment planning.",
            ],
        },
    }
