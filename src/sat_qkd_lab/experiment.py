"""
Blinded intervention A/B experiment harness for QKD metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import random

import numpy as np

from .finite_key import FiniteKeyParams, finite_key_rate_per_pulse
from .helpers import h2
from .eb_observables import compute_observables


@dataclass(frozen=True)
class ExperimentParams:
    seed: int = 0
    n_blocks: int = 20
    block_seconds: float = 30.0
    rep_rate_hz: float = 1e8
    pass_seconds: float = 600.0
    qber_abort_threshold: float = 0.11
    base_qber: float = 0.03
    qber_jitter: float = 0.002
    intervention_qber_shift: float = 0.0
    intervention_secret_bits_scale: float = 1.0
    sifted_fraction: float = 0.25


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def generate_schedule(n_blocks: int, seed: int) -> List[str]:
    """Generate a randomized control/intervention schedule."""
    rng = random.Random(seed)
    n_control = n_blocks // 2 + (n_blocks % 2)
    n_intervention = n_blocks - n_control
    labels = ["control"] * n_control + ["intervention"] * n_intervention
    rng.shuffle(labels)
    return labels


def _assignment_codes(n_blocks: int, rng: random.Random) -> List[str]:
    """Generate blinded assignment codes."""
    return [f"blk-{rng.getrandbits(32):08x}" for _ in range(n_blocks)]


def _simulate_block_metrics(
    label: str,
    params: ExperimentParams,
    rng: random.Random,
    finite_key: Optional[FiniteKeyParams],
    bell_mode: bool,
) -> Dict[str, float]:
    """Simulate per-block metrics with optional finite-key calculation."""
    qber_mean = params.base_qber + rng.gauss(0.0, params.qber_jitter)
    if label == "intervention":
        qber_mean += params.intervention_qber_shift
    qber_mean = min(0.5, max(0.0, qber_mean))

    n_sent = int(round(params.rep_rate_hz * params.block_seconds))
    n_sifted = int(round(n_sent * params.sifted_fraction))
    n_errors = int(round(qber_mean * n_sifted))

    if finite_key is not None:
        fk = finite_key_rate_per_pulse(
            n_sent=n_sent,
            n_sifted=n_sifted,
            n_errors=n_errors,
            params=finite_key,
            qber_abort_threshold=params.qber_abort_threshold,
        )
        key_rate_per_pulse = fk["key_rate_per_pulse"]
        finite_key_info = fk["finite_key"]
        if finite_key_info["status"] != "secure":
            key_rate_per_pulse = 0.0
    else:
        secret_fraction = max(0.0, 1.0 - 2.0 * h2(qber_mean))
        key_rate_per_pulse = params.sifted_fraction * secret_fraction

    if label == "intervention":
        key_rate_per_pulse *= params.intervention_secret_bits_scale

    total_secret_bits = key_rate_per_pulse * n_sent
    headroom = params.qber_abort_threshold - qber_mean

    metrics: Dict[str, Any] = {
        "qber_mean": float(qber_mean),
        "headroom": float(headroom),
        "total_secret_bits": float(total_secret_bits),
    }
    if finite_key is not None:
        metrics["finite_key"] = finite_key_info
    if bell_mode:
        n_pairs = max(1, n_sifted)
        visibility_est = 1.0 - 2.0 * qber_mean
        visibility_est = min(1.0, max(-1.0, visibility_est))
        matrix_z = _counts_from_correlation(n_pairs, visibility_est)
        matrix_x = _counts_from_correlation(n_pairs, visibility_est)
        matrix_anti = _counts_from_correlation(n_pairs, -visibility_est)
        obs = compute_observables(
            correlation_counts={
                "AB": matrix_z,
                "ABp": matrix_x,
                "ApB": matrix_z,
                "ApBp": matrix_anti,
            },
            n_pairs=n_pairs,
        )
        metrics["bell"] = {
            "coincidence_matrices": {
                "Z": matrix_z,
                "X": matrix_x,
            },
            "visibility": float(obs.visibility),
            "chsh_s": float(obs.chsh_s),
            "chsh_sigma": float(obs.chsh_sigma),
            "correlations": obs.correlations,
        }

    return metrics


def simulate_block_metrics(
    label: str,
    params: ExperimentParams,
    rng: random.Random,
    finite_key: Optional[FiniteKeyParams],
    bell_mode: bool = False,
) -> Dict[str, float]:
    """Public wrapper for block metric simulation."""
    return _simulate_block_metrics(label, params, rng, finite_key, bell_mode)


def _counts_from_correlation(n_pairs: int, correlation: float) -> List[List[int]]:
    total = max(1, int(n_pairs))
    corr = min(1.0, max(-1.0, float(correlation)))
    n_same = int(round((1.0 + corr) * 0.5 * total))
    n_diff = total - n_same
    n00 = n_same // 2
    n11 = n_same - n00
    n01 = n_diff // 2
    n10 = n_diff - n01
    return [[n00, n01], [n10, n11]]


def _difference_in_means(values_control: List[float], values_intervention: List[float]) -> Dict[str, Optional[float]]:
    """Compute difference-in-means and CI for two groups."""
    n_c = len(values_control)
    n_i = len(values_intervention)
    if n_c == 0 or n_i == 0:
        return {
            "control_mean": None,
            "intervention_mean": None,
            "delta": None,
            "ci_low": None,
            "ci_high": None,
            "statistic": None,
        }

    mean_c = float(np.mean(values_control))
    mean_i = float(np.mean(values_intervention))
    delta = mean_i - mean_c

    if n_c < 2 or n_i < 2:
        return {
            "control_mean": mean_c,
            "intervention_mean": mean_i,
            "delta": delta,
            "ci_low": None,
            "ci_high": None,
            "statistic": None,
        }

    var_c = float(np.var(values_control, ddof=1))
    var_i = float(np.var(values_intervention, ddof=1))
    se = math.sqrt(var_c / n_c + var_i / n_i)
    if se <= 0:
        return {
            "control_mean": mean_c,
            "intervention_mean": mean_i,
            "delta": delta,
            "ci_low": delta,
            "ci_high": delta,
            "statistic": 0.0,
        }

    z = 1.96
    return {
        "control_mean": mean_c,
        "intervention_mean": mean_i,
        "delta": delta,
        "ci_low": delta - z * se,
        "ci_high": delta + z * se,
        "statistic": delta / se,
    }


def run_experiment(
    params: ExperimentParams,
    metrics: List[str],
    outdir: Path,
    finite_key: Optional[FiniteKeyParams],
    bell_mode: bool = False,
    unblind: bool = False,
    sync_params_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run blinded experiment and write outputs to disk."""
    rng = random.Random(params.seed)
    labels = generate_schedule(params.n_blocks, params.seed)
    codes = _assignment_codes(params.n_blocks, rng)

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "reports").mkdir(exist_ok=True)

    schedule_blinded = {
        "schema_version": "1.0",
        "mode": "blinded_schedule",
        "timestamp_utc": _timestamp_utc(),
        "blocks": [
            {
                "block_index": i,
                "assignment_code": codes[i],
            }
            for i in range(params.n_blocks)
        ],
    }

    schedule_blinded_path = outdir / "reports" / "schedule_blinded.json"
    with open(schedule_blinded_path, "w") as f:
        json.dump(schedule_blinded, f, indent=2)
        f.write("\n")

    schedule_unblinded_path = outdir / "reports" / "schedule_unblinded.json"
    if unblind:
        schedule_unblinded = {
            "schema_version": "1.0",
            "mode": "unblinded_schedule",
            "timestamp_utc": _timestamp_utc(),
            "blocks": [
                {
                    "block_index": i,
                    "assignment_code": codes[i],
                    "label": labels[i],
                }
                for i in range(params.n_blocks)
            ],
        }
        with open(schedule_unblinded_path, "w") as f:
            json.dump(schedule_unblinded, f, indent=2)
            f.write("\n")

    block_results = []
    for i, label in enumerate(labels):
        metrics_out = _simulate_block_metrics(label, params, rng, finite_key, bell_mode)
        block_results.append({
            "block_index": i,
            "t_start_seconds": float(i * params.block_seconds),
            "t_end_seconds": float((i + 1) * params.block_seconds),
            "metrics": {k: metrics_out[k] for k in metrics},
        })
        if bell_mode and "bell" in metrics_out:
            block_results[-1]["bell"] = metrics_out["bell"]

    analysis_metrics: Dict[str, Any] = {}
    if unblind:
        metrics_by_group = {m: {"control": [], "intervention": []} for m in metrics}
        for label, block in zip(labels, block_results):
            for metric_name, value in block["metrics"].items():
                metrics_by_group[metric_name][label].append(value)
        for metric_name in metrics:
            stats = _difference_in_means(
                metrics_by_group[metric_name]["control"],
                metrics_by_group[metric_name]["intervention"],
            )
            analysis_metrics[metric_name] = {
                "control_mean": stats["control_mean"],
                "intervention_mean": stats["intervention_mean"],
                "delta_intervention_minus_control": stats["delta"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "test": {
                    "name": "difference_in_means",
                    "statistic": stats["statistic"],
                },
            }
    else:
        for metric_name in metrics:
            analysis_metrics[metric_name] = {
                "control_mean": None,
                "intervention_mean": None,
                "delta_intervention_minus_control": None,
                "ci_low": None,
                "ci_high": None,
                "test": {
                    "name": "difference_in_means",
                    "statistic": None,
                },
            }

    output = {
        "schema_version": "1.0",
        "mode": "experiment-run",
        "timestamp_utc": _timestamp_utc(),
        "inputs": {
            "seed": params.seed,
            "n_blocks": params.n_blocks,
            "block_seconds": params.block_seconds,
            "rep_rate_hz": params.rep_rate_hz,
            "pass_seconds": params.pass_seconds,
            "metrics": metrics,
            "bell_mode": bool(bell_mode),
            "sync_params": sync_params_path,
            "finite_key": {
                "enabled": finite_key is not None,
                "epsilon_sec": finite_key.eps_sec if finite_key is not None else None,
                "epsilon_cor": finite_key.eps_cor if finite_key is not None else None,
            },
        },
        "units": {
            "block_seconds": "s",
            "rep_rate_hz": "Hz",
            "pass_seconds": "s",
            "qber": "unitless",
            "headroom": "unitless",
            "secret_bits": "bits",
        },
        "blinding": {
            "blinded": True,
            "schedule_path": str(schedule_blinded_path.relative_to(outdir)),
            "unblind_required": True,
        },
        "block_results": block_results,
        "analysis": {
            "group_labels_included": bool(unblind),
            "metrics": analysis_metrics,
        },
        "artifacts": {
            "schedule_files": {
                "blinded": str(schedule_blinded_path.relative_to(outdir)),
                "unblinded_optional": str(schedule_unblinded_path.relative_to(outdir)),
            }
        },
    }

    output_path = outdir / "reports" / "latest_experiment.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    return output
