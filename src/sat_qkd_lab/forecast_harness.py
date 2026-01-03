"""
Forecast harness orchestration for blinded MPI-style scoring.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import random

from .experiment import ExperimentParams, simulate_block_metrics
from .forecast import Forecast, load_forecasts
from .pass_model import PassModelParams, compute_pass_records
from .detector import DEFAULT_DETECTOR, DetectorParams
from .fim_identifiability import compute_fim_identifiability, propagate_uncertainty
from .telemetry import TelemetryRecord
from .scoring import robust_z_score, score_forecast
from .windows import generate_windows, assign_groups_blinded


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _collect_metric_values(outcomes_by_window: Dict[str, Dict[str, float]]) -> Dict[str, List[float]]:
    values: Dict[str, List[float]] = {}
    for metrics in outcomes_by_window.values():
        for key, value in metrics.items():
            if value is None:
                continue
            values.setdefault(key, []).append(value)
    return values


def _group_stats(
    labels_by_window: Dict[str, str],
    outcomes_by_window: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    groups = {"control", "intervention"}
    for group in groups:
        group_values: Dict[str, List[float]] = {}
        for window_id, label in labels_by_window.items():
            if label != group:
                continue
            metrics = outcomes_by_window.get(window_id, {})
            for key, value in metrics.items():
                group_values.setdefault(key, []).append(value)
        stats[group] = {
            f"{metric}_mean": float(sum(vals) / len(vals)) if vals else None
            for metric, vals in group_values.items()
        }
    return stats


def run_forecast_harness(
    forecasts_path: str,
    outdir: Path,
    seed: int,
    n_blocks: int,
    block_seconds: float,
    rep_rate_hz: float,
    unblind: bool,
    estimate_identifiability: bool = False,
) -> Dict[str, Any]:
    """Run forecast harness and write blinded/unblinded reports."""
    forecasts = load_forecasts(forecasts_path)
    windows = generate_windows(n_blocks=n_blocks, block_seconds=block_seconds, seed=seed)
    assignments = assign_groups_blinded(windows, seed=seed)
    labels_by_window = assignments["labels_by_window"]

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "reports").mkdir(exist_ok=True)

    schedule_blinded_path = outdir / "reports" / "schedule_blinded.json"
    schedule_blinded = {
        "schema_version": "1.0",
        "mode": "forecast_schedule_blinded",
        "timestamp_utc": _timestamp_utc(),
        "windows": assignments["schedule_blinded"],
    }
    with open(schedule_blinded_path, "w") as f:
        json.dump(schedule_blinded, f, indent=2)
        f.write("\n")

    schedule_unblinded_path = outdir / "reports" / "schedule_unblinded.json"
    if unblind:
        schedule_unblinded = {
            "schema_version": "1.0",
            "mode": "forecast_schedule_unblinded",
            "timestamp_utc": _timestamp_utc(),
            "windows": assignments["schedule_unblinded"],
        }
        with open(schedule_unblinded_path, "w") as f:
            json.dump(schedule_unblinded, f, indent=2)
            f.write("\n")

    exp_params = ExperimentParams(
        seed=seed,
        n_blocks=n_blocks,
        block_seconds=block_seconds,
        rep_rate_hz=rep_rate_hz,
        pass_seconds=n_blocks * block_seconds,
    )
    rng = random.Random(seed)

    outcomes_by_window: Dict[str, Dict[str, float]] = {}
    for window in windows:
        label = labels_by_window[window.window_id]
        metrics = simulate_block_metrics(label, exp_params, rng, finite_key=None)
        outcomes_by_window[window.window_id] = metrics

    pass_params = PassModelParams(
        pass_seconds=n_blocks * block_seconds,
        dt_seconds=block_seconds,
        rep_rate_hz=rep_rate_hz,
    )
    pass_records, pass_summary = compute_pass_records(params=pass_params)
    if pass_records:
        key_rate_bps_mean = sum(r["key_rate_bps"] for r in pass_records) / len(pass_records)
    else:
        key_rate_bps_mean = 0.0
    for window in windows:
        outcomes_by_window[window.window_id].update({
            "peak_key_rate_bps": pass_summary["peak_key_rate_bps"],
            "key_rate_bps": float(key_rate_bps_mean),
            "secure_window_seconds": pass_summary["secure_window_seconds"],
        })

    values_by_metric = _collect_metric_values(outcomes_by_window)
    baseline_by_metric = {
        metric: (sum(vals) / len(vals) if vals else None)
        for metric, vals in values_by_metric.items()
    }

    scores = []
    for fc in forecasts:
        outcome_value = outcomes_by_window.get(fc.window_id, {}).get(fc.metric_name)
        scored = score_forecast(fc, outcome_value, baseline=baseline_by_metric.get(fc.metric_name))
        z_score = None
        if outcome_value is not None:
            z_score = robust_z_score(outcome_value, values_by_metric.get(fc.metric_name, []))

        scores.append({
            "forecast_id": fc.forecast_id,
            "window_id": fc.window_id,
            "metric": fc.metric_name,
            "operator": fc.operator,
            "value": fc.value,
            "hit": scored["hit"],
            "error": scored["error"],
            "z_score_robust": z_score,
        })

    hits = [s["hit"] for s in scores if s["hit"] is not None]
    n_hits = sum(1 for h in hits if h)
    summary = {
        "n_forecasts": len(scores),
        "n_hits": n_hits,
        "hit_rate": (n_hits / len(hits)) if hits else 0.0,
    }

    output = {
        "schema_version": "1.0",
        "mode": "forecast-run",
        "timestamp_utc": _timestamp_utc(),
        "inputs": {
            "seed": seed,
            "n_blocks": n_blocks,
            "block_seconds": block_seconds,
            "forecasts_path": str(forecasts_path),
            "unblind": bool(unblind),
            "estimate_identifiability": bool(estimate_identifiability),
        },
        "blinding": {
            "blinded": True,
            "schedule_path": str(schedule_blinded_path.relative_to(outdir)),
            "unblind_required": True,
        },
        "windows": [asdict(window) for window in windows],
        "outcomes": {
            "by_window": outcomes_by_window,
        },
        "scores": scores,
        "summary": summary,
    }

    if estimate_identifiability and pass_records:
        telemetry_records = [
            TelemetryRecord(loss_db=rec["loss_db"], qber_mean=rec["qber_mean"])
            for rec in pass_records
        ]
        base_params = {
            "eta_scale": 1.0,
            "p_bg": DEFAULT_DETECTOR.p_bg,
            "flip_prob": pass_params.flip_prob,
        }
        ident = compute_fim_identifiability(
            records=telemetry_records,
            eta_base=DEFAULT_DETECTOR.eta,
            params=base_params,
        )

        def _pass_metrics(params: Dict[str, float]) -> Dict[str, float]:
            det = DetectorParams(
                eta=DEFAULT_DETECTOR.eta * params["eta_scale"],
                p_bg=params["p_bg"],
                eta_z=DEFAULT_DETECTOR.eta_z,
                eta_x=DEFAULT_DETECTOR.eta_x,
            )
            pm = PassModelParams(
                max_elevation_deg=pass_params.max_elevation_deg,
                min_elevation_deg=pass_params.min_elevation_deg,
                pass_seconds=pass_params.pass_seconds,
                dt_seconds=pass_params.dt_seconds,
                flip_prob=params["flip_prob"],
                rep_rate_hz=pass_params.rep_rate_hz,
                qber_abort_threshold=pass_params.qber_abort_threshold,
                background_mode=pass_params.background_mode,
            )
            recs, _ = compute_pass_records(params=pm, detector=det)
            if not recs:
                return {"key_rate_bps": 0.0, "headroom": 0.0}
            key_rate_bps_mean = sum(r["key_rate_bps"] for r in recs) / len(recs)
            headroom_mean = sum(r["headroom"] for r in recs) / len(recs)
            return {"key_rate_bps": float(key_rate_bps_mean), "headroom": float(headroom_mean)}

        key_rate_mean, key_rate_std = propagate_uncertainty(
            lambda p: _pass_metrics(p)["key_rate_bps"],
            base_params,
            ident.covariance,
        )
        headroom_mean, headroom_std = propagate_uncertainty(
            lambda p: _pass_metrics(p)["headroom"],
            base_params,
            ident.covariance,
        )
        output["identifiability"] = {
            "params": ident.params,
            "covariance": ident.covariance,
            "condition_number": ident.condition_number,
            "is_degenerate": ident.is_degenerate,
            "warnings": ident.warnings,
        }
        output["uncertainty"] = {
            "method": "fim_linear",
            "metrics": {
                "key_rate_bps": {
                    "mean": key_rate_mean,
                    "ci_low": key_rate_mean - 1.96 * key_rate_std,
                    "ci_high": key_rate_mean + 1.96 * key_rate_std,
                },
                "headroom": {
                    "mean": headroom_mean,
                    "ci_low": headroom_mean - 1.96 * headroom_std,
                    "ci_high": headroom_mean + 1.96 * headroom_std,
                },
                "car": None,
            },
        }

    output_path = outdir / "reports" / "forecast_blinded.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    if unblind:
        output_unblinded = dict(output)
        output_unblinded["analysis"] = {
            "group_labels_included": True,
            "labels_by_window": labels_by_window,
            "group_stats": _group_stats(labels_by_window, outcomes_by_window),
        }
        output_unblinded_path = outdir / "reports" / "forecast_unblinded.json"
        with open(output_unblinded_path, "w") as f:
            json.dump(output_unblinded, f, indent=2)
            f.write("\n")

    return output
