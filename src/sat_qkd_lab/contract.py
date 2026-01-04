"""Validation helpers for hardware output and calibration contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, Set


CONTRACT_CHECKS = (
    "latest.schema_version",
    "latest.generated_utc",
    "latest.parameters.pass_duration_s",
    "latest.parameters.time_step_s",
    "latest.parameters.pulses",
    "latest.parameters.rep_rate",
    "latest.parameters.eta",
    "latest.parameters.p_bg",
    "latest.parameters.flip_prob",
    "latest.pass_sweep.records.loss_db",
    "latest.pass_sweep.records.qber",
    "latest.pass_sweep.records.secret_fraction",
    "latest.pass_sweep.records.key_rate_per_pulse",
    "latest.pass_sweep.records.n_secret_est",
    "latest.pass_sweep.records.aborted",
    "latest.pass_sweep.summary.secure_window_seconds",
    "latest.pass_sweep.summary.secure_start_s",
    "latest.pass_sweep.summary.secure_end_s",
    "latest.pass_sweep.summary.secure_start_elevation_deg",
    "latest.pass_sweep.summary.secure_end_elevation_deg",
    "latest.pass_sweep.summary.peak_key_rate",
    "latest.pass_sweep.summary.total_secret_bits",
    "latest.pass_sweep.summary.mean_key_rate_in_window",
    "latest.field_classification",
    "calibration.calibration_version",
    "calibration.schema_version",
    "calibration.generated_utc",
    "calibration.git_commit",
    "calibration.seed_policy",
    "calibration.source.type",
    "calibration.source.dataset_path",
    "calibration.source.dataset_hash_sha256",
    "calibration.fit_method",
    "calibration.fit_quality",
    "calibration.notes",
    "calibration.parameters.eta",
    "calibration.parameters.p_bg",
    "calibration.parameters.flip_prob",
    "calibration.parameters.p_afterpulse",
    "calibration.parameters.dead_time_pulses",
)


class ContractError(ValueError):
    """Raised when a contract assertion fails."""


def _require(condition: bool, check_id: str, message: str, checks: Set[str]) -> None:
    checks.add(check_id)
    if not condition:
        raise ContractError(message)


def _is_utc_iso8601(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo == timezone.utc


def _as_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{field} must be a float, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    raise ContractError(f"{field} must be a float")


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{field} must be an int, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ContractError(f"{field} must be an int")


def _get_param_value(param: Any, field: str) -> float:
    if isinstance(param, dict):
        if "value" not in param:
            raise ContractError(f"{field} must include value")
        return _as_float(param["value"], field)
    return _as_float(param, field)


def validate_latest_report_contract(report: Dict[str, Any], *, return_checks: bool = False) -> Set[str] | None:
    """Validate reports/latest.json against specs/contracts/hardware_outputs.md."""
    checks: Set[str] = set()
    _require(
        report.get("schema_version") == "0.4",
        "latest.schema_version",
        "schema_version must be '0.4'",
        checks,
    )
    _require(
        _is_utc_iso8601(report.get("generated_utc")),
        "latest.generated_utc",
        "generated_utc must be a valid UTC ISO 8601 timestamp",
        checks,
    )

    parameters = report.get("parameters")
    if not isinstance(parameters, dict):
        raise ContractError("parameters must be an object")
    pass_duration = _as_float(parameters.get("pass_duration_s"), "parameters.pass_duration_s")
    _require(pass_duration >= 0.0, "latest.parameters.pass_duration_s", "pass_duration_s must be >= 0", checks)
    time_step = _as_float(parameters.get("time_step_s"), "parameters.time_step_s")
    _require(time_step > 0.0, "latest.parameters.time_step_s", "time_step_s must be > 0", checks)
    pulses = _as_int(parameters.get("pulses"), "parameters.pulses")
    _require(pulses >= 1, "latest.parameters.pulses", "pulses must be >= 1", checks)

    rep_rate = parameters.get("rep_rate")
    if rep_rate is not None:
        rep_rate_val = _as_float(rep_rate, "parameters.rep_rate")
        _require(rep_rate_val >= 0.0, "latest.parameters.rep_rate", "rep_rate must be >= 0", checks)
    else:
        checks.add("latest.parameters.rep_rate")

    eta = _as_float(parameters.get("eta"), "parameters.eta")
    _require(0.0 <= eta <= 1.0, "latest.parameters.eta", "eta must be in [0, 1]", checks)
    p_bg = _as_float(parameters.get("p_bg"), "parameters.p_bg")
    _require(0.0 <= p_bg <= 1.0, "latest.parameters.p_bg", "p_bg must be in [0, 1]", checks)
    flip_prob = _as_float(parameters.get("flip_prob"), "parameters.flip_prob")
    _require(0.0 <= flip_prob <= 0.5, "latest.parameters.flip_prob", "flip_prob must be in [0, 0.5]", checks)
    min_elevation = _as_float(parameters.get("min_elevation_deg"), "parameters.min_elevation_deg")
    max_elevation = _as_float(parameters.get("max_elevation_deg"), "parameters.max_elevation_deg")
    _require(0.0 <= min_elevation <= 90.0, "latest.parameters.min_elevation_deg", "min_elevation_deg must be in [0, 90]", checks)
    _require(0.0 <= max_elevation <= 90.0, "latest.parameters.max_elevation_deg", "max_elevation_deg must be in [0, 90]", checks)
    _require(min_elevation <= max_elevation, "latest.parameters.min_elevation_deg", "min_elevation_deg must be <= max_elevation_deg", checks)

    pass_sweep = report.get("pass_sweep")
    if not isinstance(pass_sweep, dict):
        raise ContractError("pass_sweep must be an object")
    records = pass_sweep.get("records")
    if not isinstance(records, list) or not records:
        raise ContractError("pass_sweep.records must be a non-empty list")

    classification = report.get("field_classification")
    _require(isinstance(classification, dict), "latest.field_classification", "field_classification must be an object", checks)
    required_classifications = (
        "schema_version",
        "generated_utc",
        "parameters.pass_duration_s",
        "parameters.time_step_s",
        "parameters.pulses",
        "parameters.rep_rate",
        "parameters.eta",
        "parameters.p_bg",
        "parameters.flip_prob",
        "pass_sweep.records.loss_db",
        "pass_sweep.records.qber",
        "pass_sweep.records.secret_fraction",
        "pass_sweep.records.key_rate_per_pulse",
        "pass_sweep.records.n_secret_est",
        "pass_sweep.records.aborted",
        "pass_sweep.summary.secure_window_seconds",
        "pass_sweep.summary.secure_start_s",
        "pass_sweep.summary.secure_end_s",
        "pass_sweep.summary.secure_start_elevation_deg",
        "pass_sweep.summary.secure_end_elevation_deg",
        "pass_sweep.summary.peak_key_rate",
        "pass_sweep.summary.total_secret_bits",
        "pass_sweep.summary.mean_key_rate_in_window",
    )
    allowed_tags = {"simulated", "inferred", "measured", "placeholder"}
    for key in required_classifications:
        tag = classification.get(key)
        if tag not in allowed_tags:
            raise ContractError(f"field_classification.{key} must be one of {sorted(allowed_tags)}")

    for record in records:
        if not isinstance(record, dict):
            raise ContractError("pass_sweep.records entries must be objects")
        loss_db = _as_float(record.get("loss_db"), "pass_sweep.records.loss_db")
        _require(0.0 <= loss_db <= 50.0, "latest.pass_sweep.records.loss_db", "loss_db must be in [0, ~50]", checks)
        qber = record.get("qber")
        n_sifted = _as_int(record.get("n_sifted"), "pass_sweep.records.n_sifted")
        if isinstance(qber, float) and math.isnan(qber):
            if n_sifted != 0:
                raise ContractError("qber can be NaN only when n_sifted == 0")
        else:
            qber_val = _as_float(qber, "pass_sweep.records.qber")
            _require(0.0 <= qber_val <= 0.5, "latest.pass_sweep.records.qber", "qber must be in [0, 0.5]", checks)
        secret_fraction = _as_float(record.get("secret_fraction"), "pass_sweep.records.secret_fraction")
        _require(0.0 <= secret_fraction <= 1.0, "latest.pass_sweep.records.secret_fraction", "secret_fraction must be in [0, 1]", checks)
        key_rate = _as_float(record.get("key_rate_per_pulse"), "pass_sweep.records.key_rate_per_pulse")
        _require(0.0 <= key_rate <= 1.0, "latest.pass_sweep.records.key_rate_per_pulse", "key_rate_per_pulse must be in [0, 1]", checks)
        n_secret = _as_int(record.get("n_secret_est"), "pass_sweep.records.n_secret_est")
        _require(n_secret >= 0, "latest.pass_sweep.records.n_secret_est", "n_secret_est must be >= 0", checks)
        _require(
            n_secret <= n_sifted,
            "latest.pass_sweep.records.n_secret_est",
            "n_secret_est must be <= n_sifted",
            checks,
        )
        aborted = record.get("aborted")
        _require(isinstance(aborted, bool), "latest.pass_sweep.records.aborted", "aborted must be boolean", checks)

    summary = pass_sweep.get("summary")
    if not isinstance(summary, dict):
        raise ContractError("pass_sweep.summary must be an object")
    secure_window_seconds = _as_float(
        summary.get("secure_window_seconds"),
        "pass_sweep.summary.secure_window_seconds",
    )
    _require(
        0.0 <= secure_window_seconds <= pass_duration,
        "latest.pass_sweep.summary.secure_window_seconds",
        "secure_window_seconds must be in [0, pass_duration_s]",
        checks,
    )
    secure_start_s = _as_float(summary.get("secure_start_s"), "pass_sweep.summary.secure_start_s")
    _require(
        0.0 <= secure_start_s <= pass_duration,
        "latest.pass_sweep.summary.secure_start_s",
        "secure_start_s must be in [0, pass_duration_s]",
        checks,
    )
    secure_end_s = _as_float(summary.get("secure_end_s"), "pass_sweep.summary.secure_end_s")
    _require(
        0.0 <= secure_end_s <= pass_duration,
        "latest.pass_sweep.summary.secure_end_s",
        "secure_end_s must be in [0, pass_duration_s]",
        checks,
    )
    secure_start_elev = _as_float(
        summary.get("secure_start_elevation_deg"),
        "pass_sweep.summary.secure_start_elevation_deg",
    )
    _require(
        min_elevation <= secure_start_elev <= max_elevation,
        "latest.pass_sweep.summary.secure_start_elevation_deg",
        "secure_start_elevation_deg must be in [min_elevation, max_elevation]",
        checks,
    )
    secure_end_elev = _as_float(
        summary.get("secure_end_elevation_deg"),
        "pass_sweep.summary.secure_end_elevation_deg",
    )
    _require(
        min_elevation <= secure_end_elev <= max_elevation,
        "latest.pass_sweep.summary.secure_end_elevation_deg",
        "secure_end_elevation_deg must be in [min_elevation, max_elevation]",
        checks,
    )
    peak_key_rate = _as_float(summary.get("peak_key_rate"), "pass_sweep.summary.peak_key_rate")
    _require(
        0.0 <= peak_key_rate <= 1.0,
        "latest.pass_sweep.summary.peak_key_rate",
        "peak_key_rate must be in [0, 1]",
        checks,
    )
    total_secret_bits = _as_float(summary.get("total_secret_bits"), "pass_sweep.summary.total_secret_bits")
    _require(
        total_secret_bits >= 0.0,
        "latest.pass_sweep.summary.total_secret_bits",
        "total_secret_bits must be >= 0",
        checks,
    )
    mean_key_rate = _as_float(
        summary.get("mean_key_rate_in_window"),
        "pass_sweep.summary.mean_key_rate_in_window",
    )
    _require(
        0.0 <= mean_key_rate <= 1.0,
        "latest.pass_sweep.summary.mean_key_rate_in_window",
        "mean_key_rate_in_window must be in [0, 1]",
        checks,
    )

    return checks if return_checks else None


def validate_calibration_record(
    record: Dict[str, Any],
    *,
    expected_schema_version: str,
    return_checks: bool = False,
) -> Set[str] | None:
    """Validate calibration record against specs/contracts/calibration_hooks.md."""
    checks: Set[str] = set()
    _require(
        isinstance(record.get("calibration_version"), str),
        "calibration.calibration_version",
        "calibration_version must be a string",
        checks,
    )
    _require(
        record.get("schema_version") == expected_schema_version,
        "calibration.schema_version",
        "schema_version must match simulator schema_version",
        checks,
    )
    _require(
        _is_utc_iso8601(record.get("generated_utc")),
        "calibration.generated_utc",
        "generated_utc must be a valid UTC ISO 8601 timestamp",
        checks,
    )
    git_commit = record.get("git_commit")
    _require(
        isinstance(git_commit, str) and 4 <= len(git_commit) <= 12,
        "calibration.git_commit",
        "git_commit must be a short hash",
        checks,
    )
    seed_policy = record.get("seed_policy")
    _require(
        seed_policy in {"fixed", "random", "derived"},
        "calibration.seed_policy",
        "seed_policy must be one of fixed/random/derived",
        checks,
    )
    source = record.get("source")
    if not isinstance(source, dict):
        raise ContractError("source must be an object")
    _require(
        source.get("type") in {"empirical", "synthetic", "prior"},
        "calibration.source.type",
        "source.type must be empirical/synthetic/prior",
        checks,
    )
    if "dataset_path" in source:
        _require(
            isinstance(source.get("dataset_path"), str),
            "calibration.source.dataset_path",
            "source.dataset_path must be a string",
            checks,
        )
    else:
        checks.add("calibration.source.dataset_path")
    if "dataset_hash_sha256" in source:
        _require(
            isinstance(source.get("dataset_hash_sha256"), str),
            "calibration.source.dataset_hash_sha256",
            "source.dataset_hash_sha256 must be a string",
            checks,
        )
    else:
        checks.add("calibration.source.dataset_hash_sha256")

    if "fit_method" in record:
        _require(
            isinstance(record.get("fit_method"), dict),
            "calibration.fit_method",
            "fit_method must be an object",
            checks,
        )
    else:
        checks.add("calibration.fit_method")
    if "fit_quality" in record:
        _require(
            isinstance(record.get("fit_quality"), dict),
            "calibration.fit_quality",
            "fit_quality must be an object",
            checks,
        )
    else:
        checks.add("calibration.fit_quality")
    if "notes" in record:
        _require(
            isinstance(record.get("notes"), str),
            "calibration.notes",
            "notes must be a string",
            checks,
        )
    else:
        checks.add("calibration.notes")

    parameters = record.get("parameters")
    if not isinstance(parameters, dict):
        raise ContractError("parameters must be an object")

    bounds = {
        "eta": (0.05, 0.95),
        "p_bg": (1e-6, 1e-2),
        "flip_prob": (0.0, 0.05),
        "p_afterpulse": (0.0, 0.10),
        "dead_time_pulses": (0.0, 100.0),
    }
    for key, (min_val, max_val) in bounds.items():
        if key in parameters:
            value = _get_param_value(parameters[key], f"parameters.{key}")
            check_id = f"calibration.parameters.{key}"
            _require(
                min_val <= value <= max_val,
                check_id,
                f"{key} must be in [{min_val}, {max_val}]",
                checks,
            )
        else:
            checks.add(f"calibration.parameters.{key}")

    return checks if return_checks else None
