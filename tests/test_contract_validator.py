import pytest

from sat_qkd_lab.contract import (
    CONTRACT_CHECKS,
    ContractError,
    validate_calibration_record,
    validate_latest_report_contract,
)


def _valid_latest_report() -> dict:
    return {
        "schema_version": "0.4",
        "generated_utc": "2026-01-04T12:00:00Z",
        "pass_sweep": {
            "records": [
                {
                    "loss_db": 25.0,
                    "qber": 0.1,
                    "secret_fraction": 0.8,
                    "key_rate_per_pulse": 0.01,
                    "n_secret_est": 8,
                    "n_sifted": 10,
                    "aborted": False,
                }
            ],
            "summary": {
                "secure_window_seconds": 2.0,
                "secure_start_s": 0.0,
                "secure_end_s": 2.0,
                "secure_start_elevation_deg": 10.0,
                "secure_end_elevation_deg": 20.0,
                "peak_key_rate": 0.5,
                "total_secret_bits": 10.0,
                "mean_key_rate_in_window": 0.25,
            },
        },
        "parameters": {
            "pass_duration_s": 10.0,
            "time_step_s": 1.0,
            "pulses": 100,
            "rep_rate": None,
            "eta": 0.2,
            "p_bg": 1e-4,
            "flip_prob": 0.01,
            "min_elevation_deg": 10.0,
            "max_elevation_deg": 20.0,
        },
        "field_classification": {
            "schema_version": "placeholder",
            "generated_utc": "inferred",
            "parameters.pass_duration_s": "placeholder",
            "parameters.time_step_s": "placeholder",
            "parameters.pulses": "placeholder",
            "parameters.rep_rate": "placeholder",
            "parameters.eta": "placeholder",
            "parameters.p_bg": "placeholder",
            "parameters.flip_prob": "placeholder",
            "pass_sweep.records.loss_db": "simulated",
            "pass_sweep.records.qber": "inferred",
            "pass_sweep.records.secret_fraction": "inferred",
            "pass_sweep.records.key_rate_per_pulse": "inferred",
            "pass_sweep.records.n_secret_est": "inferred",
            "pass_sweep.records.aborted": "inferred",
            "pass_sweep.summary.secure_window_seconds": "inferred",
            "pass_sweep.summary.secure_start_s": "inferred",
            "pass_sweep.summary.secure_end_s": "inferred",
            "pass_sweep.summary.secure_start_elevation_deg": "inferred",
            "pass_sweep.summary.secure_end_elevation_deg": "inferred",
            "pass_sweep.summary.peak_key_rate": "inferred",
            "pass_sweep.summary.total_secret_bits": "inferred",
            "pass_sweep.summary.mean_key_rate_in_window": "inferred",
        },
    }


def _valid_calibration_record() -> dict:
    return {
        "calibration_version": "1.0",
        "schema_version": "0.4",
        "generated_utc": "2026-01-04T12:00:00Z",
        "git_commit": "abc1234",
        "seed_policy": "fixed",
        "source": {"type": "empirical"},
        "parameters": {
            "eta": {"value": 0.22},
            "p_bg": {"value": 1.2e-4},
            "flip_prob": {"value": 0.003},
            "p_afterpulse": {"value": 0.01},
            "dead_time_pulses": {"value": 10},
        },
    }


def test_latest_contract_valid() -> None:
    checks = validate_latest_report_contract(_valid_latest_report(), return_checks=True)
    assert checks


def test_latest_contract_missing_field() -> None:
    report = _valid_latest_report()
    del report["parameters"]["pass_duration_s"]
    with pytest.raises(ContractError, match="pass_duration_s"):
        validate_latest_report_contract(report)


def test_latest_contract_range_violation() -> None:
    report = _valid_latest_report()
    report["pass_sweep"]["records"][0]["loss_db"] = 80.0
    with pytest.raises(ContractError, match="loss_db"):
        validate_latest_report_contract(report)


def test_latest_contract_summary_range_violation() -> None:
    report = _valid_latest_report()
    report["pass_sweep"]["summary"]["peak_key_rate"] = 1.5
    with pytest.raises(ContractError, match="peak_key_rate"):
        validate_latest_report_contract(report)


def test_latest_contract_summary_missing_field() -> None:
    report = _valid_latest_report()
    del report["pass_sweep"]["summary"]["secure_start_s"]
    with pytest.raises(ContractError, match="secure_start_s"):
        validate_latest_report_contract(report)


def test_latest_contract_qber_nan_rules() -> None:
    report = _valid_latest_report()
    report["pass_sweep"]["records"][0]["qber"] = float("nan")
    report["pass_sweep"]["records"][0]["n_sifted"] = 0
    report["pass_sweep"]["records"][0]["n_secret_est"] = 0
    validate_latest_report_contract(report)

    report = _valid_latest_report()
    report["pass_sweep"]["records"][0]["qber"] = float("nan")
    report["pass_sweep"]["records"][0]["n_sifted"] = 1
    with pytest.raises(ContractError, match="qber"):
        validate_latest_report_contract(report)


def test_calibration_contract_valid() -> None:
    validate_calibration_record(_valid_calibration_record(), expected_schema_version="0.4")


def test_calibration_contract_range_violation() -> None:
    record = _valid_calibration_record()
    record["parameters"]["eta"] = {"value": 0.99}
    with pytest.raises(ContractError, match="eta"):
        validate_calibration_record(record, expected_schema_version="0.4")


def test_calibration_contract_optional_field_types() -> None:
    record = _valid_calibration_record()
    record["source"]["dataset_path"] = 123
    with pytest.raises(ContractError, match="dataset_path"):
        validate_calibration_record(record, expected_schema_version="0.4")

    record = _valid_calibration_record()
    record["fit_method"] = "grid"
    with pytest.raises(ContractError, match="fit_method"):
        validate_calibration_record(record, expected_schema_version="0.4")

    record = _valid_calibration_record()
    record["notes"] = {"note": "bad type"}
    with pytest.raises(ContractError, match="notes"):
        validate_calibration_record(record, expected_schema_version="0.4")


def test_latest_contract_classification_tag_mismatch() -> None:
    report = _valid_latest_report()
    report["field_classification"]["pass_sweep.summary.total_secret_bits"] = "invalid"
    with pytest.raises(ContractError, match="field_classification"):
        validate_latest_report_contract(report)


def test_contract_coverage_checklist() -> None:
    expected = {
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
    }
    missing = expected.difference(set(CONTRACT_CHECKS))
    assert not missing
