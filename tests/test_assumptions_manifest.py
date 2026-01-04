import json
import subprocess

from sat_qkd_lab.assumptions import build_assumptions_manifest
from sat_qkd_lab.run import SCHEMA_VERSION


def test_manifest_keys_and_serializable() -> None:
    manifest = build_assumptions_manifest(SCHEMA_VERSION)
    for key in (
        "schema_version",
        "schema_version_note",
        "protocol",
        "channel_model",
        "attack_model",
        "key_rate_semantics",
        "disclaimers",
    ):
        assert key in manifest
    json.dumps(manifest)


def test_assumptions_cli_outputs_json() -> None:
    result = subprocess.run(
        ["./py", "-m", "sat_qkd_lab.run", "assumptions"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == SCHEMA_VERSION
