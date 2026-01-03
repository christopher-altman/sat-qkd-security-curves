import json
from pathlib import Path

from sat_qkd_lab.hil_adapters import ingest_timetag_file, playback_pass


def _write_tags(path: Path):
    tags = [
        {"time_s": 0.0, "detector": "A", "is_signal": True, "basis": 0, "bit": 0},
        {"time_s": 0.0, "detector": "B", "is_signal": True, "basis": 0, "bit": 0},
        {"time_s": 1.0, "detector": "A", "is_signal": True, "basis": 0, "bit": 1},
        {"time_s": 1.0, "detector": "B", "is_signal": True, "basis": 0, "bit": 0},
        {"time_s": 2.0, "detector": "A", "is_signal": False, "basis": 0, "bit": 1},
        {"time_s": 2.0, "detector": "B", "is_signal": True, "basis": 0, "bit": 1},
    ]
    path.write_text(json.dumps(tags))


def test_ingest_and_playback(tmp_path: Path):
    path = tmp_path / "tags.json"
    _write_tags(path)
    tags_a, tags_b = ingest_timetag_file(str(path))
    result = playback_pass(tags_a, tags_b, tau_seconds=1e-9)
    assert result["coincidences"] == 2
    assert result["accidentals"] == 1
    assert result["car"] == 2.0
    assert abs(result["qber_mean"] - (1.0 / 3.0)) < 1e-6
