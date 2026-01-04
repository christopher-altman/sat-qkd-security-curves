import json
import subprocess


def test_replay_preserves_settings(tmp_path) -> None:
    base_outdir = tmp_path / "base"
    replay_outdir = tmp_path / "replay"
    base_outdir.mkdir()
    replay_outdir.mkdir()

    result = subprocess.run(
        [
            "./py",
            "-m",
            "sat_qkd_lab.run",
            "sweep",
            "--loss-min",
            "1",
            "--loss-max",
            "2",
            "--steps",
            "2",
            "--flip-prob",
            "0.01",
            "--pulses",
            "1000",
            "--trials",
            "1",
            "--seed",
            "7",
            "--outdir",
            str(base_outdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0

    base_report_path = base_outdir / "reports" / "latest.json"
    with open(base_report_path, "r") as f:
        base_report = json.load(f)

    replay_result = subprocess.run(
        [
            "./py",
            "-m",
            "sat_qkd_lab.run",
            "replay",
            "--report",
            str(base_report_path),
            "--outdir",
            str(replay_outdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert replay_result.returncode == 0

    replay_report_path = replay_outdir / "reports" / "latest.json"
    with open(replay_report_path, "r") as f:
        replay_report = json.load(f)

    assert replay_report["replay_of"] == str(base_report_path)
    assert replay_report.get("replay_of_git_commit") == base_report.get("git_commit")

    for key in (
        "loss_min",
        "loss_max",
        "steps",
        "flip_prob",
        "pulses",
        "trials",
        "seed",
        "attack",
        "eta",
        "p_bg",
    ):
        assert replay_report["parameters"][key] == base_report["parameters"][key]
