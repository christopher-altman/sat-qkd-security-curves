import subprocess


def test_mission_command_output() -> None:
    result = subprocess.run(
        ["./py", "-m", "sat_qkd_lab.run", "mission"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    stdout = result.stdout.lower()
    assert "simulated" in stdout
    assert "assumptions" in stdout
    assert "security cliff" in stdout
