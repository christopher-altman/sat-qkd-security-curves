import subprocess


def test_sweep_cli_summary_simulated(tmp_path) -> None:
    result = subprocess.run(
        [
            "./py",
            "-m",
            "sat_qkd_lab.run",
            "sweep",
            "--loss-min",
            "5",
            "--loss-max",
            "6",
            "--steps",
            "2",
            "--pulses",
            "1000",
            "--trials",
            "1",
            "--seed",
            "1",
            "--outdir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    stdout = result.stdout
    assert "simulated sweep summary" in stdout
    assert "loss range:" in stdout
    assert "pulses:" in stdout
    assert "trials:" in stdout
    assert "seed policy:" in stdout
    assert "qber_mean" in stdout
    assert "key_rate_per_pulse" in stdout
    assert "measured" not in stdout.lower()
    assert "observed" not in stdout.lower()
    wrote_index = stdout.find("Wrote:")
    summary_index = stdout.find("simulated sweep summary")
    assert wrote_index != -1
    assert summary_index > wrote_index
