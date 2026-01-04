import subprocess


def test_main_help():
    """Test that main help runs without error."""
    result = subprocess.run(
        ["./py", "-m", "sat_qkd_lab.run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "sweep" in result.stdout
    assert "pass-sweep" in result.stdout
    assert "eb-sweep" in result.stdout
    assert "cv-sweep" in result.stdout
    assert "replay" in result.stdout
    assert "assumptions" in result.stdout
    assert "mission" in result.stdout


def test_sweep_help():
    """Test that sweep subcommand help includes examples and key flags."""
    result = subprocess.run(
        ["./py", "-m", "sat_qkd_lab.run", "sweep", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--loss-min" in result.stdout
    assert "--loss-max" in result.stdout
    assert "--pulses" in result.stdout
    assert "--finite-key" in result.stdout
    assert "Examples:" in result.stdout
    assert "./py -m sat_qkd_lab.run sweep" in result.stdout
    assert "default:" in result.stdout


def test_pass_sweep_help():
    """Test that pass-sweep subcommand help includes examples and key flags."""
    result = subprocess.run(
        ["./py", "-m", "sat_qkd_lab.run", "pass-sweep", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--max-elevation" in result.stdout
    assert "--pass-duration" in result.stdout
    assert "--pulses" in result.stdout
    assert "Examples:" in result.stdout
    assert "./py -m sat_qkd_lab.run pass-sweep" in result.stdout


def test_eb_sweep_help():
    """Test that eb-sweep subcommand help includes examples and key flags."""
    result = subprocess.run(
        ["./py", "-m", "sat_qkd_lab.run", "eb-sweep", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--loss-min" in result.stdout
    assert "--n-pairs" in result.stdout
    assert "--finite-key" in result.stdout
    assert "Examples:" in result.stdout
    assert "./py -m sat_qkd_lab.run eb-sweep" in result.stdout


def test_cv_sweep_help():
    """Test that cv-sweep subcommand help includes examples and key flags."""
    result = subprocess.run(
        ["./py", "-m", "sat_qkd_lab.run", "cv-sweep", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--loss-min" in result.stdout
    assert "--loss-max" in result.stdout
    assert "--steps" in result.stdout
    assert "Examples:" in result.stdout
    assert "./py -m sat_qkd_lab.run cv-sweep" in result.stdout


def test_replay_help():
    """Test that replay subcommand help includes examples and key flags."""
    result = subprocess.run(
        ["./py", "-m", "sat_qkd_lab.run", "replay", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--report" in result.stdout
    assert "--loss-min" in result.stdout
    assert "Examples:" in result.stdout
    assert "./py -m sat_qkd_lab.run replay" in result.stdout


def test_assumptions_help():
    """Test that assumptions subcommand help includes examples."""
    result = subprocess.run(
        ["./py", "-m", "sat_qkd_lab.run", "assumptions", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Examples:" in result.stdout
    assert "./py -m sat_qkd_lab.run assumptions" in result.stdout


def test_mission_help():
    """Test that mission subcommand help includes examples."""
    result = subprocess.run(
        ["./py", "-m", "sat_qkd_lab.run", "mission", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Examples:" in result.stdout
    assert "./py -m sat_qkd_lab.run mission" in result.stdout
