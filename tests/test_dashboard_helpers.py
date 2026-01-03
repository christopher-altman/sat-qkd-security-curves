from sat_qkd_lab.dashboard import (
    _build_dashboard_state,
    _ensure_outdir,
    _gate_unblind_output,
    _plot_index,
    _read_json,
    _write_latest_dashboard,
)


def test_dashboard_helpers(tmp_path):
    outdir = _ensure_outdir(str(tmp_path))
    assert (outdir / "figures").exists()
    assert (outdir / "reports").exists()
    assert _read_json(outdir / "missing.json") is None
    plot_index = _plot_index(outdir)
    assert "car_vs_loss" in plot_index
    assert plot_index["car_vs_loss"] is None
    dashboard = _write_latest_dashboard(outdir, last_action="sweep", ui_state={"unblind": True})
    dashboard_path = outdir / "reports" / "latest_dashboard.json"
    assert dashboard_path.exists()
    assert dashboard["last_action"] == "sweep"
    assert dashboard["ui"]["unblind"] is True


def test_dashboard_state_builder(tmp_path):
    outdir = _ensure_outdir(str(tmp_path))
    state = _build_dashboard_state(outdir, last_action="ops", ui_state={"unblind": False})
    assert state["ui"]["unblind"] is False


def test_dashboard_unblind_gate():
    output = {"analysis": {"group_labels_included": True}}
    redacted = _gate_unblind_output(output, allow_unblind=False)
    assert redacted["analysis"]["group_labels_included"] is False
