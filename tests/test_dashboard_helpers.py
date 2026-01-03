from sat_qkd_lab.dashboard import _ensure_outdir, _plot_index, _read_json, _write_latest_dashboard


def test_dashboard_helpers(tmp_path):
    outdir = _ensure_outdir(str(tmp_path))
    assert (outdir / "figures").exists()
    assert (outdir / "reports").exists()
    assert _read_json(outdir / "missing.json") is None
    plot_index = _plot_index(outdir)
    assert "car_vs_loss" in plot_index
    assert plot_index["car_vs_loss"] is None
    dashboard = _write_latest_dashboard(outdir, last_action="sweep")
    dashboard_path = outdir / "reports" / "latest_dashboard.json"
    assert dashboard_path.exists()
    assert dashboard["last_action"] == "sweep"
