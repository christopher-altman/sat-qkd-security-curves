from sat_qkd_lab.dashboard import _ensure_outdir, _read_json


def test_dashboard_helpers(tmp_path):
    outdir = _ensure_outdir(str(tmp_path))
    assert (outdir / "figures").exists()
    assert (outdir / "reports").exists()
    assert _read_json(outdir / "missing.json") is None
