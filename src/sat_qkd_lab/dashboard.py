"""
Optional Streamlit dashboard for running simulations and viewing outputs.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import json

from sat_qkd_lab.detector import DetectorParams, DEFAULT_DETECTOR
from sat_qkd_lab.sweep import sweep_loss
from sat_qkd_lab.plotting import plot_key_metrics_vs_loss, plot_key_rate_vs_elevation, plot_secure_window
from sat_qkd_lab.pass_model import PassModelParams, compute_pass_records
from sat_qkd_lab.experiment import ExperimentParams, run_experiment
from sat_qkd_lab.forecast_harness import run_forecast_harness

DASHBOARD_PLOTS = {
    "car_vs_loss": "figures/car_vs_loss.png",
    "chsh_s_vs_loss": "figures/chsh_s_vs_loss.png",
    "basis_bias_vs_elevation": "figures/basis_bias_vs_elevation.png",
    "clock_sync_diagnostics": "figures/clock_sync_diagnostics.png",
    "sync_lock_state": "figures/sync_lock_state.png",
    "pass_fading_evolution": "figures/pass_fading_evolution.png",
    "background_rate_vs_time": "figures/background_rate_vs_time.png",
    "car_vs_time": "figures/car_vs_time.png",
    "polarization_drift_vs_time": "figures/polarization_drift_vs_time.png",
}


def _ensure_outdir(path: str) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)
    (outdir / "reports").mkdir(exist_ok=True)
    return outdir


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _plot_index(outdir: Path) -> Dict[str, Optional[str]]:
    index: Dict[str, Optional[str]] = {}
    for name, rel_path in DASHBOARD_PLOTS.items():
        path = outdir / rel_path
        index[name] = rel_path if path.exists() else None
    return index


def _build_dashboard_state(
    outdir: Path,
    last_action: Optional[str],
    ui_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "mode": "dashboard",
        "timestamp_utc": _timestamp_utc(),
        "last_action": last_action,
        "plots": _plot_index(outdir),
        "ui": ui_state or {},
    }
    return payload


def _write_latest_dashboard(
    outdir: Path,
    last_action: Optional[str],
    ui_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = _build_dashboard_state(outdir, last_action, ui_state=ui_state)
    _write_json(outdir / "reports" / "latest_dashboard.json", payload)
    return payload


def _gate_unblind_output(output: Dict[str, Any], allow_unblind: bool) -> Dict[str, Any]:
    if allow_unblind:
        return output
    redacted = dict(output)
    if "analysis" in redacted:
        redacted["analysis"] = {"group_labels_included": False}
    return redacted


def run() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is not installed. Install with `pip install .[dashboard]`."
        ) from exc

    st.set_page_config(page_title="sat-qkd-lab", layout="wide")
    st.title("Satellite QKD Lab Control Panel")

    outdir = st.sidebar.text_input("Output directory", value=".")
    outdir_path = _ensure_outdir(outdir)
    ui_unblind = st.sidebar.checkbox("Unblind (explicit)", value=False, key="ui_unblind")
    _write_latest_dashboard(outdir_path, last_action=None, ui_state={"unblind": bool(ui_unblind)})

    tab_sweep, tab_pass, tab_experiment, tab_forecast, tab_ops, tab_instrument, tab_protocol, tab_incidents = st.tabs(
        ["Sweep", "Pass", "Experiment", "Forecast", "Ops", "Instrument", "Protocol", "Incidents"]
    )

    with tab_sweep:
        st.subheader("BB84 Sweep")
        loss_min = st.number_input("Loss min (dB)", value=20.0, key="sweep_loss_min")
        loss_max = st.number_input("Loss max (dB)", value=60.0, key="sweep_loss_max")
        steps = st.number_input("Steps", value=21, step=1, key="sweep_steps")
        flip_prob = st.number_input("Flip probability", value=0.005, key="sweep_flip_prob")
        pulses = st.number_input("Pulses", value=200000, step=1000, key="sweep_pulses")
        seed = st.number_input("Seed", value=0, step=1, key="sweep_seed")
        eta = st.number_input("Detector efficiency", value=float(DEFAULT_DETECTOR.eta), key="sweep_eta")
        p_bg = st.number_input("Background probability", value=float(DEFAULT_DETECTOR.p_bg), key="sweep_p_bg")

        if st.button("Run sweep"):
            det = DetectorParams(eta=float(eta), p_bg=float(p_bg))
            loss_vals = [loss_min + i * (loss_max - loss_min) / (steps - 1) for i in range(int(steps))]
            records_no = sweep_loss(loss_vals, flip_prob=float(flip_prob), n_pulses=int(pulses), seed=int(seed), detector=det)
            records_attack = sweep_loss(loss_vals, flip_prob=float(flip_prob), n_pulses=int(pulses), seed=int(seed), detector=det, attack="intercept_resend")
            prefix = str(outdir_path / "figures" / "sweep")
            q_path, k_path = plot_key_metrics_vs_loss(records_no, records_attack, prefix)
            report = {
                "mode": "dashboard-sweep",
                "records_no_attack": records_no,
                "records_attack": records_attack,
                "artifacts": {
                    "qber_plot": q_path,
                    "key_fraction_plot": k_path,
                },
            }
            _write_json(outdir_path / "reports" / "dashboard_sweep.json", report)
            _write_latest_dashboard(outdir_path, last_action="sweep", ui_state={"unblind": bool(ui_unblind)})
            st.success("Sweep complete.")
            st.image(q_path)
            st.image(k_path)
            st.json(report)

    with tab_pass:
        st.subheader("Pass Sweep")
        max_elevation = st.number_input("Max elevation (deg)", value=60.0, key="pass_max_elev")
        min_elevation = st.number_input("Min elevation (deg)", value=10.0, key="pass_min_elev")
        pass_seconds = st.number_input("Pass seconds", value=300.0, key="pass_seconds")
        dt_seconds = st.number_input("Time step (s)", value=5.0, key="pass_dt")
        rep_rate = st.number_input("Rep rate (Hz)", value=1e8, key="pass_rep_rate")

        if st.button("Run pass sweep"):
            params = PassModelParams(
                max_elevation_deg=float(max_elevation),
                min_elevation_deg=float(min_elevation),
                pass_seconds=float(pass_seconds),
                dt_seconds=float(dt_seconds),
                rep_rate_hz=float(rep_rate),
            )
            records, summary = compute_pass_records(params=params)
            elev_plot = plot_key_rate_vs_elevation(records, str(outdir_path / "figures" / "key_rate_vs_elevation.png"))
            secure_plot = plot_secure_window(records, str(outdir_path / "figures" / "secure_window.png"),
                                             secure_start_s=summary["secure_window"]["t_start_seconds"],
                                             secure_end_s=summary["secure_window"]["t_end_seconds"])
            report = {
                "mode": "dashboard-pass",
                "params": asdict(params),
                "summary": summary,
                "artifacts": {
                    "key_rate_vs_elevation": elev_plot,
                    "secure_window": secure_plot,
                },
            }
            _write_json(outdir_path / "reports" / "dashboard_pass.json", report)
            _write_latest_dashboard(outdir_path, last_action="pass", ui_state={"unblind": bool(ui_unblind)})
            st.success("Pass sweep complete.")
            st.image(elev_plot)
            st.image(secure_plot)
            st.json(report)

    with tab_experiment:
        st.subheader("Blinded Experiment")
        n_blocks = st.number_input("Blocks", value=20, step=1, key="exp_n_blocks")
        block_seconds = st.number_input("Block seconds", value=30.0, key="exp_block_seconds")
        rep_rate_hz = st.number_input("Rep rate (Hz)", value=1e8, key="exp_rep_rate")
        seed = st.number_input("Seed", value=0, step=1, key="exp_seed")
        unblind = st.checkbox("Unblind (explicit)", value=bool(ui_unblind), key="exp_unblind")

        if st.button("Run experiment"):
            params = ExperimentParams(
                seed=int(seed),
                n_blocks=int(n_blocks),
                block_seconds=float(block_seconds),
                rep_rate_hz=float(rep_rate_hz),
                pass_seconds=float(block_seconds) * int(n_blocks),
            )
            output = run_experiment(params, ["qber_mean", "headroom", "total_secret_bits"], outdir_path, None, unblind=bool(unblind))
            _write_latest_dashboard(outdir_path, last_action="experiment", ui_state={"unblind": bool(ui_unblind)})
            st.success("Experiment complete.")
            st.json(_gate_unblind_output(output, unblind))

    with tab_forecast:
        st.subheader("Forecast Scoring")
        forecasts_path = st.text_input("Forecasts path", value="forecasts.json", key="forecast_path")
        n_blocks = st.number_input("Blocks ", value=20, step=1, key="forecast_n_blocks")
        block_seconds = st.number_input("Block seconds ", value=30.0, key="forecast_block_seconds")
        rep_rate_hz = st.number_input("Rep rate (Hz) ", value=1e8, key="forecast_rep_rate")
        seed = st.number_input("Seed ", value=0, step=1, key="forecast_seed")
        unblind = st.checkbox("Unblind forecast output", value=bool(ui_unblind), key="forecast_unblind")

        if st.button("Run forecast"):
            output = run_forecast_harness(
                forecasts_path=forecasts_path,
                outdir=outdir_path,
                seed=int(seed),
                n_blocks=int(n_blocks),
                block_seconds=float(block_seconds),
                rep_rate_hz=float(rep_rate_hz),
                unblind=bool(unblind),
            )
            _write_latest_dashboard(outdir_path, last_action="forecast", ui_state={"unblind": bool(ui_unblind)})
            st.success("Forecast scoring complete.")
            st.json(_gate_unblind_output(output, unblind))

    with tab_ops:
        st.subheader("Ops")
        inventory = _read_json(outdir_path / "reports" / "constellation_inventory.json")
        if inventory:
            st.write("Constellation inventory")
            st.line_chart(inventory.get("inventory", {}).get("inventory_bits", []))
            if inventory.get("schedule"):
                st.dataframe(inventory["schedule"])
        else:
            st.write("No constellation inventory report found.")

    with tab_instrument:
        st.subheader("Instrument")
        sync_report = _read_json(outdir_path / "reports" / "latest_clock_sync.json")
        if sync_report:
            st.write("Sync lock state")
            sync_plot = outdir_path / "figures" / "sync_lock_state.png"
            if sync_plot.exists():
                st.image(str(sync_plot))
        pass_report = _read_json(outdir_path / "reports" / "latest_pass.json")
        if pass_report and "time_series" in pass_report:
            series = pass_report["time_series"]
            if "background_rate_hz" in series:
                st.line_chart(series["background_rate_hz"])
            if "polarization_angle_deg" in series:
                st.line_chart(series["polarization_angle_deg"])
            if "polarization_angle_comp_deg" in series:
                st.line_chart(series["polarization_angle_comp_deg"])

    with tab_protocol:
        st.subheader("Protocol")
        forecast = _read_json(outdir_path / "reports" / "forecast_blinded.json")
        if forecast:
            st.write("Forecast scoring (blinded)")
            st.dataframe(forecast.get("scores", []))
            if ui_unblind:
                forecast_unblind = _read_json(outdir_path / "reports" / "forecast_unblinded.json")
                if forecast_unblind:
                    st.write("Unblinded analysis")
                    st.json(forecast_unblind.get("analysis", {}))
        else:
            st.write("No forecast report found.")

    with tab_incidents:
        st.subheader("Incidents")
        pass_report = _read_json(outdir_path / "reports" / "latest_pass.json")
        if pass_report and pass_report.get("incidents"):
            st.dataframe(pass_report["incidents"])
        else:
            st.write("No incident cards available.")

    with st.expander("Plot index", expanded=False):
        plot_index = _plot_index(outdir_path)
        for name, rel_path in plot_index.items():
            if rel_path is None:
                st.write(f"{name}: missing")
                continue
            st.write(f"{name}: {rel_path}")
            image_path = outdir_path / rel_path
            if image_path.exists():
                st.image(str(image_path))


def main() -> None:
    run()


if __name__ == "__main__":
    main()
