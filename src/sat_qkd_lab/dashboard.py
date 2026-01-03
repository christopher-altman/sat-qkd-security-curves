"""
Optional Streamlit dashboard for running simulations and viewing outputs.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional
import json

from .detector import DetectorParams, DEFAULT_DETECTOR
from .sweep import sweep_loss
from .plotting import plot_key_metrics_vs_loss, plot_key_rate_vs_elevation, plot_secure_window
from .pass_model import PassModelParams, compute_pass_records
from .experiment import ExperimentParams, run_experiment
from .forecast_harness import run_forecast_harness


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

    tab_sweep, tab_pass, tab_experiment, tab_forecast = st.tabs(
        ["Sweep", "Pass", "Experiment", "Forecast"]
    )

    with tab_sweep:
        st.subheader("BB84 Sweep")
        loss_min = st.number_input("Loss min (dB)", value=20.0)
        loss_max = st.number_input("Loss max (dB)", value=60.0)
        steps = st.number_input("Steps", value=21, step=1)
        flip_prob = st.number_input("Flip probability", value=0.005)
        pulses = st.number_input("Pulses", value=200000, step=1000)
        seed = st.number_input("Seed", value=0, step=1)
        eta = st.number_input("Detector efficiency", value=float(DEFAULT_DETECTOR.eta))
        p_bg = st.number_input("Background probability", value=float(DEFAULT_DETECTOR.p_bg))

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
            st.success("Sweep complete.")
            st.image(q_path)
            st.image(k_path)
            st.json(report)

    with tab_pass:
        st.subheader("Pass Sweep")
        max_elevation = st.number_input("Max elevation (deg)", value=60.0)
        min_elevation = st.number_input("Min elevation (deg)", value=10.0)
        pass_seconds = st.number_input("Pass seconds", value=300.0)
        dt_seconds = st.number_input("Time step (s)", value=5.0)
        rep_rate = st.number_input("Rep rate (Hz)", value=1e8)

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
            st.success("Pass sweep complete.")
            st.image(elev_plot)
            st.image(secure_plot)
            st.json(report)

    with tab_experiment:
        st.subheader("Blinded Experiment")
        n_blocks = st.number_input("Blocks", value=20, step=1)
        block_seconds = st.number_input("Block seconds", value=30.0)
        rep_rate_hz = st.number_input("Rep rate (Hz)", value=1e8)
        seed = st.number_input("Seed", value=0, step=1)
        unblind = st.checkbox("Unblind (explicit)", value=False)

        if st.button("Run experiment"):
            params = ExperimentParams(
                seed=int(seed),
                n_blocks=int(n_blocks),
                block_seconds=float(block_seconds),
                rep_rate_hz=float(rep_rate_hz),
                pass_seconds=float(block_seconds) * int(n_blocks),
            )
            output = run_experiment(params, ["qber_mean", "headroom", "total_secret_bits"], outdir_path, None, unblind=bool(unblind))
            st.success("Experiment complete.")
            st.json(output)

    with tab_forecast:
        st.subheader("Forecast Scoring")
        forecasts_path = st.text_input("Forecasts path", value="forecasts.json")
        n_blocks = st.number_input("Blocks ", value=20, step=1)
        block_seconds = st.number_input("Block seconds ", value=30.0)
        rep_rate_hz = st.number_input("Rep rate (Hz) ", value=1e8)
        seed = st.number_input("Seed ", value=0, step=1)
        unblind = st.checkbox("Unblind forecast output", value=False)

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
            st.success("Forecast scoring complete.")
            st.json(output)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
