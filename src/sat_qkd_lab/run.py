from __future__ import annotations
import argparse
import math
import json
from typing import Any
from datetime import datetime, timezone
import numpy as np
from pathlib import Path

SCHEMA_VERSION = "0.4"

from .helpers import validate_int, validate_float, validate_seed
from .finite_key import FiniteKeyParams
from .sweep import (
    sweep_loss,
    sweep_loss_with_ci,
    sweep_loss_finite_key,
    sweep_finite_key_vs_n_sent,
    sweep_pass,
    compute_summary_stats,
    compute_engineering_outputs,
    compute_headroom,
)
from .plotting import (
    plot_key_metrics_vs_loss,
    plot_qber_vs_loss_ci,
    plot_key_rate_vs_loss_ci,
    plot_decoy_key_rate_vs_loss,
    plot_decoy_key_rate_vs_loss_comparison,
    plot_finite_key_comparison,
    plot_finite_key_bits_vs_loss,
    plot_finite_size_penalty,
    plot_finite_key_rate_vs_n_sent,
    plot_key_rate_vs_elevation,
    plot_secure_window,
    plot_loss_vs_elevation,
    plot_attack_comparison_key_rate,
    plot_qber_headroom_vs_loss,
    plot_car_vs_loss,
    plot_chsh_s_vs_loss,
    plot_visibility_vs_loss,
    plot_inventory_timeseries,
    plot_inventory_flow,
    plot_eta_fading_samples,
    plot_secure_window_impact,
    plot_pointing_lock_state,
    plot_transmittance_with_pointing,
    plot_background_rate_vs_bandwidth,
)
from .free_space_link import FreeSpaceLinkParams, generate_elevation_profile
from .detector import DetectorParams, DEFAULT_DETECTOR
from .decoy_bb84 import DecoyParams, sweep_decoy_loss
from .attacks import Attack, AttackConfig
from .calibration import CalibrationModel
from .telemetry import load_telemetry
from .calibration_fit import (
    fit_telemetry_parameters,
    predict_with_uncertainty,
)
from .pass_model import (
    PassModelParams,
    FadingParams,
    compute_pass_records,
    records_to_time_series,
    sample_fading_factors,
)
from .change_points import detect_change_points, attribute_incidents
from .pointing import PointingParams
from .experiment import ExperimentParams, run_experiment
from .forecast_harness import run_forecast_harness
from .timetags import (
    generate_pair_time_tags,
    generate_background_time_tags,
    merge_time_tags,
    apply_dead_time,
    add_afterpulsing,
)
from .coincidence import match_coincidences
from .eb_observables import compute_observables
from .timing import TimingModel
from .event_stream import StreamParams, generate_event_stream
from .optics import OpticalParams, background_rate_hz, dark_count_rate_hz
from .constellation import schedule_passes, simulate_inventory
from .fading_samples import sample_fading_transmittance, plot_eta_samples


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sat-qkd-security-curves")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- sweep command (original + extensions) ---
    s = sub.add_parser("sweep", help="Sweep loss in dB and generate plots + report.")
    s.add_argument("--loss-min", type=float, default=20.0)
    s.add_argument("--loss-max", type=float, default=60.0)
    s.add_argument("--steps", type=int, default=21)
    s.add_argument("--flip-prob", type=float, default=0.005)
    s.add_argument("--pulses", type=int, default=200_000)
    s.add_argument("--n-sent", type=int, default=None,
                   help="Total pulses sent (overrides --pulses if set)")
    s.add_argument("--rep-rate", type=float, default=None,
                   help="Pulse repetition rate in Hz (requires --pass-seconds)")
    s.add_argument("--rep-rate-hz", type=float, default=None,
                   help="Pulse repetition rate in Hz for engineering outputs (bps calculation)")
    s.add_argument("--pass-seconds", type=float, default=None,
                   help="Pass duration in seconds used with --rep-rate")
    s.add_argument("--target-bits", type=int, default=None,
                   help="Target secret key volume in bits (computes required rep rate)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--outdir", type=str, default=".")
    # New detector parameters
    s.add_argument("--eta", type=float, default=DEFAULT_DETECTOR.eta,
                   help="Detector efficiency (0..1)")
    s.add_argument("--p-bg", type=float, default=DEFAULT_DETECTOR.p_bg,
                   help="Background/dark click probability per pulse")
    s.add_argument("--attack", type=str, default="intercept_resend",
                   choices=["none", "intercept_resend", "pns", "time_shift", "blinding"],
                   help="Attack model for comparison curve")
    s.add_argument("--mu", type=float, default=0.6,
                   help="Mean photon number per pulse for PNS toy model")
    s.add_argument("--timeshift-bias", type=float, default=0.0,
                   help="Time-shift bias toward higher-efficiency basis (0..1)")
    s.add_argument("--blinding-mode", type=str, default="loud",
                   choices=["loud", "stealth"],
                   help="Blinding mode (loud or stealth)")
    s.add_argument("--blinding-prob", type=float, default=0.05,
                   help="Forced click probability for blinding attack")
    s.add_argument("--leakage-fraction", type=float, default=0.0,
                   help="Information leakage fraction (0..1)")
    s.add_argument("--eta-z", type=float, default=None,
                   help="Z-basis detection efficiency (default: --eta)")
    s.add_argument("--eta-x", type=float, default=None,
                   help="X-basis detection efficiency (default: --eta)")
    # Monte Carlo CI parameters
    s.add_argument("--trials", type=int, default=1,
                   help="Number of trials per loss value (>1 enables CI)")
    s.add_argument("--workers", type=int, default=1,
                   help="Number of parallel workers")
    # Finite-key analysis parameters
    s.add_argument("--finite-key", action="store_true",
                   help="Enable finite-key analysis mode")
    s.add_argument("--eps-pe", type=float, default=1e-10,
                   help="Parameter estimation failure probability")
    s.add_argument("--eps-sec", type=float, default=1e-10,
                   help="Secrecy failure probability")
    s.add_argument("--eps-cor", type=float, default=1e-15,
                   help="Correctness failure probability")
    s.add_argument("--ec-efficiency", type=float, default=1.16,
                   help="Error correction efficiency (>= 1.0)")
    s.add_argument("--f-ec", type=float, default=None,
                   help="Error correction inefficiency factor (alias for --ec-efficiency)")
    s.add_argument("--pe-frac", type=float, default=0.5,
                   help="Fraction of sifted bits used for parameter estimation")
    s.add_argument("--m-pe", type=int, default=None,
                   help="Explicit parameter estimation sample size (overrides --pe-frac)")
    s.add_argument("--calibration-file", type=str, default=None,
                   help="Optional JSON file with calibration specs (off by default)")

    # --- decoy-sweep command ---
    d = sub.add_parser("decoy-sweep", help="Decoy-state BB84 sweep over loss.")
    d.add_argument("--loss-min", type=float, default=20.0)
    d.add_argument("--loss-max", type=float, default=60.0)
    d.add_argument("--steps", type=int, default=21)
    d.add_argument("--flip-prob", type=float, default=0.005)
    d.add_argument("--pulses", type=int, default=200_000)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--outdir", type=str, default=".")
    # Detector parameters
    d.add_argument("--eta", type=float, default=DEFAULT_DETECTOR.eta,
                   help="Detector efficiency (0..1)")
    d.add_argument("--p-bg", type=float, default=DEFAULT_DETECTOR.p_bg,
                   help="Background/dark click probability per pulse")
    # Decoy parameters
    d.add_argument("--mu-s", type=float, default=0.6,
                   help="Signal intensity (mean photon number)")
    d.add_argument("--mu-d", type=float, default=0.1,
                   help="Decoy intensity (mean photon number)")
    d.add_argument("--mu-v", type=float, default=0.0,
                   help="Vacuum intensity (must be 0)")
    d.add_argument("--mu-sigma", type=float, default=0.0,
                   help="Std dev of signal intensity noise (default: 0)")
    d.add_argument("--decoy-mu-sigma", type=float, default=0.0,
                   help="Std dev of decoy intensity noise (default: 0)")
    d.add_argument("--p-s", type=float, default=0.8,
                   help="Probability of signal state")
    d.add_argument("--p-d", type=float, default=0.15,
                   help="Probability of decoy state")
    d.add_argument("--p-v", type=float, default=0.05,
                   help="Probability of vacuum state")
    d.add_argument("--trials", type=int, default=1,
                   help="Number of trials per loss value")
    d.add_argument("--afterpulse-prob", type=float, default=0.0,
                   help="Afterpulsing probability per detection (default: 0)")
    d.add_argument("--afterpulse-window", type=int, default=0,
                   help="Afterpulse window length in pulses (default: 0)")
    d.add_argument("--afterpulse-decay", type=float, default=0.0,
                   help="Afterpulse decay constant in pulses (default: 0)")
    d.add_argument("--dead-time-pulses", type=int, default=0,
                   help="Dead time in pulses after a detection (default: 0)")
    d.add_argument("--eta-z", type=float, default=None,
                   help="Z-basis detection efficiency (default: --eta)")
    d.add_argument("--eta-x", type=float, default=None,
                   help="X-basis detection efficiency (default: --eta)")

    # --- attack-sweep command ---
    a = sub.add_parser("attack-sweep", help="Compare multiple attack modes over loss.")
    a.add_argument("--loss-min", type=float, default=20.0)
    a.add_argument("--loss-max", type=float, default=60.0)
    a.add_argument("--steps", type=int, default=15)
    a.add_argument("--flip-prob", type=float, default=0.005)
    a.add_argument("--pulses", type=int, default=50_000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--outdir", type=str, default=".")
    a.add_argument("--eta", type=float, default=DEFAULT_DETECTOR.eta,
                   help="Detector efficiency (0..1)")
    a.add_argument("--p-bg", type=float, default=DEFAULT_DETECTOR.p_bg,
                   help="Background/dark click probability per pulse")
    a.add_argument("--eta-z", type=float, default=None,
                   help="Z-basis detection efficiency (default: --eta)")
    a.add_argument("--eta-x", type=float, default=None,
                   help="X-basis detection efficiency (default: --eta)")
    a.add_argument("--attacks", type=str, nargs="+",
                   default=["none", "intercept_resend", "pns"],
                   choices=["none", "intercept_resend", "pns", "time_shift", "blinding"],
                   help="Attack modes to compare")
    a.add_argument("--mu", type=float, default=0.6,
                   help="Mean photon number per pulse for PNS toy model")
    a.add_argument("--timeshift-bias", type=float, default=0.5,
                   help="Time-shift bias toward higher-efficiency basis (0..1)")
    a.add_argument("--blinding-mode", type=str, default="loud",
                   choices=["loud", "stealth"],
                   help="Blinding mode (loud or stealth)")
    a.add_argument("--blinding-prob", type=float, default=0.05,
                   help="Forced click probability for blinding attack")
    a.add_argument("--leakage-fraction", type=float, default=0.0,
                   help="Information leakage fraction (0..1)")

    # --- pass-sweep command ---
    ps = sub.add_parser("pass-sweep", help="Simulate QKD during a satellite pass over elevation profile.")
    ps.add_argument("--max-elevation", type=float, default=60.0,
                    help="Maximum elevation angle in degrees (default: 60)")
    ps.add_argument("--min-elevation", type=float, default=10.0,
                    help="Minimum elevation angle in degrees (default: 10)")
    ps.add_argument("--pass-duration", type=float, default=300.0,
                    help="Total pass duration in seconds (default: 300)")
    ps.add_argument("--pass-seconds", dest="pass_duration", type=float, default=300.0,
                    help="Alias for --pass-duration")
    ps.add_argument("--time-step", type=float, default=5.0,
                    help="Time resolution in seconds (default: 5)")
    ps.add_argument("--flip-prob", type=float, default=0.005,
                    help="Bit flip probability (default: 0.005)")
    ps_pulses = ps.add_mutually_exclusive_group()
    ps_pulses.add_argument("--pulses", type=int, default=200_000,
                           help="Total pulses sent over the pass (default: 200000)")
    ps_pulses.add_argument("--n-sent", type=int, default=None,
                           help="Total pulses sent over the pass (highest precedence)")
    ps_pulses.add_argument("--rep-rate", type=float, default=None,
                           help="Pulse repetition rate in Hz (total = rate * duration)")
    ps.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducibility (default: 0)")
    ps.add_argument("--outdir", type=str, default=".",
                    help="Output directory for figures/reports (default: .)")
    # Detector parameters
    ps.add_argument("--eta", type=float, default=DEFAULT_DETECTOR.eta,
                    help="Detector efficiency (0..1)")
    ps.add_argument("--p-bg", type=float, default=DEFAULT_DETECTOR.p_bg,
                    help="Background/dark click probability per pulse")
    # Free-space link parameters
    ps.add_argument("--wavelength", type=float, default=850e-9,
                    help="Wavelength in meters (default: 850e-9)")
    ps.add_argument("--tx-diameter", type=float, default=0.30,
                    help="Transmitter aperture diameter in meters (default: 0.30)")
    ps.add_argument("--rx-diameter", type=float, default=1.0,
                    help="Receiver aperture diameter in meters (default: 1.0)")
    ps.add_argument("--sigma-point", type=float, default=2e-6,
                    help="Pointing error std dev in radians (default: 2e-6)")
    ps.add_argument("--altitude", type=float, default=500e3,
                    help="Satellite altitude in meters (default: 500e3)")
    ps.add_argument("--atm-loss-db", type=float, default=0.5,
                    help="Atmospheric loss at zenith in dB (default: 0.5)")
    ps.add_argument("--sigma-ln", type=float, default=0.3,
                    help="Log-normal scintillation parameter (default: 0.3)")
    ps.add_argument("--system-loss-db", type=float, default=3.0,
                    help="Additional system losses in dB (default: 3.0)")
    ps.add_argument("--turbulence", action="store_true",
                    help="Enable turbulence/scintillation modeling")
    ps.add_argument("--is-night", action="store_true", default=True,
                    help="Night-time operation (default: True)")
    ps.add_argument("--day", action="store_true",
                    help="Day-time operation (overrides --is-night)")
    ps.add_argument("--day-bg-factor", type=float, default=100.0,
                    help="Day/night background ratio (default: 100)")
    # Finite-key analysis for pass-sweep
    ps.add_argument("--finite-key", action="store_true",
                    help="Enable finite-key analysis for pass sweep")
    ps.add_argument("--eps-pe", type=float, default=1e-10,
                    help="Parameter estimation failure probability")
    ps.add_argument("--eps-sec", type=float, default=1e-10,
                    help="Secrecy failure probability")
    ps.add_argument("--eps-cor", type=float, default=1e-15,
                    help="Correctness failure probability")
    ps.add_argument("--ec-efficiency", type=float, default=1.16,
                    help="Error correction efficiency (>= 1.0)")
    ps.add_argument("--pe-frac", type=float, default=0.2,
                    help="Fraction of sifted bits used for parameter estimation")
    ps.add_argument("--m-pe", type=int, default=None,
                    help="Explicit parameter estimation sample size (overrides --pe-frac)")
    ps.add_argument("--calibration-file", type=str, default=None,
                    help="Optional JSON file with calibration specs (off by default)")
    # Fading/pointing jitter model for pass-sweep
    ps.add_argument("--fading", action="store_true",
                    help="Enable lognormal fading for transmittance")
    ps.add_argument("--fading-sigma-ln", type=float, default=0.3,
                    help="Lognormal sigma for fading (default: 0.3)")
    ps.add_argument("--fading-samples", type=int, default=50,
                    help="Number of fading samples per time step")
    ps.add_argument("--fading-trials", type=int, default=1,
                    help="Independent fading trials for secure window stats")
    ps.add_argument("--fading-seed", type=int, default=0,
                    help="Seed for fading sampling (default: 0)")
    # Pointing acquisition/track/dropout dynamics
    ps.add_argument("--pointing", action="store_true",
                    help="Enable pointing acquisition/track/dropout model")
    ps.add_argument("--acq-seconds", type=float, default=0.0,
                    help="Acquisition time before lock (seconds)")
    ps.add_argument("--dropout-prob", type=float, default=0.0,
                    help="Dropout probability per second while locked")
    ps.add_argument("--relock-seconds", type=float, default=0.0,
                    help="Recovery time after dropout (seconds)")
    ps.add_argument("--pointing-jitter-urad", type=float, default=2.0,
                    help="Pointing jitter sigma in microradians")
    ps.add_argument("--pointing-seed", type=int, default=0,
                    help="Seed for pointing model")
    # Optical chain parameters
    ps.add_argument("--filter-bandwidth-nm", type=float, default=1.0,
                    help="Optical filter bandwidth in nm")
    ps.add_argument("--detector-temp-c", type=float, default=20.0,
                    help="Detector temperature in C")

    # --- experiment-run command ---
    ex = sub.add_parser("experiment-run", help="Run blinded intervention A/B experiment harness.")
    ex.add_argument("--seed", type=int, default=0)
    ex.add_argument("--n-blocks", type=int, default=20)
    ex.add_argument("--block-seconds", type=float, default=30.0)
    ex.add_argument("--rep-rate-hz", type=float, default=1e8)
    ex.add_argument("--pass-seconds", type=float, default=600.0)
    ex.add_argument("--metrics", type=str, nargs="+",
                    default=["qber_mean", "headroom", "total_secret_bits"])
    ex.add_argument("--finite-key", action="store_true",
                    help="Enable finite-key analysis in experiment metrics")
    ex.add_argument("--eps-sec", type=float, default=1e-10,
                    help="Secrecy failure probability")
    ex.add_argument("--eps-cor", type=float, default=1e-15,
                    help="Correctness failure probability")
    ex.add_argument("--bell-mode", action="store_true",
                    help="Include Bell/visibility observables in outputs")
    ex.add_argument("--unblind", action="store_true",
                    help="Write unblinded schedule and include group analysis")
    ex.add_argument("--outdir", type=str, default=".",
                    help="Output directory for reports (default: .)")

    # --- forecast-run command ---
    fr = sub.add_parser("forecast-run", help="Run forecast ingestion + blinded scoring harness.")
    fr.add_argument("--forecasts", type=str, required=True,
                    help="Path to forecast JSON/CSV file")
    fr.add_argument("--outdir", type=str, default=".",
                    help="Output directory for reports (default: .)")
    fr.add_argument("--seed", type=int, default=0)
    fr.add_argument("--n-blocks", type=int, default=20)
    fr.add_argument("--block-seconds", type=float, default=30.0)
    fr.add_argument("--rep-rate-hz", type=float, default=1e8)
    fr.add_argument("--unblind", action="store_true",
                    help="Write unblinded forecast analysis output")
    fr.add_argument("--estimate-identifiability", action="store_true",
                    help="Estimate calibration identifiability and uncertainty")

    # --- calibration-fit command ---
    cf = sub.add_parser("calibration-fit", help="Fit detector parameters from telemetry.")
    cf.add_argument("--telemetry", type=str, required=True,
                    help="Path to telemetry JSON/CSV")
    cf.add_argument("--eta-base", type=float, default=DEFAULT_DETECTOR.eta,
                    help="Base detector efficiency for fitting")
    cf.add_argument("--p-bg-min", type=float, default=1e-6)
    cf.add_argument("--p-bg-max", type=float, default=5e-3)
    cf.add_argument("--flip-min", type=float, default=0.0)
    cf.add_argument("--flip-max", type=float, default=0.05)
    cf.add_argument("--eta-scale-min", type=float, default=0.5)
    cf.add_argument("--eta-scale-max", type=float, default=1.0)
    cf.add_argument("--grid-steps", type=int, default=12)
    cf.add_argument("--outdir", type=str, default=".",
                    help="Output directory for reports (default: .)")

    # --- constellation-sweep command ---
    csw = sub.add_parser("constellation-sweep", help="Schedule constellation passes and inventory.")
    csw.add_argument("--n-sats", type=int, default=4)
    csw.add_argument("--n-stations", type=int, default=2)
    csw.add_argument("--horizon-hours", type=float, default=24.0)
    csw.add_argument("--passes-per-sat", type=int, default=3)
    csw.add_argument("--pass-duration-s", type=float, default=600.0)
    csw.add_argument("--initial-bits", type=float, default=0.0)
    csw.add_argument("--production-bits-per-pass", type=float, default=5e6)
    csw.add_argument("--consumption-bps", type=float, default=2e5)
    csw.add_argument("--seed", type=int, default=0)
    csw.add_argument("--outdir", type=str, default=".",
                     help="Output directory for reports/figures (default: .)")

    # --- coincidence-sim command ---
    cs = sub.add_parser("coincidence-sim", help="Simulate time-tagged coincidences and CAR vs loss.")
    cs.add_argument("--loss-min", type=float, default=20.0)
    cs.add_argument("--loss-max", type=float, default=60.0)
    cs.add_argument("--steps", type=int, default=9)
    cs.add_argument("--duration", type=float, default=1.0,
                    help="Observation window duration in seconds")
    cs.add_argument("--pair-rate-hz", type=float, default=5e5,
                    help="Entangled pair generation rate (Hz)")
    cs.add_argument("--background-rate-hz", type=float, default=5e4,
                    help="Background rate per detector (Hz)")
    cs.add_argument("--jitter-ps", type=float, default=80.0,
                    help="Timing jitter sigma in picoseconds")
    cs.add_argument("--tau-ps", type=float, default=200.0,
                    help="Coincidence window in picoseconds")
    cs.add_argument("--dead-time-ps", type=float, default=0.0,
                    help="Detector dead time in picoseconds")
    cs.add_argument("--afterpulse-prob", type=float, default=0.0,
                    help="Afterpulsing probability per detection")
    cs.add_argument("--afterpulse-window-ps", type=float, default=0.0,
                    help="Afterpulse window in picoseconds")
    cs.add_argument("--afterpulse-decay-ps", type=float, default=0.0,
                    help="Afterpulse decay time in picoseconds (0 = flat)")
    cs.add_argument("--clock-offset-s", type=float, default=0.0,
                    help="Clock offset between Alice/Bob in seconds")
    cs.add_argument("--clock-drift-ppm", type=float, default=0.0,
                    help="Clock drift in ppm (linear over time)")
    cs.add_argument("--tdc-ps", type=float, default=0.0,
                    help="TDC resolution in picoseconds (0 = no quantization)")
    cs.add_argument("--estimate-offset", action="store_true",
                    help="Estimate clock offset via coincidence scan")
    cs.add_argument("--stream-mode", action="store_true",
                    help="Use event-stream instrument pipeline")
    cs.add_argument("--gate-duty-cycle", type=float, default=1.0,
                    help="Gating duty cycle (0..1) for stream mode")
    cs.add_argument("--dead-time-ns", type=float, default=0.0,
                    help="Detector dead time in nanoseconds (stream mode)")
    cs.add_argument("--filter-bandwidth-nm", type=float, default=1.0,
                    help="Optical filter bandwidth in nm")
    cs.add_argument("--detector-temp-c", type=float, default=20.0,
                    help="Detector temperature in C")
    cs.add_argument("--seed", type=int, default=0)
    cs.add_argument("--outdir", type=str, default=".",
                    help="Output directory for figures/reports (default: .)")
    cs.add_argument("--min-visibility", type=float, default=0.0,
                    help="Abort if visibility below this threshold")
    cs.add_argument("--min-chsh-s", type=float, default=2.0,
                    help="Abort if CHSH S below this threshold")

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    # Validate arguments post-parse
    _validate_args(args)

    if args.cmd == "sweep":
        _run_sweep(args)
    elif args.cmd == "decoy-sweep":
        _run_decoy_sweep(args)
    elif args.cmd == "attack-sweep":
        _run_attack_sweep(args)
    elif args.cmd == "pass-sweep":
        _run_pass_sweep(args)
    elif args.cmd == "experiment-run":
        _run_experiment(args)
    elif args.cmd == "forecast-run":
        _run_forecast_run(args)
    elif args.cmd == "coincidence-sim":
        _run_coincidence_sim(args)
    elif args.cmd == "calibration-fit":
        _run_calibration_fit(args)
    elif args.cmd == "constellation-sweep":
        _run_constellation_sweep(args)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments after parsing."""
    # Common validations for sweep/decoy-sweep commands
    if args.cmd in ("sweep", "decoy-sweep", "attack-sweep"):
        validate_float("loss-min", args.loss_min, min_value=0.0)
        validate_float("loss-max", args.loss_max, min_value=0.0)
        if args.loss_min > args.loss_max:
            raise ValueError(f"loss-min ({args.loss_min}) > loss-max ({args.loss_max})")
        validate_int("steps", args.steps, min_value=1)
        if hasattr(args, "trials"):
            validate_int("trials", args.trials, min_value=1)

    # Common validations for all commands with sweep-style controls
    if hasattr(args, "flip_prob"):
        validate_float("flip-prob", args.flip_prob, min_value=0.0, max_value=0.5)
    if hasattr(args, "pulses"):
        validate_int("pulses", args.pulses, min_value=1)
    if hasattr(args, "seed"):
        validate_seed(args.seed)
    if hasattr(args, "eta"):
        validate_float("eta", args.eta, min_value=0.0, max_value=1.0)
    if hasattr(args, "p_bg"):
        validate_float("p-bg", args.p_bg, min_value=0.0, max_value=1.0)

    # Command-specific validations
    if args.cmd == "sweep":
        validate_int("workers", args.workers, min_value=1)
        if args.n_sent is not None:
            validate_int("n-sent", args.n_sent, min_value=1)
        if args.rep_rate is not None:
            validate_float("rep-rate", args.rep_rate, min_value=1e-9)
        if args.rep_rate_hz is not None:
            validate_float("rep-rate-hz", args.rep_rate_hz, min_value=1e-9)
        if args.target_bits is not None:
            validate_int("target-bits", args.target_bits, min_value=1)
        if args.pass_seconds is not None:
            validate_float("pass-seconds", args.pass_seconds, min_value=1e-9)
            if args.rep_rate is None and args.rep_rate_hz is None and args.target_bits is None:
                raise ValueError("pass-seconds requires either --rep-rate, --rep-rate-hz, or --target-bits")
        # Finite-key parameter validation
        if hasattr(args, "finite_key") and args.finite_key:
            validate_float("eps-pe", args.eps_pe, min_value=0.0, max_value=1.0)
            validate_float("eps-sec", args.eps_sec, min_value=0.0, max_value=1.0)
            validate_float("eps-cor", args.eps_cor, min_value=0.0, max_value=1.0)
            validate_float("ec-efficiency", args.ec_efficiency, min_value=1.0)
            if args.f_ec is not None:
                validate_float("f-ec", args.f_ec, min_value=1.0)
            validate_float("pe-frac", args.pe_frac, min_value=1e-9, max_value=1.0)
            if args.m_pe is not None:
                validate_int("m-pe", args.m_pe, min_value=1)
        validate_float("mu", args.mu, min_value=0.0)
        validate_float("timeshift-bias", args.timeshift_bias, min_value=0.0, max_value=1.0)
        validate_float("blinding-prob", args.blinding_prob, min_value=0.0, max_value=1.0)
        validate_float("leakage-fraction", args.leakage_fraction, min_value=0.0, max_value=1.0)
        if args.eta_z is not None:
            validate_float("eta-z", args.eta_z, min_value=0.0, max_value=1.0)
        if args.eta_x is not None:
            validate_float("eta-x", args.eta_x, min_value=0.0, max_value=1.0)
    elif args.cmd == "decoy-sweep":
        validate_float("mu-s", args.mu_s, min_value=0.0)
        validate_float("mu-d", args.mu_d, min_value=0.0)
        validate_float("mu-v", args.mu_v, min_value=0.0, max_value=0.0)
        validate_float("mu-sigma", args.mu_sigma, min_value=0.0)
        validate_float("decoy-mu-sigma", args.decoy_mu_sigma, min_value=0.0)
        validate_float("p-s", args.p_s, min_value=0.0, max_value=1.0)
        validate_float("p-d", args.p_d, min_value=0.0, max_value=1.0)
        validate_float("p-v", args.p_v, min_value=0.0, max_value=1.0)
        validate_float("afterpulse-prob", args.afterpulse_prob, min_value=0.0, max_value=1.0)
        validate_int("afterpulse-window", args.afterpulse_window, min_value=0)
        validate_float("afterpulse-decay", args.afterpulse_decay, min_value=0.0)
        validate_int("dead-time-pulses", args.dead_time_pulses, min_value=0)
        if args.eta_z is not None:
            validate_float("eta-z", args.eta_z, min_value=0.0, max_value=1.0)
        if args.eta_x is not None:
            validate_float("eta-x", args.eta_x, min_value=0.0, max_value=1.0)
    elif args.cmd == "attack-sweep":
        validate_float("mu", args.mu, min_value=0.0)
        validate_float("timeshift-bias", args.timeshift_bias, min_value=0.0, max_value=1.0)
        validate_float("blinding-prob", args.blinding_prob, min_value=0.0, max_value=1.0)
        validate_float("leakage-fraction", args.leakage_fraction, min_value=0.0, max_value=1.0)
        if args.eta_z is not None:
            validate_float("eta-z", args.eta_z, min_value=0.0, max_value=1.0)
        if args.eta_x is not None:
            validate_float("eta-x", args.eta_x, min_value=0.0, max_value=1.0)
    elif args.cmd == "pass-sweep":
        validate_float("max-elevation", args.max_elevation, min_value=0.0, max_value=90.0)
        validate_float("min-elevation", args.min_elevation, min_value=0.0, max_value=90.0)
        if args.min_elevation >= args.max_elevation:
            raise ValueError(f"min-elevation ({args.min_elevation}) >= max-elevation ({args.max_elevation})")
        validate_float("pass-duration", args.pass_duration, min_value=1.0)
        validate_float("time-step", args.time_step, min_value=0.1)
        validate_float("wavelength", args.wavelength, min_value=1e-9)
        validate_float("tx-diameter", args.tx_diameter, min_value=0.001)
        validate_float("rx-diameter", args.rx_diameter, min_value=0.001)
        validate_float("sigma-point", args.sigma_point, min_value=0.0)
        validate_float("altitude", args.altitude, min_value=100e3)
        validate_float("atm-loss-db", args.atm_loss_db, min_value=0.0)
        validate_float("sigma-ln", args.sigma_ln, min_value=0.0)
        validate_float("system-loss-db", args.system_loss_db, min_value=0.0)
        validate_float("day-bg-factor", args.day_bg_factor, min_value=1.0)
        if args.n_sent is not None:
            validate_int("n-sent", args.n_sent, min_value=1)
        if args.rep_rate is not None:
            validate_float("rep-rate", args.rep_rate, min_value=1e-9)
        if args.finite_key:
            validate_float("eps-pe", args.eps_pe, min_value=0.0, max_value=1.0)
            validate_float("eps-sec", args.eps_sec, min_value=0.0, max_value=1.0)
            validate_float("eps-cor", args.eps_cor, min_value=0.0, max_value=1.0)
            validate_float("ec-efficiency", args.ec_efficiency, min_value=1.0)
            validate_float("pe-frac", args.pe_frac, min_value=1e-9, max_value=1.0)
            if args.m_pe is not None:
                validate_int("m-pe", args.m_pe, min_value=1)
        if args.fading:
            validate_float("fading-sigma-ln", args.fading_sigma_ln, min_value=0.0)
            validate_int("fading-samples", args.fading_samples, min_value=1)
            validate_int("fading-trials", args.fading_trials, min_value=1)
            validate_seed(args.fading_seed)
        if args.pointing:
            validate_float("acq-seconds", args.acq_seconds, min_value=0.0)
            validate_float("dropout-prob", args.dropout_prob, min_value=0.0, max_value=1.0)
            validate_float("relock-seconds", args.relock_seconds, min_value=0.0)
            validate_float("pointing-jitter-urad", args.pointing_jitter_urad, min_value=0.0)
            validate_seed(args.pointing_seed)
        validate_float("filter-bandwidth-nm", args.filter_bandwidth_nm, min_value=0.01)
        validate_float("detector-temp-c", args.detector_temp_c, min_value=-80.0, max_value=80.0)
        time_s, _ = generate_elevation_profile(
            max_elevation_deg=args.max_elevation,
            min_elevation_deg=args.min_elevation,
            time_step_s=args.time_step,
            pass_duration_s=args.pass_duration,
        )
        n_steps = len(time_s)
        if args.n_sent is not None:
            total_sent = args.n_sent
        elif args.rep_rate is not None:
            total_sent = int(round(args.rep_rate * args.pass_duration))
        else:
            total_sent = args.pulses
        if total_sent < n_steps:
            raise ValueError(
                "Total pulses must be >= number of time steps; "
                "increase total pulses or decrease --time-step."
            )
    elif args.cmd == "experiment-run":
        validate_seed(args.seed)
        validate_int("n-blocks", args.n_blocks, min_value=1)
        validate_float("block-seconds", args.block_seconds, min_value=1e-9)
        validate_float("rep-rate-hz", args.rep_rate_hz, min_value=1e-9)
        validate_float("pass-seconds", args.pass_seconds, min_value=1e-9)
        if args.finite_key:
            validate_float("eps-sec", args.eps_sec, min_value=0.0, max_value=1.0)
            validate_float("eps-cor", args.eps_cor, min_value=0.0, max_value=1.0)
    elif args.cmd == "forecast-run":
        validate_seed(args.seed)
        validate_int("n-blocks", args.n_blocks, min_value=1)
        validate_float("block-seconds", args.block_seconds, min_value=1e-9)
        validate_float("rep-rate-hz", args.rep_rate_hz, min_value=1e-9)
        if not Path(args.forecasts).exists():
            raise FileNotFoundError(f"Forecasts file not found: {args.forecasts}")
    elif args.cmd == "coincidence-sim":
        validate_float("loss-min", args.loss_min, min_value=0.0)
        validate_float("loss-max", args.loss_max, min_value=0.0)
        if args.loss_min > args.loss_max:
            raise ValueError(f"loss-min ({args.loss_min}) > loss-max ({args.loss_max})")
        validate_int("steps", args.steps, min_value=1)
        validate_float("duration", args.duration, min_value=1e-9)
        validate_float("pair-rate-hz", args.pair_rate_hz, min_value=0.0)
        validate_float("background-rate-hz", args.background_rate_hz, min_value=0.0)
        validate_float("jitter-ps", args.jitter_ps, min_value=0.0)
        validate_float("tau-ps", args.tau_ps, min_value=1e-9)
        validate_float("dead-time-ps", args.dead_time_ps, min_value=0.0)
        validate_float("afterpulse-prob", args.afterpulse_prob, min_value=0.0, max_value=1.0)
        validate_float("afterpulse-window-ps", args.afterpulse_window_ps, min_value=0.0)
        validate_float("afterpulse-decay-ps", args.afterpulse_decay_ps, min_value=0.0)
        validate_float("clock-offset-s", args.clock_offset_s, min_value=-1.0, max_value=1.0)
        validate_float("clock-drift-ppm", args.clock_drift_ppm, min_value=-1e6, max_value=1e6)
        validate_float("tdc-ps", args.tdc_ps, min_value=0.0)
        validate_seed(args.seed)
        validate_float("min-visibility", args.min_visibility, min_value=0.0, max_value=1.0)
        validate_float("min-chsh-s", args.min_chsh_s, min_value=0.0)
        validate_float("gate-duty-cycle", args.gate_duty_cycle, min_value=0.0, max_value=1.0)
        validate_float("dead-time-ns", args.dead_time_ns, min_value=0.0)
        validate_float("filter-bandwidth-nm", args.filter_bandwidth_nm, min_value=0.01)
        validate_float("detector-temp-c", args.detector_temp_c, min_value=-80.0, max_value=80.0)
    elif args.cmd == "calibration-fit":
        if not Path(args.telemetry).exists():
            raise FileNotFoundError(f"Telemetry file not found: {args.telemetry}")
        validate_float("eta-base", args.eta_base, min_value=0.0, max_value=1.0)
        validate_float("p-bg-min", args.p_bg_min, min_value=0.0, max_value=1.0)
        validate_float("p-bg-max", args.p_bg_max, min_value=0.0, max_value=1.0)
        validate_float("flip-min", args.flip_min, min_value=0.0, max_value=0.5)
        validate_float("flip-max", args.flip_max, min_value=0.0, max_value=0.5)
        validate_float("eta-scale-min", args.eta_scale_min, min_value=0.0, max_value=1.0)
        validate_float("eta-scale-max", args.eta_scale_max, min_value=0.0, max_value=1.0)
        validate_int("grid-steps", args.grid_steps, min_value=2)
        if args.p_bg_min > args.p_bg_max:
            raise ValueError("p-bg-min must be <= p-bg-max")
        if args.flip_min > args.flip_max:
            raise ValueError("flip-min must be <= flip-max")
        if args.eta_scale_min > args.eta_scale_max:
            raise ValueError("eta-scale-min must be <= eta-scale-max")
    elif args.cmd == "constellation-sweep":
        validate_int("n-sats", args.n_sats, min_value=1)
        validate_int("n-stations", args.n_stations, min_value=1)
        validate_float("horizon-hours", args.horizon_hours, min_value=0.1)
        validate_int("passes-per-sat", args.passes_per_sat, min_value=1)
        validate_float("pass-duration-s", args.pass_duration_s, min_value=1.0)
        validate_float("initial-bits", args.initial_bits, min_value=0.0)
        validate_float("production-bits-per-pass", args.production_bits_per_pass, min_value=0.0)
        validate_float("consumption-bps", args.consumption_bps, min_value=0.0)
        validate_seed(args.seed)


def _resolve_n_sent_for_sweep(args: argparse.Namespace) -> int:
    """Resolve total pulses sent for sweep commands."""
    if args.n_sent is not None:
        n_sent = args.n_sent
    elif args.rep_rate is not None and args.pass_seconds is not None:
        n_sent = int(args.rep_rate * args.pass_seconds)
    else:
        n_sent = args.pulses

    if n_sent <= 0:
        raise ValueError(f"n_sent must be positive, got {n_sent}")
    return n_sent


def _load_calibration_model(args: argparse.Namespace) -> Any:
    """Load calibration model if --calibration-file is provided.

    Returns:
        CalibrationModel instance if file provided, else None
    """
    if hasattr(args, "calibration_file") and args.calibration_file is not None:
        print(f"Loading calibration from {args.calibration_file}")
        return CalibrationModel.from_file(args.calibration_file)
    return None


def _resolve_pass_pulse_accounting(args: argparse.Namespace, n_steps: int) -> tuple[list[int], int]:
    """Resolve per-step pulse schedule and total pulses for pass-sweep.

    Precedence: --n-sent, then --rep-rate, then --pulses.
    """
    if args.n_sent is not None:
        total_sent = args.n_sent
    elif args.rep_rate is not None:
        # Round to avoid systematic undercount from float truncation.
        total_sent = int(round(args.rep_rate * args.pass_duration))
    else:
        total_sent = args.pulses

    if total_sent <= 0:
        raise ValueError("Pulse accounting must yield positive values.")

    if n_steps <= 0:
        raise ValueError("n_steps must be positive for pass pulse accounting.")
    if total_sent < n_steps:
        raise ValueError(
            "Total pulses must be >= number of time steps; "
            "increase total pulses or decrease --time-step."
        )
    base = total_sent // n_steps
    remainder = total_sent % n_steps
    pulses_per_step = [base + 1] * remainder + [base] * (n_steps - remainder)

    return pulses_per_step, total_sent


def _run_sweep(args: argparse.Namespace) -> None:
    """Execute BB84 sweep command."""
    loss_vals = np.linspace(args.loss_min, args.loss_max, args.steps)
    detector = DetectorParams(eta=args.eta, p_bg=args.p_bg, eta_z=args.eta_z, eta_x=args.eta_x)
    n_sent = _resolve_n_sent_for_sweep(args)
    attack_cfg = AttackConfig(
        attack=args.attack,
        mu=args.mu,
        timeshift_bias=args.timeshift_bias,
        blinding_mode=args.blinding_mode,
        blinding_prob=args.blinding_prob,
        leakage_fraction=args.leakage_fraction,
    )

    # Load optional calibration model
    calibration = _load_calibration_model(args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)
    (outdir / "reports").mkdir(exist_ok=True)

    # Check for finite-key mode
    if getattr(args, "finite_key", False):
        _run_sweep_finite_key(args, loss_vals, detector, outdir, n_sent)
        return

    if args.trials > 1:
        # Run with Monte Carlo CI
        print(f"Running sweep with {args.trials} trials per point")
        no_attack = sweep_loss_with_ci(
            loss_vals,
            flip_prob=args.flip_prob,
            attack="none",
            n_pulses=n_sent,
            seed=args.seed,
            n_trials=args.trials,
            detector=detector,
            n_workers=args.workers,
        )
        attack = sweep_loss_with_ci(
            loss_vals,
            flip_prob=args.flip_prob,
            attack=args.attack,
            n_pulses=n_sent,
            seed=args.seed + 100_000,
            n_trials=args.trials,
            detector=detector,
            n_workers=args.workers,
            attack_config=attack_cfg,
        )

        # Add engineering outputs and headroom to each record
        for rec in no_attack:
            # Apply calibration first if provided
            if calibration:
                calibrated_rec = calibration.apply_to_record(rec, loss_db=rec.get("loss_db"))
                rec.update(calibrated_rec)

            # Engineering outputs
            eng_out = compute_engineering_outputs(
                rec["key_rate_per_pulse_mean"],
                rep_rate_hz=args.rep_rate_hz,
                pass_seconds=args.pass_seconds,
                target_bits=args.target_bits,
            )
            rec.update(eng_out)

            # Headroom calculations
            headroom = compute_headroom(
                rec["qber_mean"],
                qber_ci_low=rec.get("qber_ci_low"),
                qber_ci_high=rec.get("qber_ci_high"),
            )
            rec.update(headroom)

        for rec in attack:
            # Apply calibration first if provided
            if calibration:
                calibrated_rec = calibration.apply_to_record(rec, loss_db=rec.get("loss_db"))
                rec.update(calibrated_rec)

            eng_out = compute_engineering_outputs(
                rec["key_rate_per_pulse_mean"],
                rep_rate_hz=args.rep_rate_hz,
                pass_seconds=args.pass_seconds,
                target_bits=args.target_bits,
            )
            rec.update(eng_out)

            headroom = compute_headroom(
                rec["qber_mean"],
                qber_ci_low=rec.get("qber_ci_low"),
                qber_ci_high=rec.get("qber_ci_high"),
            )
            rec.update(headroom)

        # Generate CI plots
        qber_ci_path = plot_qber_vs_loss_ci(
            no_attack, attack,
            str(outdir / "figures" / "qber_vs_loss_ci.png"),
            attack_label=args.attack,
        )
        # Canonical name for secret fraction CI plot
        sf_ci_path = plot_key_rate_vs_loss_ci(
            no_attack, attack,
            str(outdir / "figures" / "secret_fraction_vs_loss_ci.png"),
            attack_label=args.attack,
        )
        # Legacy alias for backwards compatibility
        import shutil
        legacy_sf_ci_path = outdir / "figures" / "key_rate_vs_loss_ci.png"
        shutil.copy(sf_ci_path, legacy_sf_ci_path)

        # Generate headroom plot
        headroom_path = plot_qber_headroom_vs_loss(
            no_attack,
            str(outdir / "figures" / "qber_headroom_vs_loss.png"),
            show_ci=True,
        )
        print("CI Plots:", qber_ci_path, sf_ci_path)
        print("Headroom Plot:", headroom_path)

        # Build report with CI data
        report = {
            "schema_version": SCHEMA_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "loss_sweep_ci": {
                "no_attack": no_attack,
                str(args.attack): attack,
            },
            "summary_stats": {
                "no_attack": compute_summary_stats(no_attack),
                str(args.attack): compute_summary_stats(attack),
            },
            "parameters": {
                "loss_min": args.loss_min,
                "loss_max": args.loss_max,
                "steps": args.steps,
                "flip_prob": args.flip_prob,
                "pulses": n_sent,
                "n_sent": n_sent,
                "rep_rate": args.rep_rate,
                "pass_seconds": args.pass_seconds,
                "trials": args.trials,
                "eta": args.eta,
                "eta_z": detector.eta_z,
                "eta_x": detector.eta_x,
                "p_bg": args.p_bg,
                "attack": args.attack,
                "attack_mu": args.mu,
                "timeshift_bias": args.timeshift_bias,
                "blinding_mode": args.blinding_mode,
                "blinding_prob": args.blinding_prob,
                "leakage_fraction": args.leakage_fraction,
            },
            "artifacts": {
                "qber_ci_plot": "qber_vs_loss_ci.png",
                "secret_fraction_ci_plot": "secret_fraction_vs_loss_ci.png",
                "key_ci_plot": "key_rate_vs_loss_ci.png",  # legacy alias
                "headroom_plot": "qber_headroom_vs_loss.png",
            },
        }

        # Add engineering parameters if provided
        if args.rep_rate_hz is not None:
            report["parameters"]["rep_rate_hz"] = args.rep_rate_hz
        if args.target_bits is not None:
            report["parameters"]["target_bits"] = args.target_bits
        if args.leakage_fraction > 0.0:
            leak_vals = [rec.get("leakage_budget_bits", 0.0) for rec in attack]
            report["parameters"]["leakage_budget_bits"] = float(np.mean(leak_vals)) if leak_vals else 0.0

        # Add calibration metadata if applied
        if calibration:
            report["calibration"] = calibration.get_metadata()
    else:
        # Single trial (original behavior)
        no_attack = sweep_loss(
            loss_vals,
            flip_prob=args.flip_prob,
            attack="none",
            n_pulses=n_sent,
            seed=args.seed,
            detector=detector,
        )
        attack = sweep_loss(
            loss_vals,
            flip_prob=args.flip_prob,
            attack=args.attack,
            n_pulses=n_sent,
            seed=args.seed + 10_000,
            detector=detector,
            attack_config=attack_cfg,
        )

        # Add engineering outputs and headroom to each record
        for rec in no_attack:
            # Apply calibration first if provided
            if calibration:
                calibrated_rec = calibration.apply_to_record(rec, loss_db=rec.get("loss_db"))
                rec.update(calibrated_rec)

            eng_out = compute_engineering_outputs(
                rec["key_rate_per_pulse"],
                rep_rate_hz=args.rep_rate_hz,
                pass_seconds=args.pass_seconds,
                target_bits=args.target_bits,
            )
            rec.update(eng_out)

            headroom = compute_headroom(rec["qber"])
            rec.update(headroom)

        for rec in attack:
            # Apply calibration first if provided
            if calibration:
                calibrated_rec = calibration.apply_to_record(rec, loss_db=rec.get("loss_db"))
                rec.update(calibrated_rec)

            eng_out = compute_engineering_outputs(
                rec["key_rate_per_pulse"],
                rep_rate_hz=args.rep_rate_hz,
                pass_seconds=args.pass_seconds,
                target_bits=args.target_bits,
            )
            rec.update(eng_out)

            headroom = compute_headroom(rec["qber"])
            rec.update(headroom)

        q_path, k_path = plot_key_metrics_vs_loss(
            no_attack, attack,
            str(outdir / "figures" / "key"),
            attack_label=args.attack,
        )
        # Legacy alias for backwards compatibility
        import shutil
        legacy_k_path = outdir / "figures" / "key_key_fraction_vs_loss.png"
        shutil.copy(k_path, legacy_k_path)

        # Generate headroom plot
        headroom_path = plot_qber_headroom_vs_loss(
            no_attack,
            str(outdir / "figures" / "qber_headroom_vs_loss.png"),
            show_ci=False,
        )
        print("Plots:", q_path, k_path)
        print("Headroom Plot:", headroom_path)

        report = {
            "schema_version": SCHEMA_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "loss_sweep": {
                "no_attack": no_attack,
                str(args.attack): attack,
            },
            "parameters": {
                "loss_min": args.loss_min,
                "loss_max": args.loss_max,
                "steps": args.steps,
                "flip_prob": args.flip_prob,
                "pulses": n_sent,
                "n_sent": n_sent,
                "rep_rate": args.rep_rate,
                "pass_seconds": args.pass_seconds,
                "eta": args.eta,
                "eta_z": detector.eta_z,
                "eta_x": detector.eta_x,
                "p_bg": args.p_bg,
                "attack": args.attack,
                "attack_mu": args.mu,
                "timeshift_bias": args.timeshift_bias,
                "blinding_mode": args.blinding_mode,
                "blinding_prob": args.blinding_prob,
                "leakage_fraction": args.leakage_fraction,
            },
            "artifacts": {
                "qber_plot": str(Path(q_path).name),
                "secret_fraction_plot": str(Path(k_path).name),
                "key_fraction_plot": "key_key_fraction_vs_loss.png",  # legacy alias
                "headroom_plot": "qber_headroom_vs_loss.png",
            },
        }

        # Add engineering parameters if provided
        if args.rep_rate_hz is not None:
            report["parameters"]["rep_rate_hz"] = args.rep_rate_hz
        if args.target_bits is not None:
            report["parameters"]["target_bits"] = args.target_bits
        if args.leakage_fraction > 0.0:
            leak_vals = [rec.get("leakage_budget_bits", 0.0) for rec in attack]
            report["parameters"]["leakage_budget_bits"] = float(np.mean(leak_vals)) if leak_vals else 0.0

        # Add calibration metadata if applied
        if calibration:
            report["calibration"] = calibration.get_metadata()

    with open(outdir / "reports" / "latest.json", "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")  # Ensure trailing newline
    print("Wrote:", outdir / "reports" / "latest.json")


def _run_attack_sweep(args: argparse.Namespace) -> None:
    """Execute attack comparison sweep."""
    loss_vals = np.linspace(args.loss_min, args.loss_max, args.steps)
    detector = DetectorParams(eta=args.eta, p_bg=args.p_bg, eta_z=args.eta_z, eta_x=args.eta_x)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)
    (outdir / "reports").mkdir(exist_ok=True)

    results: Dict[str, list[dict[str, Any]]] = {}
    for i, attack in enumerate(args.attacks):
        attack_cfg = AttackConfig(
            attack=attack,
            mu=args.mu,
            timeshift_bias=args.timeshift_bias,
            blinding_mode=args.blinding_mode,
            blinding_prob=args.blinding_prob,
            leakage_fraction=args.leakage_fraction,
        )
        records = sweep_loss(
            loss_vals,
            flip_prob=args.flip_prob,
            attack=attack,
            n_pulses=args.pulses,
            seed=args.seed + i * 10_000,
            detector=detector,
            attack_config=attack_cfg,
        )
        results[str(attack)] = records

    plot_path = plot_attack_comparison_key_rate(
        results,
        str(outdir / "figures" / "attack_comparison_key_rate.png"),
    )
    print("Plot:", plot_path)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "attack_sweep": results,
        "parameters": {
            "loss_min": args.loss_min,
            "loss_max": args.loss_max,
            "steps": args.steps,
            "flip_prob": args.flip_prob,
            "pulses": args.pulses,
            "eta": args.eta,
            "eta_z": detector.eta_z,
            "eta_x": detector.eta_x,
            "p_bg": args.p_bg,
            "attacks": args.attacks,
            "attack_mu": args.mu,
            "timeshift_bias": args.timeshift_bias,
            "blinding_mode": args.blinding_mode,
            "blinding_prob": args.blinding_prob,
            "leakage_fraction": args.leakage_fraction,
            "leakage_fraction": args.leakage_fraction,
        },
        "artifacts": {
            "attack_comparison_plot": "attack_comparison_key_rate.png",
        },
    }

    if args.leakage_fraction > 0.0:
        leak_vals = [rec.get("leakage_budget_bits", 0.0) for rec in attack]
        report["parameters"]["leakage_budget_bits"] = float(np.mean(leak_vals)) if leak_vals else 0.0

    with open(outdir / "reports" / "latest.json", "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print("Wrote:", outdir / "reports" / "latest.json")


def _run_sweep_finite_key(
    args: argparse.Namespace,
    loss_vals: np.ndarray,
    detector: DetectorParams,
    outdir: Path,
    n_sent: int,
) -> None:
    """Execute BB84 sweep with finite-key analysis."""
    print("Running finite-key analysis sweep")

    f_ec = args.f_ec if args.f_ec is not None else args.ec_efficiency
    fk_params = FiniteKeyParams(
        eps_pe=args.eps_pe,
        eps_sec=args.eps_sec,
        eps_cor=args.eps_cor,
        ec_efficiency=f_ec,
        pe_frac=args.pe_frac,
        m_pe=args.m_pe,
    )
    attack_cfg = AttackConfig(
        attack=args.attack,
        mu=args.mu,
        timeshift_bias=args.timeshift_bias,
        blinding_mode=args.blinding_mode,
        blinding_prob=args.blinding_prob,
        leakage_fraction=args.leakage_fraction,
    )

    # Run finite-key sweep for no-attack scenario
    no_attack = sweep_loss_finite_key(
        loss_vals,
        flip_prob=args.flip_prob,
        attack="none",
        n_pulses=n_sent,
        seed=args.seed,
        detector=detector,
        finite_key_params=fk_params,
    )

    # Run finite-key sweep for attack scenario
    attack = sweep_loss_finite_key(
        loss_vals,
        flip_prob=args.flip_prob,
        attack=args.attack,
        n_pulses=n_sent,
        seed=args.seed + 10_000,
        detector=detector,
        finite_key_params=fk_params,
        attack_config=attack_cfg,
    )

    # Finite-key rate vs total pulses for a representative loss value
    representative_loss = float(loss_vals[len(loss_vals) // 2])
    base_n_sent = np.logspace(4, 6, num=5).astype(int).tolist()
    if n_sent not in base_n_sent:
        base_n_sent.append(int(n_sent))
    n_sent_values = sorted({int(x) for x in base_n_sent if int(x) > 0})
    n_sent_sweep = sweep_finite_key_vs_n_sent(
        representative_loss,
        n_sent_values,
        flip_prob=args.flip_prob,
        attack="none",
        seed=args.seed + 50_000,
        detector=detector,
        finite_key_params=fk_params,
    )

    # Generate finite-key plots
    comparison_path = plot_finite_key_comparison(
        no_attack,
        str(outdir / "figures" / "finite_key_comparison.png"),
    )
    bits_path = plot_finite_key_bits_vs_loss(
        no_attack,
        str(outdir / "figures" / "finite_key_bits_vs_loss.png"),
    )
    penalty_path = plot_finite_size_penalty(
        no_attack,
        str(outdir / "figures" / "finite_size_penalty.png"),
    )
    rate_vs_n_sent_path = plot_finite_key_rate_vs_n_sent(
        n_sent_sweep,
        str(outdir / "figures" / "finite_key_rate_vs_n_sent.png"),
    )
    print("Finite-key plots:", comparison_path, bits_path, penalty_path, rate_vs_n_sent_path)

    # Build report with finite-key data
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "finite_key_sweep": {
            "no_attack": no_attack,
            str(args.attack): attack,
        },
        "parameters": {
            "loss_min": args.loss_min,
            "loss_max": args.loss_max,
            "steps": args.steps,
            "flip_prob": args.flip_prob,
            "pulses": n_sent,
            "n_sent": n_sent,
            "rep_rate": args.rep_rate,
            "pass_seconds": args.pass_seconds,
            "eta": args.eta,
            "eta_z": detector.eta_z,
            "eta_x": detector.eta_x,
            "p_bg": args.p_bg,
            "attack": args.attack,
            "attack_mu": args.mu,
            "timeshift_bias": args.timeshift_bias,
            "blinding_mode": args.blinding_mode,
            "blinding_prob": args.blinding_prob,
            "finite_key": True,
            "eps_pe": args.eps_pe,
            "eps_sec": args.eps_sec,
            "eps_cor": args.eps_cor,
            "eps_total": fk_params.eps_total,
            "ec_efficiency": fk_params.ec_efficiency,
            "f_ec": fk_params.ec_efficiency,
            "pe_frac": fk_params.pe_frac,
            "m_pe": fk_params.m_pe,
            "finite_key_plot_loss_db": representative_loss,
        },
        "artifacts": {
            "finite_key_comparison_plot": "finite_key_comparison.png",
            "finite_key_bits_plot": "finite_key_bits_vs_loss.png",
            "finite_size_penalty_plot": "finite_size_penalty.png",
            "finite_key_rate_vs_n_sent_plot": "finite_key_rate_vs_n_sent.png",
        },
    }

    with open(outdir / "reports" / "latest.json", "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print("Wrote:", outdir / "reports" / "latest.json")


def _run_decoy_sweep(args: argparse.Namespace) -> None:
    """Execute decoy-state BB84 sweep command."""
    # Validate probabilities sum to 1
    prob_sum = args.p_s + args.p_d + args.p_v
    if not (0.999 < prob_sum < 1.001):
        raise ValueError(f"Probabilities must sum to 1, got {prob_sum}")

    loss_vals = np.linspace(args.loss_min, args.loss_max, args.steps)
    detector = DetectorParams(
        eta=args.eta,
        p_bg=args.p_bg,
        p_afterpulse=args.afterpulse_prob,
        afterpulse_window=args.afterpulse_window,
        afterpulse_decay=args.afterpulse_decay,
        dead_time_pulses=args.dead_time_pulses,
        eta_z=args.eta_z,
        eta_x=args.eta_x,
    )
    decoy = DecoyParams(
        mu_s=args.mu_s,
        mu_d=args.mu_d,
        mu_v=args.mu_v,
        p_s=args.p_s,
        p_d=args.p_d,
        p_v=args.p_v,
        mu_s_sigma=args.mu_sigma,
        mu_d_sigma=args.decoy_mu_sigma,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)
    (outdir / "reports").mkdir(exist_ok=True)

    print("Running decoy-state BB84 sweep")
    results = sweep_decoy_loss(
        loss_vals,
        flip_prob=args.flip_prob,
        n_pulses=args.pulses,
        seed=args.seed,
        decoy=decoy,
        detector=detector,
        n_trials=args.trials,
    )

    # Generate plot
    show_ci = args.trials > 1
    plot_path = plot_decoy_key_rate_vs_loss(
        results,
        str(outdir / "figures" / "decoy_key_rate_vs_loss.png"),
        show_ci=show_ci,
    )
    print("Plot:", plot_path)

    afterpulse_enabled = args.afterpulse_prob > 0.0 and args.afterpulse_window > 0
    realism_enabled = any([
        args.mu_sigma > 0.0,
        args.decoy_mu_sigma > 0.0,
        afterpulse_enabled,
        args.dead_time_pulses > 0,
        args.eta_z is not None and args.eta_z != args.eta,
        args.eta_x is not None and args.eta_x != args.eta,
    ])
    if realism_enabled:
        print("Note: realism enabled -> per-pulse simulation; consider reducing --pulses/--trials.")
        baseline_detector = DetectorParams(eta=args.eta, p_bg=args.p_bg)
        baseline_decoy = DecoyParams(
            mu_s=args.mu_s,
            mu_d=args.mu_d,
            mu_v=args.mu_v,
            p_s=args.p_s,
            p_d=args.p_d,
            p_v=args.p_v,
        )
        baseline_results = sweep_decoy_loss(
            loss_vals,
            flip_prob=args.flip_prob,
            n_pulses=args.pulses,
            seed=args.seed,
            decoy=baseline_decoy,
            detector=baseline_detector,
            n_trials=args.trials,
        )
        realism_plot_path = plot_decoy_key_rate_vs_loss_comparison(
            baseline_results,
            results,
            str(outdir / "figures" / "decoy_key_rate_vs_loss_realism.png"),
            show_ci=show_ci,
        )
        print("Plot:", realism_plot_path)

    # Update or create report
    report_path = outdir / "reports" / "latest.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
    else:
        report = {}

    # Always update schema version and timestamp
    report["schema_version"] = SCHEMA_VERSION
    report["generated_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Add decoy section
    report["decoy_loss_sweep"] = {
        "results": results,
        "parameters": {
            "loss_min": args.loss_min,
            "loss_max": args.loss_max,
            "steps": args.steps,
            "flip_prob": args.flip_prob,
            "pulses": args.pulses,
            "trials": args.trials,
            "mu_s": args.mu_s,
            "mu_d": args.mu_d,
            "mu_v": args.mu_v,
            "mu_sigma": args.mu_sigma,
            "decoy_mu_sigma": args.decoy_mu_sigma,
            "p_s": args.p_s,
            "p_d": args.p_d,
            "p_v": args.p_v,
            "eta": args.eta,
            "eta_z": detector.eta_z,
            "eta_x": detector.eta_x,
            "p_bg": args.p_bg,
            "afterpulse_prob": args.afterpulse_prob,
            "afterpulse_window": args.afterpulse_window,
            "afterpulse_decay": args.afterpulse_decay,
            "dead_time_pulses": args.dead_time_pulses,
        },
    }

    # Update artifacts
    if "artifacts" not in report:
        report["artifacts"] = {}
    report["artifacts"]["decoy_key_rate_plot"] = "decoy_key_rate_vs_loss.png"
    if realism_enabled:
        report["artifacts"]["decoy_key_rate_realism_plot"] = "decoy_key_rate_vs_loss_realism.png"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")  # Ensure trailing newline
    print("Wrote:", report_path)


def _run_pass_sweep(args: argparse.Namespace) -> None:
    """Execute satellite pass sweep command with free-space link model."""
    # Handle day/night toggle
    is_night = args.is_night and not args.day
    background_mode = "night" if is_night else "day"

    # Build link parameters
    link_params = FreeSpaceLinkParams(
        wavelength_m=args.wavelength,
        tx_diameter_m=args.tx_diameter,
        rx_diameter_m=args.rx_diameter,
        sigma_point_rad=args.sigma_point,
        altitude_m=args.altitude,
        atm_loss_db_zenith=args.atm_loss_db,
        sigma_ln=args.sigma_ln if args.turbulence else 0.0,
        system_loss_db=args.system_loss_db,
        is_night=is_night,
        day_background_factor=args.day_bg_factor,
    )

    optical_params = OpticalParams(
        filter_bandwidth_nm=args.filter_bandwidth_nm,
        detector_temp_c=args.detector_temp_c,
        mode=background_mode,
    )
    p_bg_rate = background_rate_hz(optical_params, base_rate_hz=args.p_bg)
    p_bg_rate = dark_count_rate_hz(optical_params, base_rate_hz=p_bg_rate)
    detector = DetectorParams(eta=args.eta, p_bg=p_bg_rate)
    calibration = _load_calibration_model(args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)
    (outdir / "reports").mkdir(exist_ok=True)

    # Generate elevation profile
    time_s, elevation_deg = generate_elevation_profile(
        max_elevation_deg=args.max_elevation,
        min_elevation_deg=args.min_elevation,
        time_step_s=args.time_step,
        pass_duration_s=args.pass_duration,
    )

    pulses_schedule, total_sent = _resolve_pass_pulse_accounting(args, len(time_s))
    rep_rate_hz = args.rep_rate if args.rep_rate is not None else total_sent / args.pass_duration

    print(f"Simulating pass sweep: {len(time_s)} time points, "
          f"elevation {args.min_elevation:.1f}° to {args.max_elevation:.1f}°")
    print(f"Mode: {'night' if is_night else 'day'}, turbulence: {args.turbulence}")

    # Run legacy Monte Carlo sweep for backwards-compatible report
    records_mc, summary_mc = sweep_pass(
        elevation_deg_values=elevation_deg,
        time_s_values=time_s,
        flip_prob=args.flip_prob,
        attack="none",
        n_pulses_per_step=pulses_schedule,
        seed=args.seed,
        detector=detector,
        link_params=link_params,
        turbulence=args.turbulence,
    )

    finite_key_params = None
    if args.finite_key:
        finite_key_params = FiniteKeyParams(
            eps_pe=args.eps_pe,
            eps_sec=args.eps_sec,
            eps_cor=args.eps_cor,
            ec_efficiency=args.ec_efficiency,
            pe_frac=args.pe_frac,
            m_pe=args.m_pe,
        )

    fading_params = None
    if args.fading:
        fading_params = FadingParams(
            enabled=True,
            sigma_ln=args.fading_sigma_ln,
            n_samples=args.fading_samples,
            seed=args.fading_seed,
        )

    pointing_params = None
    if args.pointing:
        pointing_params = PointingParams(
            acq_seconds=args.acq_seconds,
            dropout_prob=args.dropout_prob,
            relock_seconds=args.relock_seconds,
            pointing_jitter_urad=args.pointing_jitter_urad,
            seed=args.pointing_seed,
        )

    pass_params = PassModelParams(
        max_elevation_deg=args.max_elevation,
        min_elevation_deg=args.min_elevation,
        pass_seconds=args.pass_duration,
        dt_seconds=args.time_step,
        flip_prob=args.flip_prob,
        rep_rate_hz=rep_rate_hz,
        qber_abort_threshold=0.11,
        background_mode=background_mode,
    )

    records_pass, summary_pass = compute_pass_records(
        params=pass_params,
        link_params=link_params,
        detector=detector,
        finite_key=finite_key_params,
        calibration=calibration,
        fading=fading_params,
        pointing=pointing_params,
    )

    summary_base = None
    if args.fading:
        _, summary_base = compute_pass_records(
            params=pass_params,
            link_params=link_params,
            detector=detector,
            finite_key=finite_key_params,
            calibration=calibration,
            fading=None,
        )

    # Generate plots
    elev_plot_path = plot_key_rate_vs_elevation(
        records_pass,
        str(outdir / "figures" / "key_rate_vs_elevation.png"),
    )
    print("Plot:", elev_plot_path)

    secure_plot_path = plot_secure_window(
        records_pass,
        str(outdir / "figures" / "secure_window_per_pass.png"),
        secure_start_s=summary_pass["secure_window"]["t_start_seconds"],
        secure_end_s=summary_pass["secure_window"]["t_end_seconds"],
    )
    print("Plot:", secure_plot_path)

    secure_plot_alt_path = plot_secure_window(
        records_pass,
        str(outdir / "figures" / "secure_window.png"),
        secure_start_s=summary_pass["secure_window"]["t_start_seconds"],
        secure_end_s=summary_pass["secure_window"]["t_end_seconds"],
    )
    print("Plot:", secure_plot_alt_path)

    if args.fading:
        fading_samples = sample_fading_transmittance(
            sigma_ln=args.fading_sigma_ln,
            n_samples=max(200, args.fading_samples),
            seed=args.fading_seed,
        )
        fading_plot_path = plot_eta_fading_samples(
            fading_samples,
            str(outdir / "figures" / "eta_fading_samples.png"),
        )
        print("Plot:", fading_plot_path)
        eta_plot_path = plot_eta_samples(
            fading_samples,
            str(outdir / "figures" / "eta_samples.png"),
        )
        print("Plot:", eta_plot_path)
        if summary_base is not None:
            impact_plot_path = plot_secure_window_impact(
                summary_base["secure_window_seconds"],
                summary_pass["secure_window_seconds"],
                str(outdir / "figures" / "secure_window_fading_impact.png"),
            )
            print("Plot:", impact_plot_path)

    loss_plot_path = plot_loss_vs_elevation(
        records_pass,
        str(outdir / "figures" / "loss_vs_elevation.png"),
    )
    print("Plot:", loss_plot_path)

    if args.filter_bandwidth_nm > 0:
        bw_vals = np.linspace(0.5, max(0.5, args.filter_bandwidth_nm * 2.0), 10)
        bg_vals = [
            dark_count_rate_hz(
                OpticalParams(
                    filter_bandwidth_nm=float(bw),
                    detector_temp_c=args.detector_temp_c,
                    mode=background_mode,
                ),
                base_rate_hz=background_rate_hz(
                    OpticalParams(
                        filter_bandwidth_nm=float(bw),
                        detector_temp_c=args.detector_temp_c,
                        mode=background_mode,
                    ),
                    base_rate_hz=args.p_bg,
                ),
            )
            for bw in bw_vals
        ]
        bg_plot_path = plot_background_rate_vs_bandwidth(
            bw_vals,
            np.array(bg_vals, dtype=float),
            str(outdir / "figures" / "background_rate_vs_bandwidth.png"),
        )
        print("Plot:", bg_plot_path)

    if args.pointing:
        lock_plot_path = plot_pointing_lock_state(
            records_pass,
            str(outdir / "figures" / "pointing_lock_state.png"),
        )
        print("Plot:", lock_plot_path)
        trans_plot_path = plot_transmittance_with_pointing(
            records_pass,
            str(outdir / "figures" / "transmittance_with_pointing.png"),
        )
        print("Plot:", trans_plot_path)

    # Build report
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pass_sweep": {
            "records": records_mc,
            "summary": summary_mc,
        },
        "parameters": {
            "max_elevation_deg": args.max_elevation,
            "min_elevation_deg": args.min_elevation,
            "pass_duration_s": args.pass_duration,
            "time_step_s": args.time_step,
            "flip_prob": args.flip_prob,
            "pulses": total_sent,
            "pulses_per_step": pulses_schedule,
            "n_sent_total": total_sent,
            "rep_rate": args.rep_rate,
            "eta": args.eta,
            "p_bg": args.p_bg,
            "p_bg_effective": p_bg_rate,
            "is_night": is_night,
            "turbulence": args.turbulence,
            "link_params": {
                "wavelength_m": link_params.wavelength_m,
                "tx_diameter_m": link_params.tx_diameter_m,
                "rx_diameter_m": link_params.rx_diameter_m,
                "sigma_point_rad": link_params.sigma_point_rad,
                "altitude_m": link_params.altitude_m,
                "atm_loss_db_zenith": link_params.atm_loss_db_zenith,
                "sigma_ln": link_params.sigma_ln,
                "system_loss_db": link_params.system_loss_db,
                "day_background_factor": link_params.day_background_factor,
            },
        },
        "artifacts": {
            "key_rate_vs_elevation_plot": "key_rate_vs_elevation.png",
            "secure_window_plot": "secure_window_per_pass.png",
            "loss_vs_elevation_plot": "loss_vs_elevation.png",
        },
    }

    report_path = outdir / "reports" / "latest.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print("Wrote:", report_path)

    pass_time_series = records_to_time_series(
        records_pass,
        include_ci=finite_key_params is not None or args.fading,
    )
    def _count_secure_fragments(records):
        fragments = 0
        in_window = False
        for rec in records:
            active = rec.get("key_rate_per_pulse", 0.0) > 0.0
            if active and not in_window:
                fragments += 1
            in_window = active
        return fragments

    fading_variance = None
    fragment_stats = None
    if args.fading:
        eta_samples = sample_fading_transmittance(
            sigma_ln=args.fading_sigma_ln,
            n_samples=max(200, args.fading_samples),
            seed=args.fading_seed,
        )
        fading_variance = float(np.var(eta_samples, ddof=1)) if eta_samples.size > 1 else 0.0
        if args.fading_trials > 1:
            fragment_trials = []
            for trial in range(args.fading_trials):
                fading_trial = FadingParams(
                    enabled=True,
                    sigma_ln=args.fading_sigma_ln,
                    n_samples=args.fading_samples,
                    seed=args.fading_seed + trial,
                )
                recs_trial, _ = compute_pass_records(
                    params=pass_params,
                    link_params=link_params,
                    detector=detector,
                    finite_key=finite_key_params,
                    calibration=calibration,
                    fading=fading_trial,
                    pointing=pointing_params,
                )
                fragment_trials.append(_count_secure_fragments(recs_trial))
            frag_mean = float(np.mean(fragment_trials)) if fragment_trials else 0.0
            frag_std = float(np.std(fragment_trials, ddof=1)) if len(fragment_trials) > 1 else 0.0
            fragment_stats = {
                "mean": frag_mean,
                "ci_low": frag_mean - 1.96 * frag_std,
                "ci_high": frag_mean + 1.96 * frag_std,
                "trials": fragment_trials,
            }
    incidents = attribute_incidents(
        pass_time_series,
        detect_change_points(
            pass_time_series,
            metrics=("qber_mean", "headroom", "key_rate_bps"),
        ),
    )
    assumptions = [
        "loss_db derived from elevation profile using elevation_to_loss model",
        f"background_model daynight mode set to {background_mode}",
    ]
    if args.finite_key:
        assumptions.append("finite_key enabled implies Hoeffding bounds for QBER")
    if args.fading:
        assumptions.append("fading_model lognormal applied to transmittance samples")
    if args.pointing:
        assumptions.append("pointing_model applies acquisition/track/dropout transmittance multiplier")
    assumptions.append("optical_chain scales background with filter bandwidth and temperature")

    plots = {
        "key_rate_vs_elevation": "figures/key_rate_vs_elevation.png",
        "secure_window": "figures/secure_window.png",
    }
    if args.fading:
        plots["eta_fading_samples"] = "figures/eta_fading_samples.png"
        plots["secure_window_fading_impact"] = "figures/secure_window_fading_impact.png"
        plots["eta_samples"] = "figures/eta_samples.png"
    if args.filter_bandwidth_nm > 0:
        plots["background_rate_vs_bandwidth"] = "figures/background_rate_vs_bandwidth.png"
    if args.pointing:
        plots["pointing_lock_state"] = "figures/pointing_lock_state.png"
        plots["transmittance_with_pointing"] = "figures/transmittance_with_pointing.png"

    pass_report = {
        "schema_version": "1.0",
        "mode": "pass-sweep",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "rep_rate_hz": float(rep_rate_hz),
            "pass_seconds": float(args.pass_duration),
            "dt_seconds": float(args.time_step),
            "loss_model": {
                "type": "elevation_to_loss",
                "params": {
                    "min_elevation_deg": float(args.min_elevation),
                    "max_elevation_deg": float(args.max_elevation),
                },
            },
            "background_model": {
                "type": "daynight",
                "params": {
                    "mode": background_mode,
                },
            },
            "optical_chain": {
                "filter_bandwidth_nm": float(args.filter_bandwidth_nm),
                "detector_temp_c": float(args.detector_temp_c),
                "background_rate_used": float(p_bg_rate),
            },
            "finite_key": {
                "enabled": bool(args.finite_key),
                "epsilon_sec": float(args.eps_sec),
                "epsilon_cor": float(args.eps_cor),
                "pe_fraction": float(args.pe_frac),
            },
            "fading_model": {
                "enabled": bool(args.fading),
                "type": "lognormal",
                "params": {
                    "sigma_ln": float(args.fading_sigma_ln),
                    "samples": int(args.fading_samples),
                    "trials": int(args.fading_trials),
                    "seed": int(args.fading_seed),
                },
            },
            "pointing_model": {
                "enabled": bool(args.pointing),
                "acq_seconds": float(args.acq_seconds),
                "dropout_prob": float(args.dropout_prob),
                "relock_seconds": float(args.relock_seconds),
                "pointing_jitter_urad": float(args.pointing_jitter_urad),
                "seed": int(args.pointing_seed),
            },
        },
        "units": {
            "rep_rate_hz": "Hz",
            "pass_seconds": "s",
            "dt_seconds": "s",
            "elevation_deg": "deg",
            "loss_db": "dB",
            "key_rate_per_pulse": "bits/pulse",
            "key_rate_bps": "bits/s",
            "secret_bits": "bits",
            "qber": "unitless",
            "fading_variance": "unitless",
        },
        "time_series": pass_time_series,
        "summary": summary_pass,
        "incidents": incidents,
        "artifacts": {
            "plots": plots,
        },
        "assumptions": assumptions,
    }
    if fading_variance is not None:
        pass_report["summary"]["fading_variance"] = fading_variance
    if fragment_stats is not None:
        pass_report["summary"]["secure_window_fragments_mean"] = fragment_stats["mean"]
        pass_report["summary"]["secure_window_fragments_ci_low"] = fragment_stats["ci_low"]
        pass_report["summary"]["secure_window_fragments_ci_high"] = fragment_stats["ci_high"]
        pass_report["summary"]["secure_window_fragments_trials"] = fragment_stats["trials"]

    pass_report_path = outdir / "reports" / "latest_pass.json"
    with open(pass_report_path, "w") as f:
        json.dump(pass_report, f, indent=2)
        f.write("\n")
    print("Wrote:", pass_report_path)


def _run_experiment(args: argparse.Namespace) -> None:
    """Execute blinded A/B experiment harness."""
    outdir = Path(args.outdir)
    params = ExperimentParams(
        seed=args.seed,
        n_blocks=args.n_blocks,
        block_seconds=args.block_seconds,
        rep_rate_hz=args.rep_rate_hz,
        pass_seconds=args.pass_seconds,
    )

    finite_key_params = None
    if args.finite_key:
        finite_key_params = FiniteKeyParams(
            eps_sec=args.eps_sec,
            eps_cor=args.eps_cor,
        )

    output = run_experiment(
        params=params,
        metrics=args.metrics,
        outdir=outdir,
        finite_key=finite_key_params,
        bell_mode=args.bell_mode,
        unblind=args.unblind,
    )
    print("Wrote:", outdir / "reports" / "latest_experiment.json")
    if args.unblind:
        print("Wrote:", outdir / "reports" / "schedule_unblinded.json")
    print("Wrote:", outdir / "reports" / "schedule_blinded.json")


def _run_forecast_run(args: argparse.Namespace) -> None:
    """Execute forecast ingestion + blinded scoring harness."""
    outdir = Path(args.outdir)
    run_forecast_harness(
        forecasts_path=args.forecasts,
        outdir=outdir,
        seed=args.seed,
        n_blocks=args.n_blocks,
        block_seconds=args.block_seconds,
        rep_rate_hz=args.rep_rate_hz,
        unblind=args.unblind,
        estimate_identifiability=args.estimate_identifiability,
    )
    print("Wrote:", outdir / "reports" / "forecast_blinded.json")
    if args.unblind:
        print("Wrote:", outdir / "reports" / "forecast_unblinded.json")


def _run_coincidence_sim(args: argparse.Namespace) -> None:
    """Execute coincidence simulation with time-tagging."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)
    (outdir / "reports").mkdir(exist_ok=True)

    loss_values = np.linspace(args.loss_min, args.loss_max, args.steps)
    sigma_s = args.jitter_ps * 1e-12
    tau_s = args.tau_ps * 1e-12
    dead_time_s = args.dead_time_ps * 1e-12
    dead_time_stream_s = args.dead_time_ns * 1e-9
    afterpulse_window_s = args.afterpulse_window_ps * 1e-12
    afterpulse_decay_s = args.afterpulse_decay_ps * 1e-12
    tdc_seconds = args.tdc_ps * 1e-12

    rng = np.random.default_rng(args.seed)
    records = []

    optical_params = OpticalParams(
        filter_bandwidth_nm=args.filter_bandwidth_nm,
        detector_temp_c=args.detector_temp_c,
        mode="night",
    )
    bg_rate_effective = background_rate_hz(optical_params, args.background_rate_hz)
    bg_rate_effective = dark_count_rate_hz(optical_params, bg_rate_effective)

    for idx, loss_db in enumerate(loss_values):
        eta = 10 ** (-float(loss_db) / 10.0)
        expected_pairs = args.pair_rate_hz * args.duration * (eta ** 2)
        n_pairs = int(rng.poisson(expected_pairs))

        if args.stream_mode:
            stream_params = StreamParams(
                duration_s=args.duration,
                pair_rate_hz=args.pair_rate_hz * (eta ** 2),
                background_rate_hz=bg_rate_effective,
                gate_duty_cycle=args.gate_duty_cycle,
                dead_time_s=dead_time_stream_s,
                afterpulse_prob=args.afterpulse_prob,
                afterpulse_window_s=afterpulse_window_s,
                afterpulse_decay_s=afterpulse_decay_s,
                eta_z=1.0,
                eta_x=1.0,
                misalignment_prob=0.0,
                jitter_sigma_s=sigma_s,
                seed=args.seed + idx,
            )
            tags_a, tags_b = generate_event_stream(stream_params)
        else:
            sig_a, sig_b = generate_pair_time_tags(
                n_pairs=n_pairs,
                duration_s=args.duration,
                sigma_a=sigma_s,
                sigma_b=sigma_s,
                seed=args.seed + idx,
            )
            bg_a = generate_background_time_tags(
                rate_hz=bg_rate_effective,
                duration_s=args.duration,
                sigma=sigma_s,
                seed=args.seed + 1000 + idx * 2,
            )
            bg_b = generate_background_time_tags(
                rate_hz=bg_rate_effective,
                duration_s=args.duration,
                sigma=sigma_s,
                seed=args.seed + 1000 + idx * 2 + 1,
            )
            tags_a = merge_time_tags(sig_a, bg_a)
            tags_b = merge_time_tags(sig_b, bg_b)

            if args.afterpulse_prob > 0.0 and afterpulse_window_s > 0.0:
                tags_a = add_afterpulsing(
                    tags_a,
                    p_afterpulse=args.afterpulse_prob,
                    window_s=afterpulse_window_s,
                    decay=afterpulse_decay_s,
                    seed=args.seed + 2000 + idx * 2,
                )
                tags_b = add_afterpulsing(
                    tags_b,
                    p_afterpulse=args.afterpulse_prob,
                    window_s=afterpulse_window_s,
                    decay=afterpulse_decay_s,
                    seed=args.seed + 2000 + idx * 2 + 1,
                )

            if dead_time_s > 0.0:
                tags_a = apply_dead_time(tags_a, dead_time_s)
                tags_b = apply_dead_time(tags_b, dead_time_s)

        timing_model = TimingModel(
            delta_t=args.clock_offset_s,
            drift_ppm=args.clock_drift_ppm,
            tdc_seconds=tdc_seconds,
            jitter_sigma_s=0.0,
        )
        result = match_coincidences(
            tags_a,
            tags_b,
            tau_seconds=tau_s,
            timing_model=timing_model,
            estimate_offset=args.estimate_offset,
        )
        obs = compute_observables(
            correlation_counts={
                "AB": result.matrices["Z"],
                "ABp": result.matrices["X"],
                "ApB": result.matrices["Z"],
                "ApBp": result.matrices["X"],
            },
            n_pairs=max(1, n_pairs),
            min_visibility=args.min_visibility,
            min_chsh_s=args.min_chsh_s,
        )
        records.append({
            "loss_db": float(loss_db),
            "coincidences": int(result.coincidences),
            "accidentals": int(result.accidentals),
            "car": float(result.car),
            "matrices": result.matrices,
            "visibility": float(obs.visibility),
            "chsh_s": float(obs.chsh_s),
            "chsh_sigma": float(obs.chsh_sigma),
            "correlations": obs.correlations,
            "aborted": bool(obs.aborted),
            "estimated_clock_offset_s": result.estimated_offset_s,
        })

    plot_path = plot_car_vs_loss(
        records,
        str(outdir / "figures" / "car_vs_loss.png"),
    )
    print("Plot:", plot_path)
    chsh_plot_path = plot_chsh_s_vs_loss(
        records,
        str(outdir / "figures" / "chsh_s_vs_loss.png"),
    )
    print("Plot:", chsh_plot_path)
    vis_plot_path = plot_visibility_vs_loss(
        records,
        str(outdir / "figures" / "visibility_vs_loss.png"),
    )
    print("Plot:", vis_plot_path)

    cars = [r["car"] for r in records if np.isfinite(r["car"])]
    summary = {
        "max_car": float(max(cars)) if cars else 0.0,
        "mean_car": float(np.mean(cars)) if cars else 0.0,
        "min_visibility": args.min_visibility,
        "min_chsh_s": args.min_chsh_s,
    }

    report = {
        "schema_version": "1.0",
        "mode": "coincidence-sim",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "loss_min": args.loss_min,
            "loss_max": args.loss_max,
            "steps": args.steps,
            "duration_seconds": args.duration,
            "pair_rate_hz": args.pair_rate_hz,
            "background_rate_hz": args.background_rate_hz,
            "background_rate_used": bg_rate_effective,
            "jitter_ps": args.jitter_ps,
            "tau_ps": args.tau_ps,
            "dead_time_ps": args.dead_time_ps,
            "afterpulse_prob": args.afterpulse_prob,
            "afterpulse_window_ps": args.afterpulse_window_ps,
            "afterpulse_decay_ps": args.afterpulse_decay_ps,
            "clock_offset_s": args.clock_offset_s,
            "clock_drift_ppm": args.clock_drift_ppm,
            "tdc_ps": args.tdc_ps,
            "estimate_offset": bool(args.estimate_offset),
            "stream_mode": bool(args.stream_mode),
            "gate_duty_cycle": args.gate_duty_cycle,
            "dead_time_ns": args.dead_time_ns,
            "seed": args.seed,
        },
        "records": records,
        "summary": summary,
        "timing_model": {
            "delta_t": args.clock_offset_s,
            "drift_ppm": args.clock_drift_ppm,
            "tdc_seconds": tdc_seconds,
            "estimated_clock_offset_s": records[-1]["estimated_clock_offset_s"] if args.estimate_offset else None,
        },
        "optical_chain": {
            "filter_bandwidth_nm": float(args.filter_bandwidth_nm),
            "detector_temp_c": float(args.detector_temp_c),
            "background_rate_used": float(bg_rate_effective),
        },
        "stream_model": {
            "enabled": bool(args.stream_mode),
            "gate_duty_cycle": args.gate_duty_cycle,
            "dead_time_ns": args.dead_time_ns,
            "afterpulse_prob": args.afterpulse_prob,
            "afterpulse_window_ps": args.afterpulse_window_ps,
            "afterpulse_decay_ps": args.afterpulse_decay_ps,
        },
        "artifacts": {
            "car_plot": "figures/car_vs_loss.png",
            "chsh_plot": "figures/chsh_s_vs_loss.png",
            "visibility_plot": "figures/visibility_vs_loss.png",
        },
    }

    report_path = outdir / "reports" / "latest_coincidence.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print("Wrote:", report_path)


def _run_calibration_fit(args: argparse.Namespace) -> None:
    """Execute telemetry calibration fitting workflow."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "reports").mkdir(exist_ok=True)

    records = load_telemetry(args.telemetry)

    p_bg_grid = np.linspace(args.p_bg_min, args.p_bg_max, args.grid_steps)
    flip_grid = np.linspace(args.flip_min, args.flip_max, args.grid_steps)
    eta_scale_grid = np.linspace(args.eta_scale_min, args.eta_scale_max, args.grid_steps)

    fit = fit_telemetry_parameters(
        records=records,
        eta_base=args.eta_base,
        p_bg_grid=p_bg_grid,
        flip_grid=flip_grid,
        eta_scale_grid=eta_scale_grid,
    )

    predicted = predict_with_uncertainty(records, fit, args.eta_base)

    params_output = {
        "eta_base": args.eta_base,
        "eta_scale": fit.eta_scale,
        "p_bg": fit.p_bg,
        "flip_prob": fit.flip_prob,
        "rmse": fit.rmse,
        "residual_std": fit.residual_std,
    }
    if fit.clock_offset_s is not None:
        params_output["clock_offset_s"] = fit.clock_offset_s
    if fit.pointing_jitter_sigma is not None:
        params_output["pointing_jitter_sigma"] = fit.pointing_jitter_sigma
    if fit.background_rate is not None:
        params_output["background_rate"] = fit.background_rate

    params_path = outdir / "reports" / "calibration_params.json"
    with open(params_path, "w") as f:
        json.dump(params_output, f, indent=2)
        f.write("\n")
    print("Wrote:", params_path)

    report = {
        "schema_version": "1.0",
        "mode": "calibration-fit",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "telemetry_path": args.telemetry,
            "eta_base": args.eta_base,
            "p_bg_min": args.p_bg_min,
            "p_bg_max": args.p_bg_max,
            "flip_min": args.flip_min,
            "flip_max": args.flip_max,
            "eta_scale_min": args.eta_scale_min,
            "eta_scale_max": args.eta_scale_max,
            "grid_steps": args.grid_steps,
        },
        "fit": params_output,
        "predicted": predicted,
        "artifacts": {
            "calibration_params": "reports/calibration_params.json",
        },
    }

    report_path = outdir / "reports" / "calibration_fit.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print("Wrote:", report_path)


def _run_constellation_sweep(args: argparse.Namespace) -> None:
    """Execute constellation pass scheduling and inventory simulation."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "reports").mkdir(exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)

    horizon_seconds = float(args.horizon_hours) * 3600.0
    schedule = schedule_passes(
        n_sats=args.n_sats,
        n_stations=args.n_stations,
        horizon_seconds=horizon_seconds,
        passes_per_sat=args.passes_per_sat,
        pass_duration_s=args.pass_duration_s,
        seed=args.seed,
    )
    inventory_series = simulate_inventory(
        schedule=schedule,
        horizon_seconds=horizon_seconds,
        initial_bits=args.initial_bits,
        production_bits_per_pass=args.production_bits_per_pass,
        consumption_bps=args.consumption_bps,
    )

    schedule_records = [
        {
            "satellite_id": event.satellite_id,
            "station_id": event.station_id,
            "t_start_s": event.t_start_s,
            "t_end_s": event.t_end_s,
        }
        for event in schedule
    ]

    inventory_plot_path = plot_inventory_timeseries(
        inventory_series["t_seconds"],
        inventory_series["inventory_bits"],
        str(outdir / "figures" / "inventory_timeseries.png"),
    )
    flow_plot_path = plot_inventory_flow(
        inventory_series["t_seconds"],
        inventory_series["produced_bits"],
        inventory_series["consumed_bits"],
        str(outdir / "figures" / "inventory_flow.png"),
    )

    report = {
        "schema_version": "1.0",
        "mode": "constellation-sweep",
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "n_sats": int(args.n_sats),
            "n_stations": int(args.n_stations),
            "horizon_hours": float(args.horizon_hours),
            "passes_per_sat": int(args.passes_per_sat),
            "pass_duration_s": float(args.pass_duration_s),
            "initial_bits": float(args.initial_bits),
            "production_bits_per_pass": float(args.production_bits_per_pass),
            "consumption_bps": float(args.consumption_bps),
            "seed": int(args.seed),
        },
        "schedule": schedule_records,
        "inventory": inventory_series,
        "summary": {
            "n_passes": len(schedule_records),
            "total_produced_bits": inventory_series["produced_bits"][-1] if inventory_series["produced_bits"] else 0.0,
            "total_consumed_bits": inventory_series["consumed_bits"][-1] if inventory_series["consumed_bits"] else 0.0,
            "final_inventory_bits": inventory_series["inventory_bits"][-1] if inventory_series["inventory_bits"] else 0.0,
        },
        "artifacts": {
            "inventory_plot": "figures/inventory_timeseries.png",
            "flow_plot": "figures/inventory_flow.png",
        },
    }

    report_path = outdir / "reports" / "constellation_inventory.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print("Wrote:", report_path)
    print("Plot:", inventory_plot_path)
    print("Plot:", flow_plot_path)


if __name__ == "__main__":
    main()
