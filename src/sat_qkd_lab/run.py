from __future__ import annotations
import argparse
import math
import json
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
)
from .plotting import (
    plot_key_metrics_vs_loss,
    plot_qber_vs_loss_ci,
    plot_key_rate_vs_loss_ci,
    plot_decoy_key_rate_vs_loss,
    plot_finite_key_comparison,
    plot_finite_key_bits_vs_loss,
    plot_finite_size_penalty,
    plot_finite_key_rate_vs_n_sent,
    plot_key_rate_vs_elevation,
    plot_secure_window,
    plot_loss_vs_elevation,
)
from .free_space_link import FreeSpaceLinkParams, generate_elevation_profile
from .detector import DetectorParams, DEFAULT_DETECTOR
from .decoy_bb84 import DecoyParams, sweep_decoy_loss


def main() -> None:
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
    s.add_argument("--pass-seconds", type=float, default=None,
                   help="Pass duration in seconds used with --rep-rate")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--outdir", type=str, default=".")
    # New detector parameters
    s.add_argument("--eta", type=float, default=DEFAULT_DETECTOR.eta,
                   help="Detector efficiency (0..1)")
    s.add_argument("--p-bg", type=float, default=DEFAULT_DETECTOR.p_bg,
                   help="Background/dark click probability per pulse")
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
    d.add_argument("--p-s", type=float, default=0.8,
                   help="Probability of signal state")
    d.add_argument("--p-d", type=float, default=0.15,
                   help="Probability of decoy state")
    d.add_argument("--p-v", type=float, default=0.05,
                   help="Probability of vacuum state")
    d.add_argument("--trials", type=int, default=1,
                   help="Number of trials per loss value")

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

    args = p.parse_args()

    # Validate arguments post-parse
    _validate_args(args)

    if args.cmd == "sweep":
        _run_sweep(args)
    elif args.cmd == "decoy-sweep":
        _run_decoy_sweep(args)
    elif args.cmd == "pass-sweep":
        _run_pass_sweep(args)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments after parsing."""
    # Common validations for sweep/decoy-sweep commands
    if args.cmd in ("sweep", "decoy-sweep"):
        validate_float("loss-min", args.loss_min, min_value=0.0)
        validate_float("loss-max", args.loss_max, min_value=0.0)
        if args.loss_min > args.loss_max:
            raise ValueError(f"loss-min ({args.loss_min}) > loss-max ({args.loss_max})")
        validate_int("steps", args.steps, min_value=1)
        validate_int("trials", args.trials, min_value=1)

    # Common validations for all commands
    validate_float("flip-prob", args.flip_prob, min_value=0.0, max_value=0.5)
    validate_int("pulses", args.pulses, min_value=1)
    validate_seed(args.seed)
    validate_float("eta", args.eta, min_value=0.0, max_value=1.0)
    validate_float("p-bg", args.p_bg, min_value=0.0, max_value=1.0)

    # Command-specific validations
    if args.cmd == "sweep":
        validate_int("workers", args.workers, min_value=1)
        if args.n_sent is not None:
            validate_int("n-sent", args.n_sent, min_value=1)
        if args.rep_rate is not None:
            validate_float("rep-rate", args.rep_rate, min_value=1e-9)
        if args.pass_seconds is not None:
            validate_float("pass-seconds", args.pass_seconds, min_value=1e-9)
            if args.rep_rate is None:
                raise ValueError("pass-seconds requires --rep-rate")
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
    elif args.cmd == "decoy-sweep":
        validate_float("mu-s", args.mu_s, min_value=0.0)
        validate_float("mu-d", args.mu_d, min_value=0.0)
        validate_float("mu-v", args.mu_v, min_value=0.0, max_value=0.0)
        validate_float("p-s", args.p_s, min_value=0.0, max_value=1.0)
        validate_float("p-d", args.p_d, min_value=0.0, max_value=1.0)
        validate_float("p-v", args.p_v, min_value=0.0, max_value=1.0)
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
    detector = DetectorParams(eta=args.eta, p_bg=args.p_bg)
    n_sent = _resolve_n_sent_for_sweep(args)

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
            attack="intercept_resend",
            n_pulses=n_sent,
            seed=args.seed + 100_000,
            n_trials=args.trials,
            detector=detector,
            n_workers=args.workers,
        )

        # Generate CI plots
        qber_ci_path = plot_qber_vs_loss_ci(
            no_attack, attack,
            str(outdir / "figures" / "qber_vs_loss_ci.png")
        )
        # Canonical name for secret fraction CI plot
        sf_ci_path = plot_key_rate_vs_loss_ci(
            no_attack, attack,
            str(outdir / "figures" / "secret_fraction_vs_loss_ci.png")
        )
        # Legacy alias for backwards compatibility
        import shutil
        legacy_sf_ci_path = outdir / "figures" / "key_rate_vs_loss_ci.png"
        shutil.copy(sf_ci_path, legacy_sf_ci_path)
        print("CI Plots:", qber_ci_path, sf_ci_path)

        # Build report with CI data
        report = {
            "schema_version": SCHEMA_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "loss_sweep_ci": {
                "no_attack": no_attack,
                "intercept_resend": attack,
            },
            "summary_stats": {
                "no_attack": compute_summary_stats(no_attack),
                "intercept_resend": compute_summary_stats(attack),
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
                "p_bg": args.p_bg,
            },
            "artifacts": {
                "qber_ci_plot": "qber_vs_loss_ci.png",
                "secret_fraction_ci_plot": "secret_fraction_vs_loss_ci.png",
                "key_ci_plot": "key_rate_vs_loss_ci.png",  # legacy alias
            },
        }
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
            attack="intercept_resend",
            n_pulses=n_sent,
            seed=args.seed + 10_000,
            detector=detector,
        )

        q_path, k_path = plot_key_metrics_vs_loss(
            no_attack, attack,
            str(outdir / "figures" / "key")
        )
        # Legacy alias for backwards compatibility
        import shutil
        legacy_k_path = outdir / "figures" / "key_key_fraction_vs_loss.png"
        shutil.copy(k_path, legacy_k_path)
        print("Plots:", q_path, k_path)

        report = {
            "schema_version": SCHEMA_VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "loss_sweep": {
                "no_attack": no_attack,
                "intercept_resend": attack,
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
                "p_bg": args.p_bg,
            },
            "artifacts": {
                "qber_plot": str(Path(q_path).name),
                "secret_fraction_plot": str(Path(k_path).name),
                "key_fraction_plot": "key_key_fraction_vs_loss.png",  # legacy alias
            },
        }

    with open(outdir / "reports" / "latest.json", "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")  # Ensure trailing newline
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
        attack="intercept_resend",
        n_pulses=n_sent,
        seed=args.seed + 10_000,
        detector=detector,
        finite_key_params=fk_params,
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
            "intercept_resend": attack,
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
            "p_bg": args.p_bg,
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
    detector = DetectorParams(eta=args.eta, p_bg=args.p_bg)
    decoy = DecoyParams(
        mu_s=args.mu_s,
        mu_d=args.mu_d,
        mu_v=args.mu_v,
        p_s=args.p_s,
        p_d=args.p_d,
        p_v=args.p_v,
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
            "p_s": args.p_s,
            "p_d": args.p_d,
            "p_v": args.p_v,
            "eta": args.eta,
            "p_bg": args.p_bg,
        },
    }

    # Update artifacts
    if "artifacts" not in report:
        report["artifacts"] = {}
    report["artifacts"]["decoy_key_rate_plot"] = "decoy_key_rate_vs_loss.png"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")  # Ensure trailing newline
    print("Wrote:", report_path)


def _run_pass_sweep(args: argparse.Namespace) -> None:
    """Execute satellite pass sweep command with free-space link model."""
    # Handle day/night toggle
    is_night = args.is_night and not args.day

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

    detector = DetectorParams(eta=args.eta, p_bg=args.p_bg)

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

    print(f"Simulating pass sweep: {len(time_s)} time points, "
          f"elevation {args.min_elevation:.1f}° to {args.max_elevation:.1f}°")
    print(f"Mode: {'night' if is_night else 'day'}, turbulence: {args.turbulence}")

    # Run the sweep
    records, summary = sweep_pass(
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

    # Extract key rates for plotting
    key_rates = [r["secret_fraction"] for r in records]

    # Generate plots
    elev_plot_path = plot_key_rate_vs_elevation(
        records,
        str(outdir / "figures" / "key_rate_vs_elevation.png"),
    )
    print("Plot:", elev_plot_path)

    secure_plot_path = plot_secure_window(
        records,
        str(outdir / "figures" / "secure_window_per_pass.png"),
        secure_start_s=summary.get("secure_window_start_s"),
        secure_end_s=summary.get("secure_window_end_s"),
    )
    print("Plot:", secure_plot_path)

    loss_plot_path = plot_loss_vs_elevation(
        records,
        str(outdir / "figures" / "loss_vs_elevation.png"),
    )
    print("Plot:", loss_plot_path)

    # Build report
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "pass_sweep": {
            "records": records,
            "summary": summary,
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


if __name__ == "__main__":
    main()
