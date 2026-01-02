from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
import numpy as np
from pathlib import Path

SCHEMA_VERSION = "0.2"

from .helpers import validate_int, validate_float, validate_seed
from .sweep import sweep_loss, sweep_loss_with_ci, compute_summary_stats
from .plotting import (
    plot_key_metrics_vs_loss,
    plot_qber_vs_loss_ci,
    plot_key_rate_vs_loss_ci,
    plot_decoy_key_rate_vs_loss,
)
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

    args = p.parse_args()

    # Validate arguments post-parse
    _validate_args(args)

    if args.cmd == "sweep":
        _run_sweep(args)
    elif args.cmd == "decoy-sweep":
        _run_decoy_sweep(args)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments after parsing."""
    # Common validations for all commands
    validate_float("loss-min", args.loss_min, min_value=0.0)
    validate_float("loss-max", args.loss_max, min_value=0.0)
    if args.loss_min > args.loss_max:
        raise ValueError(f"loss-min ({args.loss_min}) > loss-max ({args.loss_max})")
    validate_int("steps", args.steps, min_value=1)
    validate_float("flip-prob", args.flip_prob, min_value=0.0, max_value=0.5)
    validate_int("pulses", args.pulses, min_value=1)
    validate_seed(args.seed)
    validate_float("eta", args.eta, min_value=0.0, max_value=1.0)
    validate_float("p-bg", args.p_bg, min_value=0.0, max_value=1.0)
    validate_int("trials", args.trials, min_value=1)

    # Command-specific validations
    if args.cmd == "sweep":
        validate_int("workers", args.workers, min_value=1)
    elif args.cmd == "decoy-sweep":
        validate_float("mu-s", args.mu_s, min_value=0.0)
        validate_float("mu-d", args.mu_d, min_value=0.0)
        validate_float("mu-v", args.mu_v, min_value=0.0, max_value=0.0)
        validate_float("p-s", args.p_s, min_value=0.0, max_value=1.0)
        validate_float("p-d", args.p_d, min_value=0.0, max_value=1.0)
        validate_float("p-v", args.p_v, min_value=0.0, max_value=1.0)


def _run_sweep(args: argparse.Namespace) -> None:
    """Execute BB84 sweep command."""
    loss_vals = np.linspace(args.loss_min, args.loss_max, args.steps)
    detector = DetectorParams(eta=args.eta, p_bg=args.p_bg)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)
    (outdir / "reports").mkdir(exist_ok=True)

    if args.trials > 1:
        # Run with Monte Carlo CI
        print(f"Running sweep with {args.trials} trials per point...")
        no_attack = sweep_loss_with_ci(
            loss_vals,
            flip_prob=args.flip_prob,
            attack="none",
            n_pulses=args.pulses,
            seed=args.seed,
            n_trials=args.trials,
            detector=detector,
            n_workers=args.workers,
        )
        attack = sweep_loss_with_ci(
            loss_vals,
            flip_prob=args.flip_prob,
            attack="intercept_resend",
            n_pulses=args.pulses,
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
                "pulses": args.pulses,
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
            n_pulses=args.pulses,
            seed=args.seed,
            detector=detector,
        )
        attack = sweep_loss(
            loss_vals,
            flip_prob=args.flip_prob,
            attack="intercept_resend",
            n_pulses=args.pulses,
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
                "pulses": args.pulses,
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

    print(f"Running decoy-state BB84 sweep...")
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


if __name__ == "__main__":
    main()
