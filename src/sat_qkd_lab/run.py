from __future__ import annotations
import argparse
import json
import numpy as np
from pathlib import Path

from .sweep import sweep_loss
from .plotting import plot_key_metrics_vs_loss

def main() -> None:
    p = argparse.ArgumentParser(prog="sat-qkd-security-curves")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="Sweep loss in dB and generate plots + report.")
    s.add_argument("--loss-min", type=float, default=20.0)
    s.add_argument("--loss-max", type=float, default=60.0)
    s.add_argument("--steps", type=int, default=21)
    s.add_argument("--flip-prob", type=float, default=0.005)
    s.add_argument("--pulses", type=int, default=200_000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--outdir", type=str, default=".")
    args = p.parse_args()

    if args.cmd == "sweep":
        loss_vals = np.linspace(args.loss_min, args.loss_max, args.steps)
        no_attack = sweep_loss(loss_vals, flip_prob=args.flip_prob, attack="none", n_pulses=args.pulses, seed=args.seed)
        attack = sweep_loss(loss_vals, flip_prob=args.flip_prob, attack="intercept_resend", n_pulses=args.pulses, seed=args.seed + 10_000)

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        q_path, k_path = plot_key_metrics_vs_loss(no_attack, attack, str(outdir / "figures" / "key"))
        (outdir / "figures").mkdir(exist_ok=True)
        (outdir / "reports").mkdir(exist_ok=True)

        report = {
            "loss_sweep": {
                "no_attack": no_attack,
                "intercept_resend": attack,
            },
            "artifacts": {
                "qber_plot": str(Path(q_path).name),
                "secret_fraction_plot": str(Path(k_path).name),
            },
        }
        with open(outdir / "reports" / "latest.json", "w") as f:
            json.dump(report, f, indent=2)

        print("Wrote:", outdir / "reports" / "latest.json")
        print("Plots:", q_path, k_path)

if __name__ == "__main__":
    main()
