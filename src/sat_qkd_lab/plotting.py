from __future__ import annotations
from typing import Sequence, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

def _extract(records: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    return np.array([r[key] for r in records], dtype=float)

def plot_key_metrics_vs_loss(
    records_no_attack: Sequence[Dict[str, Any]],
    records_attack: Sequence[Dict[str, Any]],
    out_prefix: str,
) -> Tuple[str, str]:
    loss = _extract(records_no_attack, "loss_db")
    q_no = _extract(records_no_attack, "qber")
    q_ev = _extract(records_attack, "qber")
    sf_no = _extract(records_no_attack, "secret_fraction")
    sf_ev = _extract(records_attack, "secret_fraction")

    plt.figure()
    plt.plot(loss, q_no, label="QBER (no attack)")
    plt.plot(loss, q_ev, label="QBER (intercept-resend)")
    plt.xlabel("Channel loss (dB)")
    plt.ylabel("QBER")
    plt.title("QBER vs loss (BB84)")
    plt.legend()
    q_path = f"{out_prefix}_qber_vs_loss.png"
    plt.savefig(q_path, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(loss, sf_no, label="Secret fraction (no attack)")
    plt.plot(loss, sf_ev, label="Secret fraction (intercept-resend)")
    plt.xlabel("Channel loss (dB)")
    plt.ylabel("Asymptotic secret fraction")
    plt.title("Secret fraction vs loss (BB84)")
    plt.legend()
    k_path = f"{out_prefix}_key_fraction_vs_loss.png"
    plt.savefig(k_path, dpi=200, bbox_inches="tight")
    plt.close()

    return q_path, k_path
