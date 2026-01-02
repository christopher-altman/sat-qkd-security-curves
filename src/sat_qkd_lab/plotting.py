from __future__ import annotations
from typing import Sequence, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt


def _extract(records: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    return np.array([r[key] for r in records], dtype=float)


def plot_key_metrics_vs_loss(
    records_no_attack: Sequence[Dict[str, Any]],
    records_attack: Sequence[Dict[str, Any]],
    out_prefix: str,
) -> Tuple[str, str]:
    """
    Plot QBER and secret fraction vs loss for BB84 with and without attack.

    Parameters
    ----------
    records_no_attack : Sequence[Dict[str, Any]]
        Sweep results without attack.
    records_attack : Sequence[Dict[str, Any]]
        Sweep results with intercept-resend attack.
    out_prefix : str
        Output path prefix for plots.

    Returns
    -------
    Tuple[str, str]
        Paths to QBER plot and key fraction plot.
    """
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
    # Canonical filename (cleaner than legacy key_key_fraction_vs_loss.png)
    k_path = f"{out_prefix}_fraction_vs_loss.png"
    plt.savefig(k_path, dpi=200, bbox_inches="tight")
    plt.close()

    return q_path, k_path


# --- Confidence Interval Plotting ---

def plot_qber_vs_loss_ci(
    records_no_attack: Sequence[Dict[str, Any]],
    records_attack: Sequence[Dict[str, Any]],
    out_path: str,
) -> str:
    """
    Plot QBER vs loss with 95% confidence interval bands.

    Parameters
    ----------
    records_no_attack : Sequence[Dict[str, Any]]
        Sweep results with CI fields (from sweep_loss_with_ci).
    records_attack : Sequence[Dict[str, Any]]
        Sweep results with CI fields.
    out_path : str
        Output path for the plot.

    Returns
    -------
    str
        Path to the saved plot.
    """
    loss = _extract(records_no_attack, "loss_db")

    # No attack
    q_no_mean = _extract(records_no_attack, "qber_mean")
    q_no_lo = _extract(records_no_attack, "qber_ci_low")
    q_no_hi = _extract(records_no_attack, "qber_ci_high")

    # With attack
    q_ev_mean = _extract(records_attack, "qber_mean")
    q_ev_lo = _extract(records_attack, "qber_ci_low")
    q_ev_hi = _extract(records_attack, "qber_ci_high")

    fig, ax = plt.subplots()

    # Plot no attack
    ax.plot(loss, q_no_mean, label="QBER (no attack)")
    ax.fill_between(loss, q_no_lo, q_no_hi, alpha=0.3)

    # Plot with attack
    ax.plot(loss, q_ev_mean, label="QBER (intercept-resend)")
    ax.fill_between(loss, q_ev_lo, q_ev_hi, alpha=0.3)

    ax.set_xlabel("Channel loss (dB)")
    ax.set_ylabel("QBER")
    ax.set_title("QBER vs loss with 95% CI (BB84)")
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_key_rate_vs_loss_ci(
    records_no_attack: Sequence[Dict[str, Any]],
    records_attack: Sequence[Dict[str, Any]],
    out_path: str,
) -> str:
    """
    Plot secret fraction vs loss with 95% confidence interval bands.

    Parameters
    ----------
    records_no_attack : Sequence[Dict[str, Any]]
        Sweep results with CI fields (from sweep_loss_with_ci).
    records_attack : Sequence[Dict[str, Any]]
        Sweep results with CI fields.
    out_path : str
        Output path for the plot.

    Returns
    -------
    str
        Path to the saved plot.
    """
    loss = _extract(records_no_attack, "loss_db")

    # No attack
    sf_no_mean = _extract(records_no_attack, "secret_fraction_mean")
    sf_no_lo = _extract(records_no_attack, "secret_fraction_ci_low")
    sf_no_hi = _extract(records_no_attack, "secret_fraction_ci_high")

    # With attack
    sf_ev_mean = _extract(records_attack, "secret_fraction_mean")
    sf_ev_lo = _extract(records_attack, "secret_fraction_ci_low")
    sf_ev_hi = _extract(records_attack, "secret_fraction_ci_high")

    fig, ax = plt.subplots()

    # Plot no attack
    ax.plot(loss, sf_no_mean, label="Secret fraction (no attack)")
    ax.fill_between(loss, sf_no_lo, sf_no_hi, alpha=0.3)

    # Plot with attack
    ax.plot(loss, sf_ev_mean, label="Secret fraction (intercept-resend)")
    ax.fill_between(loss, sf_ev_lo, sf_ev_hi, alpha=0.3)

    ax.set_xlabel("Channel loss (dB)")
    ax.set_ylabel("Asymptotic secret fraction")
    ax.set_title("Secret fraction vs loss with 95% CI (BB84)")
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


# --- Decoy-State Plotting ---

def plot_decoy_key_rate_vs_loss(
    records: Sequence[Dict[str, Any]],
    out_path: str,
    show_ci: bool = False,
) -> str:
    """
    Plot decoy-state BB84 key rate vs loss.

    Parameters
    ----------
    records : Sequence[Dict[str, Any]]
        Decoy sweep results.
    out_path : str
        Output path for the plot.
    show_ci : bool
        If True and records contain std fields, show error bars.

    Returns
    -------
    str
        Path to the saved plot.
    """
    loss = _extract(records, "loss_db")
    key_rate = _extract(records, "key_rate_asymptotic")

    fig, ax = plt.subplots()

    if show_ci and "key_rate_std" in records[0]:
        key_std = _extract(records, "key_rate_std")
        ax.errorbar(loss, key_rate, yerr=1.96 * key_std, capsize=3,
                    label="Decoy BB84 key rate")
    else:
        ax.plot(loss, key_rate, marker="o", markersize=3,
                label="Decoy BB84 key rate")

    ax.set_xlabel("Channel loss (dB)")
    ax.set_ylabel("Asymptotic key rate (per pulse)")
    ax.set_title("Decoy-State BB84: Key Rate vs Loss")
    ax.legend()

    # Use log scale only if all values are non-negative and at least one is positive
    has_positive = any(k > 0 for k in key_rate)
    all_nonnegative = all(k >= 0 for k in key_rate)

    if has_positive and all_nonnegative:
        # For log scale with zeros, set a floor slightly below min positive value
        positive_rates = [k for k in key_rate if k > 0]
        if positive_rates:
            floor = min(positive_rates) * 0.1
            ax.set_ylim(bottom=floor)
            ax.set_yscale("log")
    else:
        # Linear scale for all-zero or negative values
        ax.set_ylim(bottom=0)
        ax.set_yscale("linear")

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_decoy_comparison(
    records_bb84: Sequence[Dict[str, Any]],
    records_decoy: Sequence[Dict[str, Any]],
    out_path: str,
) -> str:
    """
    Compare standard BB84 and decoy-state BB84 key rates.

    Parameters
    ----------
    records_bb84 : Sequence[Dict[str, Any]]
        Standard BB84 sweep results.
    records_decoy : Sequence[Dict[str, Any]]
        Decoy-state BB84 sweep results.
    out_path : str
        Output path for the plot.

    Returns
    -------
    str
        Path to the saved plot.
    """
    loss_bb84 = _extract(records_bb84, "loss_db")
    loss_decoy = _extract(records_decoy, "loss_db")

    # Standard BB84 uses secret_fraction
    sf_bb84 = _extract(records_bb84, "secret_fraction")

    # Decoy uses key_rate_asymptotic (already per-pulse)
    # To compare fairly, we'd need same normalization
    # For now, just plot both
    kr_decoy = _extract(records_decoy, "key_rate_asymptotic")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Standard BB84
    ax1.plot(loss_bb84, sf_bb84, marker="o", markersize=3)
    ax1.set_xlabel("Channel loss (dB)")
    ax1.set_ylabel("Secret fraction")
    ax1.set_title("Standard BB84")
    ax1.set_ylim(0, 1.05)

    # Decoy BB84
    ax2.plot(loss_decoy, kr_decoy, marker="o", markersize=3)
    ax2.set_xlabel("Channel loss (dB)")
    ax2.set_ylabel("Key rate (per pulse)")
    ax2.set_title("Decoy-State BB84")
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path
