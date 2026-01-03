from __future__ import annotations
from typing import Sequence, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt


def _extract(records: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    return np.array([r[key] for r in records], dtype=float)


def _key_rate_scale_settings(values: Sequence[float]) -> Tuple[str, Dict[str, float]]:
    if not values:
        return "linear", {"bottom": 0}

    positives = [v for v in values if v > 0]

    if positives and all(v > 0 for v in values):
        return "log", {"bottom": min(positives) * 0.1}

    if positives and all(v >= 0 for v in values):
        linthresh = max(min(positives) * 0.1, 1e-15)
        return "symlog", {"linthresh": linthresh, "linscale": 1.0, "bottom": 0}

    return "linear", {"bottom": 0}


def plot_key_metrics_vs_loss(
    records_no_attack: Sequence[Dict[str, Any]],
    records_attack: Sequence[Dict[str, Any]],
    out_prefix: str,
    attack_label: Optional[str] = None,
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
    if attack_label and attack_label != "none":
        plt.title(f"QBER vs loss (BB84, attack={attack_label})")
    else:
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
    if attack_label and attack_label != "none":
        plt.title(f"Secret fraction vs loss (BB84, attack={attack_label})")
    else:
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
    attack_label: Optional[str] = None,
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
    if attack_label and attack_label != "none":
        ax.set_title(f"QBER vs loss with 95% CI (BB84, attack={attack_label})")
    else:
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
    attack_label: Optional[str] = None,
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
    if attack_label and attack_label != "none":
        ax.set_title(f"Secret fraction vs loss with 95% CI (BB84, attack={attack_label})")
    else:
        ax.set_title("Secret fraction vs loss with 95% CI (BB84)")
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_inventory_timeseries(
    t_seconds: Sequence[float],
    inventory_bits: Sequence[float],
    out_path: str,
) -> str:
    """Plot key inventory over time."""
    fig, ax = plt.subplots()
    ax.plot(t_seconds, inventory_bits, label="Inventory")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Key inventory (bits)")
    ax.set_title("Key inventory over time")
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_inventory_flow(
    t_seconds: Sequence[float],
    produced_bits: Sequence[float],
    consumed_bits: Sequence[float],
    out_path: str,
) -> str:
    """Plot cumulative production and consumption over time."""
    fig, ax = plt.subplots()
    ax.plot(t_seconds, produced_bits, label="Produced")
    ax.plot(t_seconds, consumed_bits, label="Consumed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative bits")
    ax.set_title("Key production and consumption")
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_clock_sync_diagnostics(
    t_seconds: Sequence[float],
    residuals: Sequence[float],
    out_path: str,
) -> str:
    """Plot clock sync residuals over time and as a histogram."""
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=False)
    axes[0].plot(t_seconds, residuals, color="tab:blue")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Residual (s)")
    axes[0].set_title("Clock sync residuals")
    axes[0].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")

    axes[1].hist(residuals, bins=24, color="tab:orange", alpha=0.8)
    axes[1].set_xlabel("Residual (s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual histogram")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_fading_evolution(
    t_seconds: Sequence[float],
    transmittance: Sequence[float],
    out_path: str,
) -> str:
    """Plot OU fading transmittance evolution over time."""
    fig, ax = plt.subplots()
    ax.plot(t_seconds, transmittance, color="tab:green")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Transmittance")
    ax.set_title("OU fading evolution")
    ax.set_ylim(bottom=0.0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_secure_window_fragmentation(
    t_seconds: Sequence[float],
    secure_mask: Sequence[bool],
    out_path: str,
) -> str:
    """Plot secure window availability over time."""
    fig, ax = plt.subplots()
    values = [1 if s else 0 for s in secure_mask]
    ax.step(t_seconds, values, where="post", color="tab:purple")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Secure window (1/0)")
    ax.set_title("Secure window fragmentation")
    ax.set_ylim(-0.1, 1.1)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_basis_bias_vs_elevation(
    elevation_deg: Sequence[float],
    bias_values: Sequence[float],
    out_path: str,
) -> str:
    """Plot basis selection bias versus elevation."""
    fig, ax = plt.subplots()
    ax.plot(elevation_deg, bias_values, color="tab:red")
    ax.set_xlabel("Elevation (deg)")
    ax.set_ylabel("Basis bias (P(Z) - P(X))")
    ax.set_title("Motion-induced basis bias vs elevation")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_calibration_quality_card(
    r2: float,
    rmse: float,
    residual_std: float,
    condition_number: float,
    identifiable: bool,
    out_path: str,
) -> str:
    """Render a lightweight calibration quality card."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    status = "yes" if identifiable else "no"
    lines = [
        "Calibration quality card",
        f"R^2: {r2:.4f}",
        f"RMSE: {rmse:.4g}",
        f"Residual std: {residual_std:.4g}",
        f"FIM cond: {condition_number:.3g}",
        f"Identifiable: {status}",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_calibration_residuals(
    loss_db: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    residuals: np.ndarray,
    autocorr_lag1: float,
    out_path: str,
) -> str:
    """Plot observed vs predicted and residuals vs loss."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(observed, predicted, color="steelblue", alpha=0.8)
    min_val = float(min(observed.min(), predicted.min()))
    max_val = float(max(observed.max(), predicted.max()))
    axes[0].plot([min_val, max_val], [min_val, max_val], color="gray", linestyle="--")
    axes[0].set_xlabel("Observed QBER")
    axes[0].set_ylabel("Predicted QBER")
    axes[0].set_title("Observed vs predicted")

    axes[1].scatter(loss_db, residuals, color="darkorange", alpha=0.8)
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Loss (dB)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title(f"Residuals (autocorr={autocorr_lag1:.2f})")

    plt.tight_layout()
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


def plot_decoy_key_rate_vs_loss_comparison(
    baseline_records: Sequence[Dict[str, Any]],
    realism_records: Sequence[Dict[str, Any]],
    out_path: str,
    show_ci: bool = False,
) -> str:
    """
    Plot decoy-state BB84 key rate vs loss comparing baseline and realism.

    Uses log scale when all key rates are positive, symlog when zeros are
    present (to preserve true zeros without epsilon clipping), and linear when
    no positive values exist.

    Parameters
    ----------
    baseline_records : Sequence[Dict[str, Any]]
        Baseline decoy sweep results.
    realism_records : Sequence[Dict[str, Any]]
        Decoy sweep results with realism enabled.
    out_path : str
        Output path for the plot.
    show_ci : bool
        If True and records contain std fields, show error bars for realism.

    Returns
    -------
    str
        Path to the saved plot.
    """
    loss_base = _extract(baseline_records, "loss_db")
    loss_real = _extract(realism_records, "loss_db")
    key_base = _extract(baseline_records, "key_rate_asymptotic")
    key_real = _extract(realism_records, "key_rate_asymptotic")

    fig, ax = plt.subplots()

    ax.plot(loss_base, key_base, marker="o", markersize=3, label="Baseline")
    if show_ci and realism_records and "key_rate_std" in realism_records[0]:
        key_std = _extract(realism_records, "key_rate_std")
        ax.errorbar(loss_real, key_real, yerr=1.96 * key_std, capsize=3,
                    label="Realism")
    else:
        ax.plot(loss_real, key_real, marker="o", markersize=3, label="Realism")

    ax.set_xlabel("Channel loss (dB)")
    ax.set_ylabel("Asymptotic key rate (per pulse)")
    ax.set_title("Decoy-State BB84: Realism vs Baseline")
    ax.legend()

    combined = list(key_base) + list(key_real)
    scale, scale_params = _key_rate_scale_settings(combined)

    if scale == "log":
        ax.set_yscale("log")
        ax.set_ylim(bottom=scale_params["bottom"])
    elif scale == "symlog":
        ax.set_yscale("symlog",
                      linthresh=scale_params["linthresh"],
                      linscale=scale_params["linscale"])
        ax.set_ylim(bottom=scale_params["bottom"])
    else:
        ax.set_yscale("linear")
        ax.set_ylim(bottom=scale_params["bottom"])

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_attack_comparison_key_rate(
    records_by_attack: Dict[str, Sequence[Dict[str, Any]]],
    out_path: str,
) -> str:
    """
    Plot key rate vs loss for multiple attack modes.

    Parameters
    ----------
    records_by_attack : Dict[str, Sequence[Dict[str, Any]]]
        Mapping of attack name to sweep records.
    out_path : str
        Output path for the plot.

    Returns
    -------
    str
        Path to the saved plot.
    """
    fig, ax = plt.subplots()

    combined_rates: list[float] = []
    for attack_name, records in records_by_attack.items():
        loss = _extract(records, "loss_db")
        key_rate = _extract(records, "key_rate_per_pulse")
        ax.plot(loss, key_rate, marker="o", markersize=3, label=attack_name)
        combined_rates.extend(list(key_rate))

    ax.set_xlabel("Channel loss (dB)")
    ax.set_ylabel("Asymptotic key rate (per pulse)")
    ax.set_title("BB84: Key Rate vs Loss (Attack Comparison)")
    ax.legend()

    scale, scale_params = _key_rate_scale_settings(combined_rates)
    if scale == "log":
        ax.set_yscale("log")
        ax.set_ylim(bottom=scale_params["bottom"])
    elif scale == "symlog":
        ax.set_yscale("symlog",
                      linthresh=scale_params["linthresh"],
                      linscale=scale_params["linscale"])
        ax.set_ylim(bottom=scale_params["bottom"])
    else:
        ax.set_yscale("linear")
        ax.set_ylim(bottom=scale_params["bottom"])

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


# --- Finite-Key Plotting ---

def plot_finite_key_comparison(
    records: Sequence[Dict[str, Any]],
    out_path: str,
) -> str:
    """
    Plot asymptotic vs finite-key rate comparison.

    Parameters
    ----------
    records : Sequence[Dict[str, Any]]
        Finite-key sweep results.
    out_path : str
        Output path for the plot.

    Returns
    -------
    str
        Path to the saved plot.
    """
    loss = _extract(records, "loss_db")
    asymp_rate = _extract(records, "asymptotic_rate")
    finite_rate = _extract(records, "finite_rate")

    fig, ax = plt.subplots()

    ax.plot(loss, asymp_rate, label="Asymptotic rate", linestyle="--")
    ax.plot(loss, finite_rate, label="Finite-key rate", linestyle="-")

    ax.set_xlabel("Channel loss (dB)")
    ax.set_ylabel("Key rate (per pulse)")
    ax.set_title("Asymptotic vs Finite-Key Rate (BB84)")
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_finite_key_bits_vs_loss(
    records: Sequence[Dict[str, Any]],
    out_path: str,
) -> str:
    """
    Plot extractable secret bits vs channel loss.

    Parameters
    ----------
    records : Sequence[Dict[str, Any]]
        Finite-key sweep results.
    out_path : str
        Output path for the plot.

    Returns
    -------
    str
        Path to the saved plot.
    """
    loss = _extract(records, "loss_db")
    l_secret = _extract(records, "l_secret_bits")

    fig, ax = plt.subplots()

    ax.plot(loss, l_secret, marker="o", markersize=3)

    ax.set_xlabel("Channel loss (dB)")
    ax.set_ylabel("Extractable secret bits")
    ax.set_title("Finite-Key Secret Bits vs Loss (BB84)")
    ax.set_ylim(bottom=0)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_finite_size_penalty(
    records: Sequence[Dict[str, Any]],
    out_path: str,
) -> str:
    """
    Plot finite-size penalty factor vs channel loss.

    Parameters
    ----------
    records : Sequence[Dict[str, Any]]
        Finite-key sweep results.
    out_path : str
        Output path for the plot.

    Returns
    -------
    str
        Path to the saved plot.
    """
    loss = _extract(records, "loss_db")
    penalty = _extract(records, "finite_size_penalty")

    fig, ax = plt.subplots()

    ax.plot(loss, penalty, marker="o", markersize=3)

    ax.set_xlabel("Channel loss (dB)")
    ax.set_ylabel("Finite-size penalty (fraction)")
    ax.set_title("Finite-Size Penalty vs Loss (BB84)")
    ax.set_ylim(0, 1.05)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_finite_key_rate_vs_n_sent(
    records: Sequence[Dict[str, Any]],
    out_path: str,
    log_x: bool = True,
) -> str:
    """
    Plot finite-key rate per pulse vs total sent pulses.

    Parameters
    ----------
    records : Sequence[Dict[str, Any]]
        Finite-key results with n_sent and key_rate_per_pulse_finite.
    out_path : str
        Output path for the plot.
    log_x : bool
        If True, use log scale for n_sent.

    Returns
    -------
    str
        Path to the saved plot.
    """
    n_sent = _extract(records, "n_sent")
    key_rate = _extract(records, "key_rate_per_pulse_finite")

    fig, ax = plt.subplots()
    ax.plot(n_sent, key_rate, marker="o", markersize=3)

    ax.set_xlabel("Total pulses sent (n_sent)")
    ax.set_ylabel("Finite-key rate (per pulse)")
    ax.set_title("Finite-Key Rate vs Total Pulses (BB84)")
    ax.set_ylim(bottom=0)
    if log_x:
        ax.set_xscale("log")

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


# --- Pass Sweep Plotting ---

def plot_key_rate_vs_elevation(
    records: Sequence[Dict[str, Any]],
    out_path: str,
    show_day_night: bool = False,
    records_day: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """
    Plot key rate vs elevation angle for satellite pass.

    Parameters
    ----------
    records : Sequence[Dict[str, Any]]
        Pass sweep results (night or primary).
    out_path : str
        Output path for the plot.
    show_day_night : bool
        If True and records_day provided, show day/night comparison.
    records_day : Sequence[Dict[str, Any]], optional
        Day-time pass results for comparison.

    Returns
    -------
    str
        Path to the saved plot.
    """
    elevation = _extract(records, "elevation_deg")
    key_rate = _extract(records, "key_rate_per_pulse")

    fig, ax = plt.subplots()

    ax.plot(elevation, key_rate, label="Night" if show_day_night else "Key rate", marker=".")

    if show_day_night and records_day is not None:
        elevation_day = _extract(records_day, "elevation_deg")
        key_rate_day = _extract(records_day, "key_rate_per_pulse")
        ax.plot(elevation_day, key_rate_day, label="Day", marker=".", linestyle="--")

    ax.set_xlabel("Elevation angle (degrees)")
    ax.set_ylabel("Key rate (per pulse)")
    ax.set_title("Secret Key Rate vs Elevation (Satellite Pass)")
    if show_day_night:
        ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=90)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_secure_window(
    records: Sequence[Dict[str, Any]],
    out_path: str,
    secure_start_s: Optional[float] = None,
    secure_end_s: Optional[float] = None,
) -> str:
    """
    Plot key rate over time with secure window highlighted.

    Parameters
    ----------
    records : Sequence[Dict[str, Any]]
        Pass sweep results with time_s and key_rate fields.
    out_path : str
        Output path for the plot.
    secure_start_s : float, optional
        Start of secure window in seconds.
    secure_end_s : float, optional
        End of secure window in seconds.

    Returns
    -------
    str
        Path to the saved plot.
    """
    time_s = _extract(records, "time_s")
    key_rate = _extract(records, "key_rate_per_pulse")

    fig, ax1 = plt.subplots()

    # Plot key rate
    ax1.plot(time_s, key_rate, "b-", label="Key rate")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Key rate (per pulse)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_ylim(bottom=0)

    # Highlight secure window
    if secure_start_s is not None and secure_end_s is not None:
        ax1.axvspan(secure_start_s, secure_end_s, alpha=0.2, color="green",
                    label=f"Secure window: {secure_end_s - secure_start_s:.0f}s")

    # Add elevation on secondary axis
    if "elevation_deg" in records[0]:
        elevation = _extract(records, "elevation_deg")
        ax2 = ax1.twinx()
        ax2.plot(time_s, elevation, "r--", alpha=0.5, label="Elevation")
        ax2.set_ylabel("Elevation (degrees)", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.set_ylim(0, 90)

    ax1.set_title("Secure Communication Window")
    ax1.legend(loc="upper left")

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_loss_vs_elevation(
    records: Sequence[Dict[str, Any]],
    out_path: str,
) -> str:
    """
    Plot total link loss vs elevation angle.

    Parameters
    ----------
    records : Sequence[Dict[str, Any]]
        Pass sweep results with elevation_deg and loss_db.
    out_path : str
        Output path for the plot.

    Returns
    -------
    str
        Path to the saved plot.
    """
    elevation = _extract(records, "elevation_deg")
    loss = _extract(records, "loss_db")

    fig, ax = plt.subplots()

    ax.plot(elevation, loss, marker=".")

    ax.set_xlabel("Elevation angle (degrees)")
    ax.set_ylabel("Total link loss (dB)")
    ax.set_title("Free-Space Link Loss vs Elevation")
    ax.set_xlim(0, 90)
    ax.invert_yaxis()  # Higher loss at bottom

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_car_vs_loss(
    records: Sequence[Dict[str, Any]],
    out_path: str,
) -> str:
    """
    Plot coincidence-to-accidental ratio (CAR) vs loss.

    Parameters
    ----------
    records : Sequence[Dict[str, Any]]
        Coincidence results with loss_db and car.
    out_path : str
        Output path for the plot.

    Returns
    -------
    str
        Path to the saved plot.
    """
    loss = _extract(records, "loss_db")
    car = _extract(records, "car")

    finite = np.isfinite(car)
    if np.any(finite):
        max_finite = np.max(car[finite])
        car_plot = np.where(finite, car, max_finite * 1.5)
    else:
        car_plot = np.zeros_like(car)

    fig, ax = plt.subplots()
    ax.plot(loss, car_plot, marker="o", markersize=3)
    ax.set_xlabel("Channel loss (dB)")
    ax.set_ylabel("CAR (coincidence / accidental)")
    ax.set_title("CAR vs loss")
    ax.set_ylim(bottom=0)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_chsh_s_vs_loss(
    records: Sequence[Dict[str, Any]],
    out_path: str,
) -> str:
    """
    Plot CHSH S vs loss.
    """
    loss = _extract(records, "loss_db")
    s_vals = _extract(records, "chsh_s")

    fig, ax = plt.subplots()
    ax.plot(loss, s_vals, marker="o", markersize=3)
    ax.axhline(2.0, color="red", linestyle="--", linewidth=1, label="Classical bound")
    ax.set_xlabel("Channel loss (dB)")
    ax.set_ylabel("CHSH S")
    ax.set_title("CHSH S vs loss")
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_visibility_vs_loss(
    records: Sequence[Dict[str, Any]],
    out_path: str,
) -> str:
    """
    Plot visibility vs loss.
    """
    loss = _extract(records, "loss_db")
    vis = _extract(records, "visibility")

    fig, ax = plt.subplots()
    ax.plot(loss, vis, marker="o", markersize=3)
    ax.set_xlabel("Channel loss (dB)")
    ax.set_ylabel("Visibility")
    ax.set_title("Visibility vs loss")
    ax.set_ylim(bottom=0, top=1.0)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def plot_eta_fading_samples(
    samples: np.ndarray,
    out_path: str,
) -> str:
    """
    Plot histogram of fading samples.
    """
    fig, ax = plt.subplots()
    ax.hist(samples, bins=30, color="steelblue", alpha=0.8)
    ax.set_xlabel("Fading transmittance factor")
    ax.set_ylabel("Count")
    ax.set_title("Fading samples (lognormal)")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_secure_window_impact(
    base_seconds: float,
    fading_seconds: float,
    out_path: str,
) -> str:
    """
    Plot secure window duration impact (baseline vs fading).
    """
    fig, ax = plt.subplots()
    ax.bar(["baseline", "fading"], [base_seconds, fading_seconds], color=["#4c72b0", "#dd8452"])
    ax.set_ylabel("Secure window (s)")
    ax.set_title("Secure window duration impact")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_pointing_lock_state(
    records: Sequence[Dict[str, Any]],
    out_path: str,
) -> str:
    """
    Plot pointing lock state over time.
    """
    time_s = _extract(records, "time_s")
    lock_state = np.array([1.0 if r.get("pointing_locked") else 0.0 for r in records])
    fig, ax = plt.subplots()
    ax.step(time_s, lock_state, where="post")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lock state")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Pointing lock state")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_sync_lock_state(
    duration_s: float,
    locked: bool,
    out_path: str,
) -> str:
    """
    Plot sync lock state over time.
    """
    time_s = np.array([0.0, max(0.0, float(duration_s))])
    lock_state = np.array([1.0 if locked else 0.0, 1.0 if locked else 0.0])
    fig, ax = plt.subplots()
    ax.step(time_s, lock_state, where="post")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lock state")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Sync lock state")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_transmittance_with_pointing(
    records: Sequence[Dict[str, Any]],
    out_path: str,
) -> str:
    """
    Plot transmittance multiplier from pointing over time.
    """
    time_s = _extract(records, "time_s")
    trans = np.array([r.get("pointing_transmittance", 0.0) for r in records], dtype=float)
    fig, ax = plt.subplots()
    ax.plot(time_s, trans, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Transmittance multiplier")
    ax.set_title("Transmittance with pointing")
    ax.set_ylim(bottom=0.0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_background_rate_vs_bandwidth(
    bandwidth_nm: np.ndarray,
    background_rate: np.ndarray,
    out_path: str,
) -> str:
    """
    Plot background rate vs filter bandwidth.
    """
    fig, ax = plt.subplots()
    ax.plot(bandwidth_nm, background_rate, marker="o", markersize=3)
    ax.set_xlabel("Filter bandwidth (nm)")
    ax.set_ylabel("Background rate (Hz)")
    ax.set_title("Background rate vs bandwidth")
    ax.set_ylim(bottom=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_background_rate_vs_time(
    time_s: np.ndarray,
    background_rate: np.ndarray,
    out_path: str,
) -> str:
    """
    Plot background rate vs time.
    """
    fig, ax = plt.subplots()
    ax.plot(time_s, background_rate, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Background rate (Hz)")
    ax.set_title("Background rate vs time")
    ax.set_ylim(bottom=0.0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_car_vs_time(
    time_s: np.ndarray,
    car_values: np.ndarray,
    out_path: str,
) -> str:
    """
    Plot CAR vs time.
    """
    fig, ax = plt.subplots()
    ax.plot(time_s, car_values, color="darkorange")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CAR")
    ax.set_title("CAR vs time")
    ax.set_ylim(bottom=0.0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return out_path


def plot_polarization_drift_vs_time(
    time_s: np.ndarray,
    angle_deg: np.ndarray,
    corrected_deg: np.ndarray | None,
    out_path: str,
) -> str:
    """
    Plot polarization drift angle vs time.
    """
    fig, ax = plt.subplots()
    ax.plot(time_s, angle_deg, color="steelblue", label="drift")
    if corrected_deg is not None:
        ax.plot(time_s, corrected_deg, color="darkorange", label="compensated")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Polarization drift vs time")
    ax.legend(loc="best")
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


def plot_qber_headroom_vs_loss(
    records: Sequence[Dict[str, Any]],
    out_path: str,
    qber_abort: float = 0.11,
    show_ci: bool = False,
) -> str:
    """
    Plot QBER headroom (distance to abort threshold) vs channel loss.

    Parameters
    ----------
    records : Sequence[Dict[str, Any]]
        Sweep results with QBER metrics.
    out_path : str
        Output path for the plot.
    qber_abort : float
        QBER abort threshold (default 0.11 = 11%).
    show_ci : bool
        Whether to show confidence interval bands.

    Returns
    -------
    str
        Path to the saved plot.
    """
    loss = _extract(records, "loss_db")

    # Check if CI data is available
    has_ci = "qber_ci_low" in records[0] and "qber_ci_high" in records[0]

    if has_ci and show_ci:
        qber_mean = _extract(records, "qber_mean")
        qber_ci_low = _extract(records, "qber_ci_low")
        qber_ci_high = _extract(records, "qber_ci_high")

        # Compute headroom for mean and CI bounds
        headroom_mean = qber_abort - qber_mean
        headroom_ci_low = qber_abort - qber_ci_high  # conservative
        headroom_ci_high = qber_abort - qber_ci_low  # optimistic

        plt.figure()
        plt.plot(loss, headroom_mean, label="Mean headroom", color="blue")
        plt.fill_between(
            loss,
            headroom_ci_low,
            headroom_ci_high,
            alpha=0.3,
            label="95% CI",
            color="blue",
        )
        plt.axhline(y=0, color="red", linestyle="--", linewidth=1, label="Abort threshold")
        plt.xlabel("Channel loss (dB)")
        plt.ylabel("QBER headroom (abort - QBER)")
        plt.title(f"Security headroom vs loss (abort threshold = {qber_abort:.2%})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        # Single trial or no CI data
        qber = _extract(records, "qber")
        headroom = qber_abort - qber

        plt.figure()
        plt.plot(loss, headroom, label="Headroom", color="blue")
        plt.axhline(y=0, color="red", linestyle="--", linewidth=1, label="Abort threshold")
        plt.xlabel("Channel loss (dB)")
        plt.ylabel("QBER headroom (abort - QBER)")
        plt.title(f"Security headroom vs loss (abort threshold = {qber_abort:.2%})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

    return out_path
