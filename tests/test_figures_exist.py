"""Test that plotting helpers are properly integrated."""
from sat_qkd_lab.plotting import (
    apply_instrument_style,
    set_simulated_title,
    label_axes,
    safe_legend,
)
import matplotlib.pyplot as plt


def test_apply_instrument_style() -> None:
    """Test that instrument style helper applies without error."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    apply_instrument_style(ax)
    assert ax.get_xticklabels() is not None
    plt.close(fig)


def test_set_simulated_title() -> None:
    """Test that simulated title prefix is applied correctly."""
    fig, ax = plt.subplots()
    set_simulated_title(ax, "Test plot")
    title = ax.get_title()
    assert title.startswith("Simulated:"), f"Title does not start with 'Simulated:': {title}"
    assert "Test plot" in title, f"Title does not contain plot name: {title}"
    plt.close(fig)


def test_label_axes() -> None:
    """Test that axis labels are set correctly."""
    fig, ax = plt.subplots()
    label_axes(ax, "X axis (dB)", "Y axis (unitless)")
    assert ax.get_xlabel() == "X axis (dB)"
    assert ax.get_ylabel() == "Y axis (unitless)"
    plt.close(fig)


def test_safe_legend() -> None:
    """Test that safe legend is applied without error."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3], label="Test curve")
    safe_legend(ax)
    legend = ax.get_legend()
    assert legend is not None, "Legend not applied"
    plt.close(fig)
