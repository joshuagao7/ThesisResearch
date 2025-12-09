"""High-level plotting helpers for the detection pipelines."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

from ..types import DerivativeNames, DetectionResult, ThresholdNames
from ..utils import PRIMARY_RED, PRIMARY_ORANGE, PRIMARY_YELLOW, PRIMARY_BLUE, SEARCH_WINDOW

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def plot_threshold_pipeline(
    result: DetectionResult,
    *,
    search_window: int = SEARCH_WINDOW,
    show: bool = True,
) -> plt.Figure:
    signals = result.signals
    params = result.metadata.get("parameters", {})

    raw_data = signals[ThresholdNames.RAW_DATA.value]
    average = signals[ThresholdNames.AVERAGE.value]
    threshold_mask = signals[ThresholdNames.THRESHOLD_MASK.value]
    fig, axes = plt.subplots(4, 1, figsize=(20, 10), sharex=True)
    fig.suptitle(
        f"Jump Detection Pipeline (Threshold Algorithm) - {result.participant_name or 'Participant'}",
        fontsize=18,
    )

    axes[0].set_ylabel("Raw Sensors")
    for channel in range(raw_data.shape[1]):
        axes[0].plot(raw_data[:, channel], linewidth=0.5, alpha=0.9)

    axes[1].set_ylabel("Average + Threshold")
    axes[1].plot(average, linewidth=0.7, color=PRIMARY_BLUE)
    threshold = params.get("threshold")
    if threshold is not None:
        threshold = float(threshold)
        axes[1].axhline(
            y=threshold,
            color=PRIMARY_RED,
            linestyle="--",
            alpha=0.8,
            label=f"Threshold ({threshold:.3f})",
        )
        axes[1].legend()

    axes[2].set_ylabel("Threshold Mask")
    axes[2].plot(threshold_mask, linewidth=1.0, color=PRIMARY_ORANGE)

    axes[3].set_ylabel("Detections")
    for channel in range(raw_data.shape[1]):
        axes[3].plot(raw_data[:, channel], linewidth=0.5, alpha=0.3, color="grey")
    _highlight_jumps(axes[3], result, average, search_window)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def _plot_pooled_signals(ax, raw_data: np.ndarray, pooled: np.ndarray, in_air_threshold: Optional[float]) -> None:
    """Plot pooled, range, and moving average signals."""
    ax.plot(pooled, linewidth=0.7, color=PRIMARY_BLUE, label="Pooled (sum)")
    
    range_signal = raw_data.max(axis=1) - raw_data.min(axis=1)
    scale_factor = float(np.trapz(np.abs(pooled))) / float(np.trapz(np.abs(range_signal))) if np.trapz(np.abs(range_signal)) != 0 else 1.0
    ax.plot(range_signal * scale_factor, linewidth=0.7, color=PRIMARY_RED, alpha=0.7, label=f"Range (scaled ×{scale_factor:.2f})")
    
    moving_avg = np.convolve(pooled, np.ones(7)/7, mode='same')
    ax.plot(moving_avg, linewidth=1.2, color=PRIMARY_YELLOW, alpha=0.8, label="Moving average (window=7)")
    
    if in_air_threshold is not None:
        ax.axhline(y=in_air_threshold, color=PRIMARY_RED, linestyle="--", alpha=0.8)
        ax.text(0.99, in_air_threshold, f"{in_air_threshold:.3f}", color=PRIMARY_RED, fontsize=9,
                ha="right", va="bottom", transform=ax.get_yaxis_transform(), backgroundcolor="white")
    ax.legend(loc="upper right", fontsize=8)


def _plot_derivative_thresholds(ax, derivative: np.ndarray, upper_threshold: Optional[float], lower_threshold: Optional[float]) -> None:
    """Plot derivative with thresholds."""
    ax.plot(derivative, linewidth=1.0, color=PRIMARY_ORANGE)
    ax.set_ylim(-100, 100)
    ax.axhline(y=0, color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    
    if upper_threshold is not None:
        ax.axhline(y=upper_threshold, color=PRIMARY_RED, linestyle="--", alpha=0.8, linewidth=1.5, label=f"Upper ({upper_threshold:.3f})")
        ax.text(0.99, upper_threshold, f"Upper: {upper_threshold:.3f}", color=PRIMARY_RED, fontsize=9,
                ha="right", va="bottom", transform=ax.get_yaxis_transform(), backgroundcolor="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    if lower_threshold is not None:
        ax.axhline(y=lower_threshold, color=PRIMARY_RED, linestyle="--", alpha=0.8, linewidth=1.5, label=f"Lower ({lower_threshold:.3f})")
        ax.text(0.99, lower_threshold, f"Lower: {lower_threshold:.3f}", color=PRIMARY_RED, fontsize=9,
                ha="right", va="top", transform=ax.get_yaxis_transform(), backgroundcolor="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    if upper_threshold is not None or lower_threshold is not None:
        ax.legend(loc="upper right", fontsize=9)


def _compute_derivative_variants(pooled: np.ndarray, raw_data: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    """Compute various derivative calculation methods."""
    n = len(pooled)
    derivatives = [np.gradient(pooled)]
    labels = ["i - (i-1) [gradient]"]
    
    for k in range(2, 11):
        deriv = np.full(n, np.nan)
        deriv[k:] = pooled[k:] - pooled[:-k]
        derivatives.append(deriv)
        labels.append(f"i - (i-{k})")
    
    range_signal = raw_data.max(axis=1) - raw_data.min(axis=1)
    derivatives.append(np.gradient(range_signal))
    labels.append("Range derivative")
    
    moving_avg = np.convolve(pooled, np.ones(7)/7, mode='same')
    derivatives.append(np.gradient(moving_avg))
    labels.append("Moving avg derivative")
    
    return derivatives, labels


def _setup_interactive_legend(fig, ax, lines: list, labels: list[str]) -> None:
    """Setup interactive legend for toggling lines."""
    colors = [PRIMARY_ORANGE, PRIMARY_BLUE, PRIMARY_YELLOW, PRIMARY_RED, "#FF8C69", "#4A90E2",
              "#FFB84D", "#E63946", "#6C757D", "#17A2B8", "#6F42C1", "#20C997"]
    legend_handles = [
        Line2D([0], [0], color=colors[i % len(colors)], linewidth=1.5, marker='s', markersize=8,
               markeredgewidth=1, markerfacecolor=colors[i % len(colors)],
               markeredgecolor=colors[i % len(colors)], label=label)
        for i, label in enumerate(labels)
    ]
    
    legend = ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    legend_lines = list(legend.get_lines())
    handle_to_line = dict(zip(legend_lines, lines))
    line_colors = [line.get_color() for line in lines]
    
    def on_any_pick(event):
        legline = event.artist
        if legline in legend.get_texts():
            text_idx = list(legend.get_texts()).index(legline)
            if text_idx < len(legend_lines):
                legline = legend_lines[text_idx]
        if legline in handle_to_line:
            origline = handle_to_line[legline]
            visible = origline.get_visible()
            origline.set_visible(not visible)
            line_idx = lines.index(origline)
            line_color = line_colors[line_idx]
            legline.set_markerfacecolor(line_color if not visible else 'white')
            legline.set_markeredgecolor(line_color)
            fig.canvas.draw()
    
    for legline in legend_lines:
        legline.set_picker(10)
    for text in legend.get_texts():
        text.set_picker(10)
    fig.canvas.mpl_connect('pick_event', on_any_pick)


def _plot_ground_truth_markers(ax, pooled: np.ndarray, ground_truth_markers: Optional[list[int]]) -> None:
    """Plot ground truth markers if available."""
    if not ground_truth_markers:
        return
    for i, marker in enumerate(ground_truth_markers):
        if 0 <= marker < len(pooled):
            ax.axvline(marker, color=PRIMARY_RED, linewidth=2, alpha=0.7, linestyle="--",
                      label="Ground Truth Marker" if i == 0 else "")
            ax.plot(marker, pooled[marker], "*", color=PRIMARY_RED, markersize=12)
    ax.legend(loc="upper right")


def plot_derivative_pipeline(
    result: DetectionResult,
    *,
    search_window: int = SEARCH_WINDOW,
    show: bool = True,
    ground_truth_markers: Optional[list[int]] = None,
) -> plt.Figure:
    signals = result.signals
    params = result.metadata.get("parameters", {})
    raw_data = signals[DerivativeNames.RAW_DATA.value]
    pooled = signals[DerivativeNames.POOLED.value]
    derivative = signals[DerivativeNames.DERIVATIVE.value]

    # Use gridspec to make first plot taller (3:1:1 ratio)
    from matplotlib import gridspec
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    
    axes = []
    axes.append(fig.add_subplot(gs[0, 0]))
    axes.append(fig.add_subplot(gs[1, 0], sharex=axes[0]))
    axes.append(fig.add_subplot(gs[2, 0], sharex=axes[0]))
    
    fig.suptitle(f"Jump Detection Pipeline (Derivative Algorithm) - {result.participant_name or 'Participant'}", fontsize=18)

    # First plot: Multisensor raw data with highlighted jumps and ground truth markers
    axes[0].set_ylabel("Raw Multisensor Data with Detections", fontsize=12)
    # Plot all sensor channels
    for channel in range(raw_data.shape[1]):
        axes[0].plot(raw_data[:, channel], linewidth=0.5, alpha=0.4, color="grey")
    # Also plot pooled signal as a reference
    axes[0].plot(pooled, linewidth=1.5, color=PRIMARY_BLUE, label="Pooled Signal (sum)", alpha=0.8)
    _plot_ground_truth_markers(axes[0], pooled, ground_truth_markers)
    _highlight_jumps(axes[0], result, pooled, search_window)
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Second plot: Derivative with thresholds
    axes[1].set_ylabel("Derivative with Thresholds")
    _plot_derivative_thresholds(axes[1], derivative, params.get("upper_threshold"), params.get("lower_threshold"))
    axes[1].grid(True, alpha=0.3)

    # Third plot: All derivative variants, smoothed values, and jerk calculations
    axes[2].set_ylabel("Derivative Variants & Permutations (click legend to toggle)")
    derivatives, labels = _compute_derivative_variants(pooled, raw_data)
    
    # Add smoothed versions
    moving_avg = np.convolve(pooled, np.ones(7)/7, mode='same')
    derivatives.append(np.gradient(moving_avg))
    labels.append("Moving avg derivative (window=7)")
    
    moving_avg_15 = np.convolve(pooled, np.ones(15)/15, mode='same')
    derivatives.append(np.gradient(moving_avg_15))
    labels.append("Moving avg derivative (window=15)")
    
    # Add jerk (second derivative)
    jerk = np.gradient(derivative)
    derivatives.append(jerk)
    labels.append("Jerk (2nd derivative)")
    
    # Add smoothed jerk
    smoothed_jerk = np.gradient(np.gradient(moving_avg))
    derivatives.append(smoothed_jerk)
    labels.append("Smoothed jerk")
    
    colors = [PRIMARY_ORANGE, PRIMARY_BLUE, PRIMARY_YELLOW, PRIMARY_RED, "#FF8C69", "#4A90E2",
              "#FFB84D", "#E63946", "#6C757D", "#17A2B8", "#6F42C1", "#20C997", "#28A745",
              "#DC3545", "#FFC107", "#17A2B8"]
    lines = [axes[2].plot(deriv, linewidth=1.5, color=colors[i % len(colors)], alpha=0.5, label=label)[0]
             for i, (deriv, label) in enumerate(zip(derivatives, labels))]
    axes[2].set_ylim(-150, 150)
    axes[2].axhline(y=0, color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    _setup_interactive_legend(fig, axes[2], lines, labels)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def _highlight_jumps(ax, result: DetectionResult, signal: np.ndarray, search_window: int) -> None:
    text_height = ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] else 0.9
    for jump in result.iter_jumps():
        # Skip invalid jumps (zero or negative duration)
        if jump.start >= jump.end or jump.duration <= 0:
            continue
        # Ensure bounds are within signal range
        start = max(0, min(jump.start, len(signal) - 1))
        end = max(start + 1, min(jump.end, len(signal)))
        ax.axvspan(start, end, alpha=0.1, color=PRIMARY_BLUE)
        ax.text(jump.center, text_height, f"{jump.duration}f",
                ha="center", va="top", fontsize=10, fontweight="bold", color=PRIMARY_RED)


