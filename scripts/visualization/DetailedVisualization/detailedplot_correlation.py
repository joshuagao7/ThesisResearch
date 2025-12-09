"""Detailed plotting for correlation algorithm - automatically processes all data files."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from jump_detection.algorithms.correlation import CorrelationParameters, detect_correlation_jumps
from jump_detection.annotations import load_annotations
from jump_detection.config import (
    DATASET_ROOT,
    CORRELATION_BUFFER_SIZE_DEFAULT,
    CORRELATION_NEGATIVE_FRAMES_DEFAULT,
    CORRELATION_ZERO_FRAMES_DEFAULT,
    CORRELATION_POSITIVE_FRAMES_DEFAULT,
    CORRELATION_THRESHOLD_DEFAULT,
    SAMPLING_RATE,
)
from jump_detection.analysis.precise import calculate_precise_jump_boundaries

import matplotlib.pyplot as plt
import numpy as np

PRIMARY_RED = '#C41E3A'
PRIMARY_ORANGE = '#FF6B35'
PRIMARY_YELLOW = '#FFD23F'
PRIMARY_BLUE = '#0066CC'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def discover_all_data_files(dataset_root: Path = DATASET_ROOT) -> list[tuple[Path, str]]:
    """Discover all data files from Test0, Test1(100Hz), Test2, and Test3 (video) folders.
    
    Returns:
        List of tuples (file_path, participant_folder_name)
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    
    data_files = []
    test_folders = ["Test0", "Test1(100Hz)", "Test2", "Test3 (video)"]
    
    for folder_name in test_folders:
        folder_path = root / folder_name
        if not folder_path.exists():
            continue
        
        if folder_name in ["Test2", "Test3 (video)", "Test1(100Hz)"]:
            files = sorted([
                path for path in folder_path.glob("*.txt")
                if not path.name.endswith("_annotations.json")
            ])
        else:
            files = sorted([
                path for path in folder_path.iterdir()
                if path.is_file() and not path.name.endswith("_annotations.json") and not path.name.endswith(".json")
            ])
        
        for file_path in files:
            data_files.append((file_path, folder_name))
    
    return data_files


def plot_correlation_pipeline(
    result,
    *,
    search_window: int = 70,
    show: bool = True,
    ground_truth_markers: list[int] | None = None,
) -> plt.Figure:
    """Plot the correlation-based detection pipeline."""
    signals = result.signals
    raw_data = signals["raw_data"]
    pooled = signals["pooled"]
    derivative = signals.get("derivative", np.gradient(pooled))
    correlation = signals["correlation"]
    template = signals["template"]
    
    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    fig.suptitle(
        f"Jump Detection Pipeline (Correlation Algorithm) - {result.participant_name or 'Participant'}",
        fontsize=18,
    )
    
    # Plot 1: Raw sensor data
    axes[0].set_ylabel("Raw Sensors", fontsize=12)
    for channel in range(raw_data.shape[1]):
        axes[0].plot(raw_data[:, channel], linewidth=0.5, alpha=0.9)
    axes[0].set_title("Raw Sensor Data", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Pooled sensor data and derivative
    axes[1].set_ylabel("Signal Value", fontsize=12)
    axes[1].plot(pooled, linewidth=1.0, color=PRIMARY_BLUE, label="Pooled Data", alpha=0.7)
    axes[1].plot(derivative, linewidth=1.0, color=PRIMARY_ORANGE, label="Derivative", alpha=0.8)
    axes[1].axhline(y=0, color="grey", linestyle="-", linewidth=0.5, alpha=0.3)
    axes[1].set_title("Pooled Sensor Signal and Derivative", fontsize=14)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Dot product values
    axes[2].set_ylabel("Dot Product Value", fontsize=12)
    axes[2].set_xlabel("Frame Index", fontsize=12)
    
    valid_correlation = ~np.isnan(correlation)
    if valid_correlation.any():
        axes[2].plot(
            correlation,
            linewidth=1.5,
            color=PRIMARY_YELLOW,
            label="Dot Product (Template · Derivative)",
        )
        axes[2].axhline(y=0, color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
        
        params = result.metadata.get("parameters", {})
        threshold = params.get("correlation_threshold")
        if threshold is not None:
            threshold = float(threshold)
            axes[2].axhline(
                y=threshold,
                color=PRIMARY_RED,
                linestyle="--",
                alpha=0.8,
                linewidth=1.5,
                label=f"Dot Product Threshold ({threshold:.3f})",
            )
            axes[2].text(
                0.99,
                threshold,
                f"Threshold: {threshold:.3f}",
                color=PRIMARY_RED,
                fontsize=9,
                ha="right",
                va="bottom",
                transform=axes[2].get_yaxis_transform(),
                backgroundcolor="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
    
    axes[2].set_title(
        f"Dot Product Between Template and Derivative (Template: {len(template)} frames "
        f"[-1×{np.sum(template == -1)}, 0×{np.sum(template == 0)}, 1×{np.sum(template == 1)}])",
        fontsize=14,
    )
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)
    
    # Highlight ground truth markers if available
    if ground_truth_markers:
        for marker in ground_truth_markers:
            if 0 <= marker < len(pooled):
                for ax in axes:
                    ax.axvline(
                        marker,
                        color=PRIMARY_RED,
                        linewidth=2,
                        alpha=0.7,
                        linestyle="--",
                        label="Ground Truth Marker" if marker == ground_truth_markers[0] and ax == axes[0] else "",
                    )
                axes[1].plot(
                    marker,
                    pooled[marker],
                    "*",
                    color=PRIMARY_RED,
                    markersize=12,
                )
        if ground_truth_markers:
            axes[0].legend(loc="upper right")
    
    # Highlight detected jumps
    for jump in result.jumps:
        for ax in axes:
            ax.axvspan(
                jump.start,
                jump.end,
                alpha=0.2,
                color=PRIMARY_YELLOW,
            )
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def process_all_files(
    params: CorrelationParameters | None = None,
    search_window: int = 70,
    save_plots: bool = True,
) -> None:
    """Process all data files and generate detailed plots for correlation algorithm."""
    if params is None:
        params = CorrelationParameters(
            buffer_size=CORRELATION_BUFFER_SIZE_DEFAULT,
            negative_frames=CORRELATION_NEGATIVE_FRAMES_DEFAULT,
            zero_frames=CORRELATION_ZERO_FRAMES_DEFAULT,
            positive_frames=CORRELATION_POSITIVE_FRAMES_DEFAULT,
            correlation_threshold=CORRELATION_THRESHOLD_DEFAULT,
        )
    
    data_files = discover_all_data_files()
    print(f"Found {len(data_files)} data files to process")
    
    project_root = Path(__file__).parent.parent.parent.parent
    save_dir = project_root / "results" / "plots" / "pipeline" / "correlation"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path, folder_name in data_files:
        participant_name = file_path.stem
        print(f"\n{'='*80}")
        print(f"Processing: {participant_name} ({folder_name})")
        print(f"{'='*80}")
        
        try:
            # Load annotations if available
            annotations = load_annotations(file_path)
            ground_truth_markers = annotations.markers if annotations else None
            
            # Detect jumps
            result = detect_correlation_jumps(
                file_path,
                participant_name,
                params=params,
                save_windows=False,
            )
            
            print(f"Correlation algorithm parameters: "
                  f"buffer_size={params.buffer_size}, "
                  f"negative_frames={params.negative_frames}, "
                  f"zero_frames={params.zero_frames}, "
                  f"positive_frames={params.positive_frames}, "
                  f"correlation_threshold={params.correlation_threshold:.3f}")
            if ground_truth_markers:
                print(f"Ground truth markers: {len(ground_truth_markers)} markers loaded")
            print(f"Detected jumps: {len(result.jumps)}")
            
            # Generate pipeline plot
            if save_plots:
                fig = plot_correlation_pipeline(
                    result,
                    search_window=search_window,
                    ground_truth_markers=ground_truth_markers,
                    show=False,
                )
                save_path = save_dir / f"{participant_name}_pipeline.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved pipeline plot to {save_path}")
                
        except Exception as e:
            print(f"Error processing {participant_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"Completed processing {len(data_files)} files")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Use default parameters
    params = CorrelationParameters(
        buffer_size=CORRELATION_BUFFER_SIZE_DEFAULT,
        negative_frames=CORRELATION_NEGATIVE_FRAMES_DEFAULT,
        zero_frames=CORRELATION_ZERO_FRAMES_DEFAULT,
        positive_frames=CORRELATION_POSITIVE_FRAMES_DEFAULT,
        correlation_threshold=CORRELATION_THRESHOLD_DEFAULT,
    )
    
    process_all_files(params=params, save_plots=True)

