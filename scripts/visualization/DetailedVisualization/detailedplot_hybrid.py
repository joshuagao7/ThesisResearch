"""Detailed plotting for hybrid algorithm - automatically processes all data files."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from jump_detection.algorithms.hybrid import HybridParameters, detect_hybrid_jumps
from jump_detection.annotations import load_annotations
from jump_detection.config import (
    DATASET_ROOT,
    HYBRID_TAKEOFF_THRESHOLD_DEFAULT,
    HYBRID_LANDING_DERIVATIVE_THRESHOLD_DEFAULT,
    HYBRID_IN_AIR_THRESHOLD_DEFAULT,
    SAMPLING_RATE,
)
from jump_detection.types import HybridNames
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


def plot_hybrid_pipeline(
    result,
    *,
    search_window: int = 70,
    show: bool = True,
    ground_truth_markers: list[int] | None = None,
) -> plt.Figure:
    """Plot the hybrid-based detection pipeline."""
    signals = result.signals
    params = result.metadata.get("parameters", {})
    raw_data = signals[HybridNames.RAW_DATA.value]
    pooled = signals[HybridNames.POOLED.value]
    derivative = signals[HybridNames.DERIVATIVE.value]
    takeoff_mask = signals[HybridNames.TAKEOFF_MASK.value]
    landing_mask = signals[HybridNames.LANDING_MASK.value]
    pair_indicator = signals[HybridNames.PAIR_INDICATOR.value]
    valid_pair_indicator = signals[HybridNames.VALID_PAIR_INDICATOR.value]
    in_air = signals[HybridNames.IN_AIR.value]
    
    fig, axes = plt.subplots(5, 1, figsize=(20, 12), sharex=True)
    fig.suptitle(
        f"Jump Detection Pipeline (Hybrid Algorithm) - {result.participant_name or 'Participant'}",
        fontsize=18,
    )
    
    # Plot 1: Raw sensor data
    axes[0].set_ylabel("Raw Sensors", fontsize=12)
    for channel in range(raw_data.shape[1]):
        axes[0].plot(raw_data[:, channel], linewidth=0.5, alpha=0.9)
    axes[0].set_title("Raw Sensor Data", fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Pooled signal with thresholds
    axes[1].set_ylabel("Signal Value", fontsize=12)
    axes[1].plot(pooled, linewidth=1.0, color=PRIMARY_BLUE, label="Pooled Data", alpha=0.7)
    
    takeoff_threshold = params.get("takeoff_threshold")
    if takeoff_threshold is not None:
        takeoff_threshold = float(takeoff_threshold)
        axes[1].axhline(
            y=takeoff_threshold,
            color=PRIMARY_RED,
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
            label=f"Takeoff Threshold ({takeoff_threshold:.3f})",
        )
    
    in_air_threshold = params.get("in_air_threshold")
    if in_air_threshold is not None:
        in_air_threshold = float(in_air_threshold)
        axes[1].axhline(
            y=in_air_threshold,
            color=PRIMARY_ORANGE,
            linestyle=":",
            alpha=0.6,
            linewidth=1.0,
            label=f"In-Air Threshold ({in_air_threshold:.3f})",
        )
    
    axes[1].set_title("Pooled Sensor Signal with Thresholds", fontsize=14)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Derivative with landing threshold
    axes[2].set_ylabel("Derivative", fontsize=12)
    axes[2].plot(derivative, linewidth=1.0, color=PRIMARY_ORANGE, label="Derivative", alpha=0.8)
    axes[2].axhline(y=0, color="grey", linestyle="-", linewidth=0.5, alpha=0.3)
    
    landing_threshold = params.get("landing_derivative_threshold")
    if landing_threshold is not None:
        landing_threshold = float(landing_threshold)
        axes[2].axhline(
            y=landing_threshold,
            color=PRIMARY_RED,
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
            label=f"Landing Threshold ({landing_threshold:.3f})",
        )
    
    axes[2].set_title("Derivative Signal with Landing Threshold", fontsize=14)
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Event masks
    axes[3].set_ylabel("Event Masks", fontsize=12)
    axes[3].plot(takeoff_mask, linewidth=1.5, color=PRIMARY_BLUE, label="Takeoff Events", alpha=0.8, marker='o', markersize=3)
    axes[3].plot(landing_mask, linewidth=1.5, color=PRIMARY_ORANGE, label="Landing Events", alpha=0.8, marker='s', markersize=3)
    axes[3].plot(pair_indicator, linewidth=1.0, color=PRIMARY_YELLOW, label="All Pairs", alpha=0.6, linestyle='--')
    axes[3].plot(valid_pair_indicator, linewidth=1.5, color=PRIMARY_RED, label="Valid Pairs", alpha=0.9, marker='*', markersize=4)
    axes[3].set_title("Takeoff and Landing Event Masks", fontsize=14)
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(-0.1, 1.1)
    
    # Plot 5: Detections
    axes[4].set_ylabel("Signal Value", fontsize=12)
    axes[4].set_xlabel("Frame Index", fontsize=12)
    axes[4].plot(pooled, linewidth=0.7, color=PRIMARY_BLUE, alpha=0.5, label="Pooled Signal")
    
    # Highlight detected jumps
    for jump in result.jumps:
        axes[4].axvspan(
            max(0, jump.start - search_window),
            min(len(pooled), jump.end + search_window),
            alpha=0.2,
            color=PRIMARY_YELLOW,
        )
        axes[4].axvline(jump.start, color=PRIMARY_BLUE, linestyle="--", linewidth=1.5, alpha=0.7)
        axes[4].axvline(jump.end, color=PRIMARY_BLUE, linestyle="--", linewidth=1.5, alpha=0.7)
        axes[4].axvline(jump.center, color=PRIMARY_RED, linestyle="-", linewidth=2, alpha=0.9)
    
    axes[4].set_title("Detected Jumps", fontsize=14)
    axes[4].legend(loc="upper right")
    axes[4].grid(True, alpha=0.3)
    
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
                        linestyle=":",
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
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def process_all_files(
    params: HybridParameters | None = None,
    search_window: int = 70,
    save_plots: bool = True,
) -> None:
    """Process all data files and generate detailed plots for hybrid algorithm."""
    if params is None:
        params = HybridParameters(
            takeoff_threshold=HYBRID_TAKEOFF_THRESHOLD_DEFAULT,
            landing_derivative_threshold=HYBRID_LANDING_DERIVATIVE_THRESHOLD_DEFAULT,
            in_air_threshold=HYBRID_IN_AIR_THRESHOLD_DEFAULT,
        )
    
    data_files = discover_all_data_files()
    print(f"Found {len(data_files)} data files to process")
    
    project_root = Path(__file__).parent.parent.parent.parent
    save_dir = project_root / "results" / "plots" / "pipeline" / "hybrid"
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
            result = detect_hybrid_jumps(
                file_path,
                participant_name,
                params=params,
                save_windows=False,
            )
            
            print(f"Hybrid algorithm parameters: "
                  f"takeoff_threshold={params.takeoff_threshold:.3f}, "
                  f"landing_derivative_threshold={params.landing_derivative_threshold:.3f}, "
                  f"in_air_threshold={params.in_air_threshold:.3f}")
            if ground_truth_markers:
                print(f"Ground truth markers: {len(ground_truth_markers)} markers loaded")
            print(f"Detected jumps: {len(result.jumps)}")
            
            # Generate pipeline plot
            if save_plots:
                fig = plot_hybrid_pipeline(
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
    params = HybridParameters(
        takeoff_threshold=HYBRID_TAKEOFF_THRESHOLD_DEFAULT,
        landing_derivative_threshold=HYBRID_LANDING_DERIVATIVE_THRESHOLD_DEFAULT,
        in_air_threshold=HYBRID_IN_AIR_THRESHOLD_DEFAULT,
    )
    
    process_all_files(params=params, save_plots=True)

