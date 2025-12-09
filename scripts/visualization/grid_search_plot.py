"""Grid search visualization for individual participants.

This script creates 2D heatmaps showing loss function values for each participant,
highlighting parameter combinations that result in perfect detection (loss = 0) in red.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jump_detection.algorithms.derivative import (
    DerivativeParameters,
    _run_derivative_pipeline,
)
from jump_detection.algorithms.threshold import (
    ThresholdParameters,
    _run_threshold_pipeline,
)
from jump_detection.analysis.loss import compute_precision_loss
from jump_detection.analysis.precise import calculate_precise_jump_boundaries
from jump_detection.annotations import load_annotations
from jump_detection.config import DATASET_ROOT, SAMPLING_RATE
from jump_detection.data import load_dataset
from jump_detection.types import (
    DetectionResult,
    DerivativeNames,
    Jump,
    ThresholdNames,
)

# Parameter ranges for grid search
# Note: Using pooled sensor value (sum of 48 sensors), so thresholds are multiplied by 48
# Reduced resolution for faster computation and clearer plots
GRID_SIZE = 25  # Reduced from 40 to 25 for less fine discretization

# Threshold: 48 * 1.7 to 48 * 2.5 = 81.6 to 120
THRESHOLD_VALUES = np.linspace(48 * 1.7, 48 * 2.5, GRID_SIZE)
# Derivative magnitude: 0.3 * 48 to 1.7 * 48 = 14.4 to 81.6
DERIVATIVE_MAGNITUDE_VALUES = np.linspace(0.3 * 48, 1.7 * 48, GRID_SIZE)

# Derivative algorithm ranges (for pooled sensor values, multiplied by 48)
# Lower threshold: -100 to 0 (max is 0)
DERIVATIVE_LOWER_VALUES = np.linspace(-100, 0, GRID_SIZE)
# Upper threshold: 0 to 100 (min is 0)
DERIVATIVE_UPPER_VALUES = np.linspace(0, 100, GRID_SIZE)

SEARCH_WINDOW = 70  # For precise jump detection
IN_AIR_THRESHOLD = 250.0  # Fixed for derivative algorithm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "plots" / "grid_search"


def find_all_data_files(dataset_root: Path = DATASET_ROOT) -> list[Path]:
    """Find all data files that have corresponding annotation files.
    
    Looks in Test0, Test2, and Test3 (video) folders specifically.
    Test0 has files with or without .txt extension.
    Test2 and Test3 (video) have files with .txt extension.
    """
    dataset_root = Path(dataset_root)
    data_files = []
    
    # Check Test0 folder
    test0_dir = dataset_root / "Test0"
    if test0_dir.exists():
        # Test0 files can have or not have .txt extension
        for file_path in test0_dir.iterdir():
            if file_path.is_file():
                # Skip annotation files and other non-data files
                if file_path.name.endswith("_annotations.json") or file_path.name.endswith(".json"):
                    continue
                
                # Check if corresponding annotation file exists
                # For files with .txt extension, use stem (filename without extension)
                # For files without extension, use full name
                if file_path.suffix == ".txt":
                    annotation_path = test0_dir / f"{file_path.stem}_annotations.json"
                else:
                    annotation_path = test0_dir / f"{file_path.name}_annotations.json"
                
                if annotation_path.exists():
                    data_files.append(file_path)
    
    # Check Test2 folder
    test2_dir = dataset_root / "Test2"
    if test2_dir.exists():
        # Test2 files have .txt extension
        for txt_file in test2_dir.glob("*.txt"):
            # Check if corresponding annotation file exists
            annotation_path = test2_dir / f"{txt_file.stem}_annotations.json"
            if annotation_path.exists():
                data_files.append(txt_file)
    
    # Check Test3 (video) folder
    test3_dir = dataset_root / "Test3 (video)"
    if test3_dir.exists():
        # Test3 (video) files have .txt extension
        for txt_file in test3_dir.glob("*.txt"):
            # Check if corresponding annotation file exists
            annotation_path = test3_dir / f"{txt_file.stem}_annotations.json"
            if annotation_path.exists():
                data_files.append(txt_file)
    
    return sorted(data_files)


def compute_precise_jumps(
    data: np.ndarray,
    result: DetectionResult,
) -> list[Jump]:
    """Calculate precise jump boundaries from detection result.
    
    Args:
        data: Raw sensor data
        result: DetectionResult from algorithm pipeline
        
    Returns:
        List of Jump objects with precise boundaries
    """
    signal = result.pooled_data if result.pooled_data is not None else data.sum(axis=1)
    precise_jumps = []
    
    for jump in result.jumps:
        # Calculate precise boundaries
        precise_data = calculate_precise_jump_boundaries(
            signal, jump.center, SEARCH_WINDOW
        )
        # Only include if precise boundaries are valid
        if precise_data["precise_start"] < precise_data["precise_end"]:
            precise_jumps.append(
                Jump(
                    start=precise_data["precise_start"],
                    end=precise_data["precise_end"],
                    center=precise_data["precise_center"],
                    duration=precise_data["precise_duration"],
                )
            )
    
    return precise_jumps


def compute_loss_for_threshold_combination(
    data: np.ndarray,
    ground_truth_markers: list[int],
    threshold: float,
    derivative_threshold: float,
) -> tuple[int, int]:
    """Compute loss and jump count for threshold algorithm with given parameters.
    
    Args:
        data: Raw sensor data
        ground_truth_markers: List of frame indices that should be in correct jumps
        threshold: Threshold parameter
        derivative_threshold: Derivative threshold parameter
        
    Returns:
        Tuple of (loss, num_jumps) where:
        - loss: Loss value (false_positives + false_negatives)
        - num_jumps: Number of jumps detected
    """
    params = ThresholdParameters(threshold=threshold, derivative_threshold=derivative_threshold)
    
    # Run threshold pipeline
    signals, jumps, metadata = _run_threshold_pipeline(data, params)
    
    # Create DetectionResult
    result = DetectionResult(
        participant_name=None,
        sampling_rate=SAMPLING_RATE,
        raw_data=data,
        pooled_data=signals[ThresholdNames.AVERAGE.value],
        signals=signals,
        jumps=jumps,
        metadata=metadata,
    )
    
    # Calculate precise jumps
    precise_jumps = compute_precise_jumps(data, result)
    
    # Compute loss using ground truth annotations
    metrics = compute_precision_loss(precise_jumps, ground_truth_markers)
    return metrics["loss"], len(precise_jumps)  # Returns (loss, num_jumps)


def compute_loss_for_derivative_combination(
    data: np.ndarray,
    ground_truth_markers: list[int],
    upper_threshold: float,
    lower_threshold: float,
) -> tuple[int, int]:
    """Compute loss and jump count for derivative algorithm with given parameters.
    
    Args:
        data: Raw sensor data
        ground_truth_markers: List of frame indices that should be in correct jumps
        upper_threshold: Upper derivative threshold parameter
        lower_threshold: Lower derivative threshold parameter
        
    Returns:
        Tuple of (loss, num_jumps) where:
        - loss: Loss value (false_positives + false_negatives)
        - num_jumps: Number of jumps detected
    """
    params = DerivativeParameters(
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        in_air_threshold=IN_AIR_THRESHOLD,
    )
    
    # Run derivative pipeline
    signals, jumps, metadata = _run_derivative_pipeline(data, params)
    
    # Create DetectionResult
    result = DetectionResult(
        participant_name=None,
        sampling_rate=SAMPLING_RATE,
        raw_data=data,
        pooled_data=signals[DerivativeNames.POOLED.value],
        signals=signals,
        jumps=jumps,
        metadata=metadata,
    )
    
    # Calculate precise jumps
    precise_jumps = compute_precise_jumps(data, result)
    
    # Compute loss using ground truth annotations
    metrics = compute_precision_loss(precise_jumps, ground_truth_markers)
    return metrics["loss"], len(precise_jumps)  # Returns (loss, num_jumps)


def grid_search_threshold_for_participant(
    data: np.ndarray,
    ground_truth_markers: list[int],
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run grid search for threshold algorithm on a single participant.
    
    Args:
        data: Raw sensor data for the participant
        ground_truth_markers: Ground truth markers for the participant
        show_progress: Whether to show progress
        
    Returns:
        Tuple of (loss_grid, jump_count_grid, threshold_values, derivative_magnitude_values)
        where grids are 2D arrays [threshold, derivative_magnitude] and values are the
        actual parameter arrays used (may be adjusted based on signal statistics)
    """
    # Calculate signal statistics to adapt threshold range
    pooled = data.sum(axis=1)
    signal_min = float(pooled.min())
    signal_max = float(pooled.max())
    signal_mean = float(pooled.mean())
    
    # Adapt threshold range based on signal statistics
    # Threshold should be between signal_min and signal_max
    # Use a range that spans from slightly below min to well above mean
    threshold_min = max(signal_min * 0.8, signal_min - 50)  # Start below min, but not too low
    threshold_max = min(signal_max * 0.9, signal_mean * 2.5)  # Go up to mean*2.5 or 90% of max
    # Ensure we have a reasonable range
    if threshold_max <= threshold_min:
        threshold_max = threshold_min + 100
    
    # Generate adaptive threshold values
    threshold_values = np.linspace(threshold_min, threshold_max, GRID_SIZE)
    
    total = len(threshold_values) * len(DERIVATIVE_MAGNITUDE_VALUES)
    loss_grid = np.zeros((len(threshold_values), len(DERIVATIVE_MAGNITUDE_VALUES)), dtype=int)
    jump_count_grid = np.zeros((len(threshold_values), len(DERIVATIVE_MAGNITUDE_VALUES)), dtype=int)
    
    if show_progress:
        print(f"  Signal range: min={signal_min:.1f}, max={signal_max:.1f}, mean={signal_mean:.1f}")
        print(f"  Adaptive threshold range: [{threshold_min:.1f}, {threshold_max:.1f}]")
        print(f"  Grid searching {total} parameter combinations...", end="", flush=True)
    
    idx = 0
    for i, threshold in enumerate(threshold_values):
        for j, derivative_threshold in enumerate(DERIVATIVE_MAGNITUDE_VALUES):
            loss, num_jumps = compute_loss_for_threshold_combination(
                data, ground_truth_markers, float(threshold), float(derivative_threshold)
            )
            loss_grid[i, j] = loss
            jump_count_grid[i, j] = num_jumps
            idx += 1
            
            if show_progress and idx % 100 == 0:
                print(".", end="", flush=True)
    
    if show_progress:
        print(" done!")
    
    return loss_grid, jump_count_grid, threshold_values, DERIVATIVE_MAGNITUDE_VALUES


def grid_search_derivative_for_participant(
    data: np.ndarray,
    ground_truth_markers: list[int],
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run grid search for derivative algorithm on a single participant.
    
    Args:
        data: Raw sensor data for the participant
        ground_truth_markers: Ground truth markers for the participant
        show_progress: Whether to show progress
        
    Returns:
        Tuple of (loss_grid, jump_count_grid) where both are 2D arrays [upper_threshold, lower_threshold]
    """
    total = len(DERIVATIVE_UPPER_VALUES) * len(DERIVATIVE_LOWER_VALUES)
    loss_grid = np.zeros((len(DERIVATIVE_UPPER_VALUES), len(DERIVATIVE_LOWER_VALUES)), dtype=int)
    jump_count_grid = np.zeros((len(DERIVATIVE_UPPER_VALUES), len(DERIVATIVE_LOWER_VALUES)), dtype=int)
    
    if show_progress:
        print(f"  Grid searching {total} parameter combinations...", end="", flush=True)
    
    idx = 0
    for i, upper_threshold in enumerate(DERIVATIVE_UPPER_VALUES):
        for j, lower_threshold in enumerate(DERIVATIVE_LOWER_VALUES):
            loss, num_jumps = compute_loss_for_derivative_combination(
                data, ground_truth_markers, float(upper_threshold), float(lower_threshold)
            )
            loss_grid[i, j] = loss
            jump_count_grid[i, j] = num_jumps
            idx += 1
            
            if show_progress and idx % 100 == 0:
                print(".", end="", flush=True)
    
    if show_progress:
        print(" done!")
    
    return loss_grid, jump_count_grid


def plot_grid_search_heatmap(
    loss_grid: np.ndarray,
    jump_count_grid: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    participant_name: str,
    algorithm_name: str,
    output_path: Path,
) -> None:
    """Create a 2D heatmap of loss values with perfect cells highlighted in red and jump counts displayed.
    
    Args:
        loss_grid: 2D array of loss values
        jump_count_grid: 2D array of jump counts for each cell
        x_values: Values for x-axis (first parameter)
        y_values: Values for y-axis (second parameter)
        x_label: Label for x-axis
        y_label: Label for y-axis
        participant_name: Name of the participant
        algorithm_name: Name of the algorithm
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_values, y_values)
    
    # Transpose grids to match (x, y) convention
    loss_grid_T = loss_grid.T
    jump_count_grid_T = jump_count_grid.T
    
    # Create custom colormap: purple → blue → orange for low loss values
    colors = ['purple', 'blue', 'orange']
    n_bins = 100
    custom_cmap = LinearSegmentedColormap.from_list('purple_blue_orange', colors, N=n_bins)
    
    # Create base heatmap with custom colormap
    im = ax.contourf(X, Y, loss_grid_T, levels=50, cmap=custom_cmap, alpha=0.8)
    
    # Highlight perfect cells (loss = 0) in red
    perfect_mask = loss_grid_T == 0
    if perfect_mask.any():
        # Create a red overlay for perfect cells
        perfect_overlay = np.ma.masked_where(~perfect_mask, np.ones_like(loss_grid_T))
        ax.contourf(
            X, Y, perfect_overlay,
            levels=[0.5, 1.5],
            colors=['red'],
            alpha=0.9,
        )
        num_perfect = perfect_mask.sum()
        print(f"    Found {num_perfect} perfect parameter combinations (loss = 0)")
    else:
        print(f"    No perfect parameter combinations found (loss = 0)")
    
    # Add jump count text annotations in each cell
    # Only show text for cells where it's readable (not too many cells)
    if len(x_values) <= 30 and len(y_values) <= 30:
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                jump_count = jump_count_grid_T[i, j]
                loss = loss_grid_T[i, j]
                # Show jump count in white text for visibility
                ax.text(
                    x_values[j], y_values[i],
                    f'{jump_count}',
                    ha='center', va='center',
                    fontsize=8,
                    color='white' if loss > 5 else 'black',
                    fontweight='bold',
                )
    
    # Add contour lines for better readability
    contours = ax.contour(X, Y, loss_grid_T, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Loss (False Positives + False Negatives)', rotation=270, labelpad=20)
    
    # Set labels and title
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(
        f"{algorithm_name} - {participant_name}\n"
        f"Grid Search Loss Function (Red = Perfect Detection, Numbers = Jump Count)",
        fontsize=14,
        fontweight='bold',
    )
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"    Saved plot to: {output_path}")


def process_participant_threshold(
    data_file: Path,
    output_dir: Path,
) -> None:
    """Process a single participant for threshold algorithm grid search."""
    # Load data and annotations
    data = load_dataset(data_file)
    annotations = load_annotations(data_file)
    
    if annotations is None or not annotations.markers:
        print(f"  Skipping {data_file.name}: No annotations found")
        return
    
    ground_truth_markers = annotations.markers
    participant_name = data_file.stem.replace("_annotations", "").replace("10CMJ", "").strip()
    if not participant_name:
        participant_name = data_file.parent.name
    
    print(f"\nProcessing {participant_name} (Threshold Algorithm):")
    print(f"  Data shape: {data.shape}, Markers: {len(ground_truth_markers)}")
    
    # Run grid search (returns adaptive threshold values)
    loss_grid, jump_count_grid, threshold_values, derivative_magnitude_values = grid_search_threshold_for_participant(
        data, ground_truth_markers, show_progress=True
    )
    
    # Create plot
    output_path = output_dir / "threshold" / f"{participant_name}_threshold.png"
    plot_grid_search_heatmap(
        loss_grid,
        jump_count_grid,
        threshold_values,
        derivative_magnitude_values,
        "Threshold",
        "Derivative Magnitude",
        participant_name,
        "Threshold Algorithm",
        output_path,
    )


def process_participant_derivative(
    data_file: Path,
    output_dir: Path,
) -> None:
    """Process a single participant for derivative algorithm grid search."""
    # Load data and annotations
    data = load_dataset(data_file)
    annotations = load_annotations(data_file)
    
    if annotations is None or not annotations.markers:
        print(f"  Skipping {data_file.name}: No annotations found")
        return
    
    ground_truth_markers = annotations.markers
    participant_name = data_file.stem.replace("_annotations", "").replace("10CMJ", "").strip()
    if not participant_name:
        participant_name = data_file.parent.name
    
    print(f"\nProcessing {participant_name} (Derivative Algorithm):")
    print(f"  Data shape: {data.shape}, Markers: {len(ground_truth_markers)}")
    
    # Run grid search
    loss_grid, jump_count_grid = grid_search_derivative_for_participant(
        data, ground_truth_markers, show_progress=True
    )
    
    # Create plot
    output_path = output_dir / "derivative" / f"{participant_name}_derivative.png"
    plot_grid_search_heatmap(
        loss_grid,
        jump_count_grid,
        DERIVATIVE_UPPER_VALUES,
        DERIVATIVE_LOWER_VALUES,
        "Upper Threshold",
        "Lower Threshold",
        participant_name,
        "Derivative Algorithm",
        output_path,
    )


def main():
    """Main function to generate grid search plots for all participants."""
    print("=" * 80)
    print("Grid Search Visualization - Individual Participants")
    print("=" * 80)
    
    # Find all data files with annotations
    print("\nFinding data files with annotations...")
    data_files = find_all_data_files()
    print(f"Found {len(data_files)} data files with annotations")
    
    if not data_files:
        print("No data files with annotations found. Exiting.")
        return
    
    # Create output directories
    threshold_dir = OUTPUT_DIR / "threshold"
    derivative_dir = OUTPUT_DIR / "derivative"
    threshold_dir.mkdir(parents=True, exist_ok=True)
    derivative_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each participant for both algorithms
    print(f"\nProcessing {len(data_files)} participants...")
    print("This may take a while as we're running grid search for each participant...")
    
    for data_file in data_files:
        try:
            # Process threshold algorithm
            process_participant_threshold(data_file, OUTPUT_DIR)
            
            # Process derivative algorithm
            process_participant_derivative(data_file, OUTPUT_DIR)
            
        except Exception as e:
            print(f"  Error processing {data_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'=' * 80}")
    print("Grid search plots completed!")
    print(f"Threshold plots saved to: {threshold_dir}")
    print(f"Derivative plots saved to: {derivative_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

