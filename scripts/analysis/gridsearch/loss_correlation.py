from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import sys
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from jump_detection.analysis.loss import compute_precision_loss
from jump_detection.annotations import load_annotations
from jump_detection.config import SAMPLING_RATE
from jump_detection.algorithms.correlation import CorrelationParameters, _run_correlation_pipeline
from jump_detection.types import DetectionResult, Jump
from jump_detection.utils import (
    GRADIENT_CMAP, PRIMARY_RED, concatenate_data_and_annotations,
    compute_precise_jumps_from_result, find_all_data_files, ProgressTracker, SEARCH_WINDOW
)

# Grid search parameter ranges
NEGATIVE_FRAMES_VALUES = np.arange(1, 21, 1)  # -1 width: 1 to 20
POSITIVE_FRAMES_VALUES = np.arange(1, 21, 1)  # +1 width: 1 to 20
CENTER_SPACING = 20  # Distance between centers of -1 and +1 sections
THRESHOLD_MAX = 400  # Maximum threshold value
THRESHOLD_MIN = 0  # Minimum threshold value
THRESHOLD_STEPS = 50  # Number of threshold steps to test

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_FILES = [str(f) for f in find_all_data_files(PROJECT_ROOT / "dataset")]
OUTPUT_DIR = PROJECT_ROOT / "results" / "plots" / "loss"


def compute_loss_with_annotations(detected_jumps: list, ground_truth_markers: list[int]) -> float:
    """Compute precision loss using ground truth annotations.
    
    Uses compute_precision_loss() which calculates:
    - False Positives: detected jumps that don't contain any ground truth marker
    - False Negatives: ground truth markers that don't fall within any detected jump's boundaries
    - Loss = FP + FN
    
    This is NOT just |10 - num_jumps|, but rather checks if each ground truth marker
    falls within the precise boundaries of detected jumps.
    """
    if not ground_truth_markers:
        raise ValueError("Ground truth markers are required but not provided")
    return float(compute_precision_loss(detected_jumps, ground_truth_markers)["loss"])


def calculate_zero_frames(negative_frames: int, positive_frames: int, center_spacing: int = CENTER_SPACING) -> int:
    """Calculate zero_frames so that centers of -1 and +1 sections are center_spacing apart.
    
    The center of the -1 section is at: (negative_frames - 1) / 2
    The center of the +1 section is at: negative_frames + zero_frames + (positive_frames - 1) / 2
    
    Distance between centers = [negative_frames + zero_frames + (positive_frames - 1) / 2] - [(negative_frames - 1) / 2]
    = negative_frames + zero_frames + (positive_frames - 1) / 2 - (negative_frames - 1) / 2
    = negative_frames + zero_frames + positive_frames/2 - 0.5 - negative_frames/2 + 0.5
    = (negative_frames + positive_frames) / 2 + zero_frames
    
    Setting this equal to center_spacing:
    zero_frames = center_spacing - (negative_frames + positive_frames) / 2
    
    Examples:
    - negative=1, positive=1: zero_frames = 20 - 1 = 19
      Centers: -1 at 0, +1 at 1+19+0 = 20, distance = 20 ✓
    - negative=20, positive=20: zero_frames = 20 - 20 = 0
      Centers: -1 at 9.5, +1 at 20+0+9.5 = 29.5, distance = 20 ✓
    - negative=10, positive=10: zero_frames = 20 - 10 = 10
      Centers: -1 at 4.5, +1 at 10+10+4.5 = 24.5, distance = 20 ✓
    """
    zero_frames = center_spacing - (negative_frames + positive_frames) / 2
    # Round to nearest integer and ensure non-negative
    zero_frames = max(0, int(round(zero_frames)))
    return zero_frames


def compute_loss_for_threshold_concatenated(
    concatenated_data: np.ndarray,
    concatenated_markers: list[int],
    negative_frames: int,
    positive_frames: int,
    threshold: float,
) -> float:
    """Compute total loss for concatenated data at a given threshold.
    
    This function:
    1. Runs the correlation pipeline to detect jumps
    2. Refines jump boundaries using precise detection (compute_precise_jumps_from_result)
    3. Computes precision loss (FP + FN) based on whether ground truth markers fall within precise jump boundaries
    
    The loss is NOT just |10 - num_jumps|, but rather counts:
    - False Positives: detected jumps that don't contain any ground truth marker
    - False Negatives: ground truth markers that don't fall within any detected jump's precise boundaries
    """
    zero_frames = calculate_zero_frames(negative_frames, positive_frames)
    params = CorrelationParameters(
        buffer_size=negative_frames + zero_frames + positive_frames,
        negative_frames=negative_frames,
        zero_frames=zero_frames,
        positive_frames=positive_frames,
        correlation_threshold=threshold,
    )
    signals, jumps, _ = _run_correlation_pipeline(concatenated_data, params)
    result = DetectionResult(
        participant_name=None, sampling_rate=SAMPLING_RATE, raw_data=concatenated_data,
        pooled_data=signals["pooled"], signals=signals, jumps=jumps, metadata={}
    )
    # Use precise jump boundaries (not raw detection boundaries)
    precise_jumps = compute_precise_jumps_from_result(concatenated_data, result, SEARCH_WINDOW)
    # Compute precision loss: FP + FN based on precise boundaries
    return compute_loss_with_annotations(precise_jumps, concatenated_markers)


def optimize_threshold_for_width_combination(
    concatenated_data: np.ndarray,
    concatenated_markers: list[int],
    negative_frames: int,
    positive_frames: int,
) -> tuple[float, float]:
    """Find threshold that minimizes total loss for given width combination.
    
    This optimizes the threshold GLOBALLY across ALL participants (using concatenated data).
    For each threshold value, it:
    1. Runs detection on concatenated data (all participants)
    2. Computes precise jump boundaries
    3. Calculates precision loss (FP + FN) across all participants
    4. Returns the threshold that minimizes this global loss
    
    Returns:
        Tuple of (best_threshold, best_loss) for this width combination
    """
    threshold_values = np.linspace(THRESHOLD_MAX, THRESHOLD_MIN, THRESHOLD_STEPS)
    best_loss, best_threshold = float('inf'), THRESHOLD_MAX
    
    for threshold in threshold_values:
        # Compute loss across ALL participants (concatenated data)
        total_loss = compute_loss_for_threshold_concatenated(
            concatenated_data, concatenated_markers, negative_frames, positive_frames, threshold
        )
        if total_loss < best_loss:
            best_loss, best_threshold = total_loss, threshold
    return best_threshold, best_loss


def optimize_threshold_for_width_combination_wrapper(
    args: tuple[np.ndarray, list[int], int, int],
) -> tuple[int, int, float, float]:
    """Wrapper function for parallel processing of width combinations.
    
    Args:
        args: Tuple of (concatenated_data, concatenated_markers, negative_frames, positive_frames)
        
    Returns:
        Tuple of (negative_frames, positive_frames, best_threshold, best_loss)
    """
    concatenated_data, concatenated_markers, negative_frames, positive_frames = args
    best_threshold, best_loss = optimize_threshold_for_width_combination(
        concatenated_data,
        concatenated_markers,
        negative_frames,
        positive_frames,
    )
    return negative_frames, positive_frames, best_threshold, best_loss


def grid_search_width_combinations(
    concatenated_data: np.ndarray,
    concatenated_markers: list[int],
    show_progress: bool = False,
    max_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep through width combinations, optimizing threshold for each.
    
    For each (-1 width, +1 width) combination:
    1. Tests all threshold values (THRESHOLD_MIN to THRESHOLD_MAX)
    2. For each threshold, computes precision loss across ALL participants (concatenated data)
    3. Finds the threshold that minimizes this global loss
    4. Stores the optimal threshold and corresponding loss
    
    Returns:
        Tuple of (best_loss_grid, best_threshold_grid) where:
        - best_loss_grid[i, j] = optimal loss for (NEGATIVE_FRAMES_VALUES[i], POSITIVE_FRAMES_VALUES[j])
        - best_threshold_grid[i, j] = optimal threshold for that combination
    """
    total = len(NEGATIVE_FRAMES_VALUES) * len(POSITIVE_FRAMES_VALUES)
    progress = ProgressTracker(total) if show_progress else None
    
    best_loss_grid = np.full((len(NEGATIVE_FRAMES_VALUES), len(POSITIVE_FRAMES_VALUES)), float('inf'))
    best_threshold_grid = np.zeros((len(NEGATIVE_FRAMES_VALUES), len(POSITIVE_FRAMES_VALUES)))
    
    width_combinations = [
        (concatenated_data, concatenated_markers, int(neg), int(pos))
        for neg in NEGATIVE_FRAMES_VALUES for pos in POSITIVE_FRAMES_VALUES
    ]
    index_map = {(int(neg), int(pos)): (i, j) for i, neg in enumerate(NEGATIVE_FRAMES_VALUES)
                 for j, pos in enumerate(POSITIVE_FRAMES_VALUES)}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_combo = {executor.submit(optimize_threshold_for_width_combination_wrapper, combo): combo
                           for combo in width_combinations}
        for future in as_completed(future_to_combo):
            neg, pos, best_threshold, best_loss = future.result()
            i, j = index_map[(neg, pos)]
            best_loss_grid[i, j] = best_loss
            best_threshold_grid[i, j] = best_threshold
            if progress:
                progress.update()
    
    if progress:
        progress.finish()
    return best_loss_grid, best_threshold_grid


def plot_2d_loss_heatmap(
    loss_grid: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> tuple[int, int, float]:
    """Create a 2D heatmap plot of the loss values.
    
    Returns:
        Tuple of (best_negative, best_positive, best_loss)
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create meshgrid for 2D plotting
    X, Y = np.meshgrid(x_values, y_values)
    
    # Create filled contour/heatmap
    im = ax.contourf(X, Y, loss_grid, levels=50, cmap=GRADIENT_CMAP, alpha=0.8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Best Loss (False Positives + False Negatives)', rotation=270, labelpad=20)
    
    # Add contour lines
    contours = ax.contour(X, Y, loss_grid, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')
    
    # Find and mark the global minimum
    min_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
    best_negative = int(x_values[min_idx[1]])
    best_positive = int(y_values[min_idx[0]])
    best_loss = loss_grid[min_idx]
    
    # Mark global minimum with a star
    ax.plot(best_negative, best_positive, 'r*', markersize=25, markeredgewidth=2,
            markeredgecolor=PRIMARY_RED, label=f'Global minimum: -1={best_negative}, +1={best_positive}, Loss={best_loss:.2f}')
    
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap plot to {output_path}")
    plt.close(fig)
    
    return best_negative, best_positive, best_loss


def plot_3d_loss_surface(
    loss_grid: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    show_interactive: bool = False,
) -> None:
    """Create a 3D surface plot of the loss values."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for 3D plotting
    X, Y = np.meshgrid(x_values, y_values)

    # Plot surface
    surf = ax.plot_surface(
        X, Y, loss_grid,
        cmap=GRADIENT_CMAP,
        alpha=0.9,
        linewidth=0,
        antialiased=True,
    )

    # Find and mark the global minimum
    min_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
    min_x = x_values[min_idx[1]]
    min_y = y_values[min_idx[0]]
    min_loss = loss_grid[min_idx]
    ax.scatter([min_x], [min_y], [min_loss], color=PRIMARY_RED, s=200, marker='*',
               edgecolors='darkred', linewidths=2, label=f'Global min: -1={min_x:.0f}, +1={min_y:.0f}, Loss={min_loss:.2f}')

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_zlabel('Best Loss (False Positives + False Negatives)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Loss')

    # Add legend
    ax.legend(loc='upper left', fontsize=9)

    # Set viewing angle
    ax.view_init(elev=30, azim=45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D plot to {output_path}")
    
    if show_interactive:
        plt.show()
    else:
        plt.close(fig)


def plot_threshold_heatmap(
    threshold_grid: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    """Create a 2D heatmap plot showing optimal thresholds for each width combination."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create meshgrid for 2D plotting
    X, Y = np.meshgrid(x_values, y_values)
    
    # Create filled contour/heatmap
    im = ax.contourf(X, Y, threshold_grid, levels=50, cmap=GRADIENT_CMAP, alpha=0.8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Optimal Threshold', rotation=270, labelpad=20)
    
    # Add contour lines
    contours = ax.contour(X, Y, threshold_grid, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved threshold heatmap to {output_path}")
    plt.close(fig)


def main() -> None:
    """Main function to run loss calculation.
    
    Note: Threshold optimization requires testing all participants for each threshold,
    so we cannot parallelize by participant. The threshold is optimized globally
    across all participants for each width combination.
    """
    print("LOSS CALCULATION AND VISUALIZATION (CORRELATION ALGORITHM)")
    print("="*70)
    print("Using ground truth annotations for loss calculation")
    print("Loss = False Positives + False Negatives (based on PRECISE jump boundaries)")
    print("  - NOT just |10 - num_jumps|")
    print("  - Uses compute_precision_loss() which checks if ground truth markers")
    print("    fall within precise jump boundaries (refined via compute_precise_jumps_from_result)")
    print("Threshold is optimized globally across ALL participants (not per participant)")
    print(f"Center spacing between -1 and +1 sections: {CENTER_SPACING} frames")
    print(f"Zero frames calculated dynamically: max(0, {CENTER_SPACING} - (neg_width + pos_width) / 2)")
    print(f"Threshold range: {THRESHOLD_MAX} to {THRESHOLD_MIN} ({THRESHOLD_STEPS} steps)")
    print(f"Started at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("="*70)

    print(f"\nLoading data files and annotations for {len(DATA_FILES)} participants...")
    data_files_with_annotations = []
    for data_file in DATA_FILES:
        file_path = Path(data_file)
        annotations = load_annotations(file_path)
        if annotations is None:
            raise FileNotFoundError(f"Annotations file not found for {file_path}")
        if not annotations.markers:
            raise ValueError(f"Annotations file for {file_path} contains no markers")
        data_files_with_annotations.append((file_path, annotations.markers))
        print(f"  ✓ {file_path.stem}: {len(annotations.markers)} markers")
    
    print(f"\nSuccessfully loaded {len(data_files_with_annotations)} participants")
    print("\nConcatenating all data files and annotations...")
    concatenated_data, concatenated_markers, file_boundaries = concatenate_data_and_annotations(
        data_files_with_annotations
    )
    print(f"  ✓ Concatenated data shape: {concatenated_data.shape}")
    print(f"  ✓ Total markers: {len(concatenated_markers)}")
    print(f"  ✓ File boundaries: {file_boundaries}")
    
    # Grid search: optimize threshold for each (-1, +1) width combination
    # Threshold is optimized across ALL participants (not per participant)
    # Parallelized across width combinations
    print(f"\nGrid searching {len(NEGATIVE_FRAMES_VALUES)} × {len(POSITIVE_FRAMES_VALUES)} = {len(NEGATIVE_FRAMES_VALUES) * len(POSITIVE_FRAMES_VALUES)} width combinations...")
    print("For each combination, testing all thresholds on concatenated data...")
    print("Parallelizing across width combinations...")
    
    best_loss_grid, best_threshold_grid = grid_search_width_combinations(
        concatenated_data,
        concatenated_markers,
        show_progress=True,
        max_workers=None,  # Use all available CPU cores
    )
    
    # The loss grid already contains total loss across all participants
    combined_best_loss_grid = best_loss_grid
    combined_best_threshold_grid = best_threshold_grid
    
    # Transpose for plotting (so negative_frames is x-axis, positive_frames is y-axis)
    combined_best_loss_grid_transposed = combined_best_loss_grid.T
    combined_best_threshold_grid_transposed = combined_best_threshold_grid.T
    
    # Find global minimum indices before plotting
    min_idx = np.unravel_index(np.argmin(combined_best_loss_grid_transposed), combined_best_loss_grid_transposed.shape)
    best_neg_idx = min_idx[1]  # Index for negative_frames (x-axis)
    best_pos_idx = min_idx[0]  # Index for positive_frames (y-axis)
    best_neg_val = int(NEGATIVE_FRAMES_VALUES[best_neg_idx])
    best_pos_val = int(POSITIVE_FRAMES_VALUES[best_pos_idx])
    best_loss_val = combined_best_loss_grid_transposed[min_idx]
    optimal_threshold = combined_best_threshold_grid_transposed[best_pos_idx, best_neg_idx]
    
    # Plot 2D heatmap of best losses
    plot_2d_loss_heatmap(
        combined_best_loss_grid_transposed,
        NEGATIVE_FRAMES_VALUES,
        POSITIVE_FRAMES_VALUES,
        "Negative Frames Width (-1 section)",
        "Positive Frames Width (+1 section)",
        "Combined Best Loss - Correlation Algorithm (Optimized Threshold for Each Width Combination)",
        OUTPUT_DIR / "combined_loss_correlation_optimized_heatmap.png",
    )
    
    # Plot 3D surface
    plot_3d_loss_surface(
        combined_best_loss_grid_transposed,
        NEGATIVE_FRAMES_VALUES,
        POSITIVE_FRAMES_VALUES,
        "Negative Frames Width (-1 section)",
        "Positive Frames Width (+1 section)",
        "Combined Best Loss - Correlation Algorithm (Optimized Threshold for Each Width Combination)",
        OUTPUT_DIR / "combined_loss_correlation_optimized_3d.png",
        show_interactive=False,
    )
    
    # Plot threshold heatmap
    plot_threshold_heatmap(
        combined_best_threshold_grid_transposed,
        NEGATIVE_FRAMES_VALUES,
        POSITIVE_FRAMES_VALUES,
        "Negative Frames Width (-1 section)",
        "Positive Frames Width (+1 section)",
        "Optimal Threshold Values for Each Width Combination",
        OUTPUT_DIR / "combined_threshold_correlation_optimized_heatmap.png",
    )
    
    # Print global minimum
    print("\n" + "="*70)
    print("GLOBAL MINIMUM:")
    print("="*70)
    best_zero_frames = calculate_zero_frames(best_neg_val, best_pos_val)
    print(f"Best -1 width: {best_neg_val}")
    print(f"Best +1 width: {best_pos_val}")
    print(f"Calculated zero width: {best_zero_frames}")
    print(f"Best loss: {best_loss_val:.2f}")
    print(f"Optimal threshold: {optimal_threshold:.2f}")
    print(f"Total buffer size: {best_neg_val + best_zero_frames + best_pos_val}")
    print("="*70)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS:")
    print("="*70)
    print(f"Minimum loss across all combinations: {np.min(combined_best_loss_grid):.2f}")
    print(f"Maximum loss across all combinations: {np.max(combined_best_loss_grid):.2f}")
    print(f"Mean loss across all combinations: {np.mean(combined_best_loss_grid):.2f}")
    print(f"Number of combinations with loss <= 1.0: {np.sum(combined_best_loss_grid <= 1.0)}")
    print(f"Number of combinations with loss <= 2.0: {np.sum(combined_best_loss_grid <= 2.0)}")
    print(f"Number of combinations with loss <= 5.0: {np.sum(combined_best_loss_grid <= 5.0)}")
    print("="*70)

    print(f"\nCompleted at: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
