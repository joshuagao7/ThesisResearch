"""Unified loss landscape comparison for all jump detection algorithms.

This script computes and plots loss functions for threshold, derivative, and correlation algorithms.
Can process all three automatically or prompt for selection. Creates individual and combined plots
with distinct color gradients, side views, and minimum loss labeling.

NOTE: This script performs grid searches to find optimal parameter values. The optimized values
found here should match the defaults in jump_detection/config.py. If they don't match, update
the config.py file with the optimized values to ensure consistency across:
- summary_plot.py (threshold algorithm)
- summary_plot_2.py (derivative algorithm)  
- summary_plot_3.py (correlation algorithm)
- detailedPlot.py (all algorithms)
"""

from pathlib import Path
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

import sys
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from jump_detection.utils import find_all_data_files, concatenate_data_and_annotations
from jump_detection.annotations import load_annotations
from jump_detection.config import (
    THRESHOLD_DEFAULT,
    DERIVATIVE_THRESHOLD_DEFAULT,
    DERIVATIVE_UPPER_DEFAULT,
    DERIVATIVE_LOWER_DEFAULT,
    IN_AIR_THRESHOLD_DEFAULT,
)

# Color gradients for each algorithm
RED_GRADIENT = LinearSegmentedColormap.from_list('red_gradient', ['#8B0000', '#FF0000', '#FF6B6B'], N=100)
BLUE_GRADIENT = LinearSegmentedColormap.from_list('blue_gradient', ['#00008B', '#0066CC', '#4A90E2'], N=100)
GREEN_GRADIENT = LinearSegmentedColormap.from_list('green_gradient', ['#006400', '#00AA00', '#90EE90'], N=100)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

from scripts.analysis.gridsearch.loss_threshold import (
    grid_search_threshold_algorithm,
    THRESHOLD_VALUES,
    DERIVATIVE_MAGNITUDE_VALUES,
    plot_3d_loss as plot_3d_threshold,
)
from scripts.analysis.gridsearch.loss_derivative import (
    grid_search_derivative_algorithm,
    DERIVATIVE_UPPER_VALUES,
    DERIVATIVE_LOWER_VALUES,
    DERIVATIVE_LOWER_VALUES_REVERSED,
    plot_3d_loss as plot_3d_derivative,
    plot_bottom_view,
)
from scripts.analysis.gridsearch.loss_correlation import (
    grid_search_width_combinations,
    NEGATIVE_FRAMES_VALUES,
    POSITIVE_FRAMES_VALUES,
    plot_2d_loss_heatmap,
    plot_3d_loss_surface,
)

# Get project root for path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "plots" / "loss"


def normalize_parameter_space(values: np.ndarray, new_range: tuple[float, float] = (0, 1)) -> np.ndarray:
    """Normalize parameter values to a new range (default 0-1)."""
    min_val, max_val = new_range
    if values.max() == values.min():
        return np.full_like(values, (min_val + max_val) / 2)
    normalized = (values - values.min()) / (values.max() - values.min())
    return normalized * (max_val - min_val) + min_val


def prompt_algorithm_selection() -> list[str]:
    """Prompt user to select which algorithms to plot."""
    print("\n" + "="*80)
    print("LOSS FUNCTION PLOTTING - Algorithm Selection")
    print("="*80)
    print("\nSelect which algorithm(s) to generate loss plots for:")
    print("  1. Threshold Algorithm")
    print("  2. Derivative Algorithm")
    print("  3. Correlation Algorithm")
    print("  4. All Three Algorithms (default)")
    print()
    
    try:
        choice = input("Enter your selection (1-4, or comma-separated like '1,2', or press Enter for all): ").strip()
        if not choice or choice == "4":
            return ["threshold", "derivative", "correlation"]
        
        selections = [s.strip() for s in choice.split(",")]
        algorithms = []
        for sel in selections:
            if sel == "1":
                algorithms.append("threshold")
            elif sel == "2":
                algorithms.append("derivative")
            elif sel == "3":
                algorithms.append("correlation")
            else:
                raise ValueError(f"Invalid selection: {sel}")
        
        if not algorithms:
            return ["threshold", "derivative", "correlation"]
        
        seen = set()
        return [alg for alg in algorithms if alg not in seen and not seen.add(alg)]
    except (ValueError, KeyboardInterrupt):
        print("\nUsing default: All three algorithms")
        return ["threshold", "derivative", "correlation"]


def plot_combined_loss_landscapes(
    threshold_loss_grid: Optional[np.ndarray],
    derivative_loss_grid: Optional[np.ndarray],
    correlation_loss_grid: Optional[np.ndarray],
    output_path: Path,
    show_interactive: bool = False,
) -> None:
    """Create combined 3D plot with selected loss landscapes, each with distinct color gradients."""
    surfaces = []
    labels = []
    min_losses = []
    min_positions = []
    
    if threshold_loss_grid is not None:
        threshold_loss_grid_T = threshold_loss_grid.T
        threshold_norm = normalize_parameter_space(THRESHOLD_VALUES)
        derivative_magnitude_norm = normalize_parameter_space(DERIVATIVE_MAGNITUDE_VALUES)
        X_thresh, Y_thresh = np.meshgrid(threshold_norm, derivative_magnitude_norm)
        min_idx = np.unravel_index(np.argmin(threshold_loss_grid_T), threshold_loss_grid_T.shape)
        min_losses.append(threshold_loss_grid_T[min_idx])
        min_positions.append((X_thresh[min_idx], Y_thresh[min_idx], threshold_loss_grid_T[min_idx]))
        surfaces.append((X_thresh, Y_thresh, threshold_loss_grid_T, BLUE_GRADIENT, 'Threshold', 'darkblue'))
        labels.append('Threshold')
    
    if derivative_loss_grid is not None:
        derivative_loss_grid_T_reversed = np.flip(derivative_loss_grid.T, axis=1)
        lower_norm = normalize_parameter_space(DERIVATIVE_LOWER_VALUES_REVERSED)
        upper_norm = normalize_parameter_space(DERIVATIVE_UPPER_VALUES)
        X_deriv, Y_deriv = np.meshgrid(lower_norm, upper_norm)
        min_idx = np.unravel_index(np.argmin(derivative_loss_grid_T_reversed), derivative_loss_grid_T_reversed.shape)
        min_losses.append(derivative_loss_grid_T_reversed[min_idx])
        min_positions.append((X_deriv[min_idx], Y_deriv[min_idx], derivative_loss_grid_T_reversed[min_idx]))
        surfaces.append((X_deriv, Y_deriv, derivative_loss_grid_T_reversed, RED_GRADIENT, 'Derivative', 'darkred'))
        labels.append('Derivative')
    
    if correlation_loss_grid is not None:
        correlation_loss_grid_T = correlation_loss_grid.T
        negative_norm = normalize_parameter_space(NEGATIVE_FRAMES_VALUES)
        positive_norm = normalize_parameter_space(POSITIVE_FRAMES_VALUES)
        X_corr, Y_corr = np.meshgrid(negative_norm, positive_norm)
        min_idx = np.unravel_index(np.argmin(correlation_loss_grid_T), correlation_loss_grid_T.shape)
        min_losses.append(correlation_loss_grid_T[min_idx])
        min_positions.append((X_corr[min_idx], Y_corr[min_idx], correlation_loss_grid_T[min_idx]))
        surfaces.append((X_corr, Y_corr, correlation_loss_grid_T, GREEN_GRADIENT, 'Correlation', 'darkgreen'))
        labels.append('Correlation')
    
    if not surfaces:
        print("No loss grids provided for plotting")
        return

    # Main 3D plot
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    plot_surfaces = []
    for X, Y, Z, cmap, label, color in surfaces:
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.6, linewidth=0.1, antialiased=True)
        plot_surfaces.append((surf, label))
    
    # Mark and label minimums
    legend_elements = []
    for i, ((x, y, z), (label, color)) in enumerate(zip(min_positions, [(l, c) for _, _, _, _, l, c in surfaces])):
        ax.scatter([x], [y], [z], color=color, s=200, marker='*', edgecolors='white', linewidths=2)
        ax.text(x, y, z, f'  {label}\n  min={min_losses[i]:.2f}', fontsize=9, color=color, weight='bold')
        legend_elements.append(Patch(facecolor=color, alpha=0.6, label=f'{label} (min={min_losses[i]:.2f})'))
    
    ax.set_xlabel('Normalized Parameter 1 (0-1)', fontsize=12)
    ax.set_ylabel('Normalized Parameter 2 (0-1)', fontsize=12)
    ax.set_zlabel('Precision Loss (FP + FN)', fontsize=12)
    ax.set_title('Combined Precision Loss Landscapes', fontsize=14, fontweight='bold')
    
    for i, (surf, label) in enumerate(plot_surfaces):
        fig.colorbar(surf, ax=ax, shrink=0.3, aspect=15, label=f'{label} Loss', pad=0.02 + i*0.15)
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    ax.text2D(0.02, 0.02,
              'Threshold: X=Threshold (90-300), Y=Derivative Magnitude (0-100)\n'
              'Derivative: X=Lower Threshold (-100 to 0), Y=Upper Threshold (0-100)\n'
              'Correlation: X=Negative Frames (1-20), Y=Positive Frames (1-20)',
              transform=ax.transAxes, fontsize=8,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
              verticalalignment='bottom')
    
    ax.view_init(elev=30, azim=45)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined loss plot to {output_path}")
    plt.close(fig)

    # Side view (orthographic projection)
    fig_side = plt.figure(figsize=(18, 10))
    ax_side = fig_side.add_subplot(111, projection='3d')
    
    for X, Y, Z, cmap, label, color in surfaces:
        ax_side.plot_surface(X, Y, Z, cmap=cmap, alpha=0.6, linewidth=0.1, antialiased=True)
    
    for i, ((x, y, z), (label, color)) in enumerate(zip(min_positions, [(l, c) for _, _, _, _, l, c in surfaces])):
        ax_side.scatter([x], [y], [z], color=color, s=200, marker='*', edgecolors='white', linewidths=2)
        ax_side.text(x, y, z, f'  {label[0]}={min_losses[i]:.2f}', fontsize=10, color=color, weight='bold')
    
    ax_side.set_xlabel('Normalized Parameter 1 (0-1)', fontsize=12)
    ax_side.set_ylabel('Normalized Parameter 2 (0-1)', fontsize=12)
    ax_side.set_zlabel('Precision Loss (FP + FN)', fontsize=12)
    ax_side.set_title('Combined Loss Landscapes - Side View (Orthographic)', fontsize=14, fontweight='bold')
    ax_side.legend(handles=legend_elements, loc='upper left', fontsize=10)
    ax_side.view_init(elev=0, azim=0)
    
    side_output_path = output_path.parent / f"{output_path.stem}_side_view.png"
    plt.savefig(side_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved side view plot to {side_output_path}")
    plt.close(fig_side)
    
    if show_interactive:
        plt.show()


def main() -> None:
    print("="*80)
    print("UNIFIED LOSS LANDSCAPE COMPARISON")
    print("="*80)
    print("Using optimized concatenated approach")
    print("Loss = False Positives + False Negatives (based on precise jump boundaries)")
    print(f"Started at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    
    # Algorithm selection via command line or prompt
    import argparse
    parser = argparse.ArgumentParser(description='Generate loss landscape plots')
    parser.add_argument('--algorithms', nargs='+', choices=['threshold', 'derivative', 'correlation', 'all'],
                       help='Algorithms to process (default: prompt user)')
    parser.add_argument('--no-prompt', action='store_true', help='Skip interactive prompt, use all algorithms')
    args = parser.parse_args()
    
    if args.algorithms:
        if 'all' in args.algorithms:
            selected_algorithms = ["threshold", "derivative", "correlation"]
        else:
            selected_algorithms = args.algorithms
    elif args.no_prompt:
        selected_algorithms = ["threshold", "derivative", "correlation"]
    else:
        selected_algorithms = prompt_algorithm_selection()
    
    print(f"\nSelected algorithms: {', '.join(selected_algorithms)}")
    
    # Load data files
    data_files = [Path(f) for f in find_all_data_files(PROJECT_ROOT / "dataset")]
    print(f"\nLoading data files and annotations for {len(data_files)} files...")
    data_files_with_annotations = []
    total_markers = 0
    
    for data_file in data_files:
        annotations = load_annotations(data_file)
        if annotations is None or not annotations.markers:
            continue
        data_files_with_annotations.append((data_file, annotations.markers))
        total_markers += len(annotations.markers)
        print(f"  ✓ {data_file.stem}: {len(annotations.markers)} markers")
    
    if not data_files_with_annotations:
        print("\nNo annotated files found. Please create annotations using annotate_jumps.py")
        return
    
    print(f"\nTotal: {len(data_files_with_annotations)} annotated files with {total_markers} total markers")
    print("\nConcatenating all data files...")
    concatenated_data, concatenated_markers, _ = concatenate_data_and_annotations(data_files_with_annotations)
    print(f"Concatenated data shape: {concatenated_data.shape}")
    print(f"Total markers: {len(concatenated_markers)}")

    threshold_loss_grid = None
    derivative_loss_grid = None
    correlation_loss_grid = None
    
    if "threshold" in selected_algorithms:
        print("\n" + "="*80)
        print("PROCESSING THRESHOLD ALGORITHM")
        print("="*80)
        threshold_loss_grid = grid_search_threshold_algorithm(
            concatenated_data, concatenated_markers, show_progress=True, max_workers=None
        )
        print(f"Threshold algorithm: Loss grid shape: {threshold_loss_grid.shape}")
        threshold_loss_grid_T = threshold_loss_grid.T
        plot_3d_threshold(
            threshold_loss_grid_T, THRESHOLD_VALUES, DERIVATIVE_MAGNITUDE_VALUES,
            "Threshold", "Derivative magnitude", "Combined Loss Function - Threshold Algorithm",
            OUTPUT_DIR / "combined_loss_threshold.png", show_interactive=False
        )
        print(f"  ✓ Saved threshold plot to: {OUTPUT_DIR / 'combined_loss_threshold.png'}")
    
    if "derivative" in selected_algorithms:
        print("\n" + "="*80)
        print("PROCESSING DERIVATIVE ALGORITHM")
        print("="*80)
        derivative_loss_grid = grid_search_derivative_algorithm(
            concatenated_data, concatenated_markers, show_progress=True, max_workers=None
        )
        print(f"Derivative algorithm: Loss grid shape: {derivative_loss_grid.shape}")
        derivative_loss_grid_reversed = np.flip(derivative_loss_grid, axis=1)
        plot_3d_derivative(
            derivative_loss_grid_reversed, DERIVATIVE_LOWER_VALUES_REVERSED, DERIVATIVE_UPPER_VALUES,
            "Lower threshold (reversed)", "Upper threshold", "Combined Loss Function - Derivative Algorithm",
            OUTPUT_DIR / "combined_loss_derivative.png", show_interactive=False
        )
        print(f"  ✓ Saved derivative plot to: {OUTPUT_DIR / 'combined_loss_derivative.png'}")
        plot_bottom_view(
            derivative_loss_grid_reversed, DERIVATIVE_LOWER_VALUES_REVERSED, DERIVATIVE_UPPER_VALUES,
            "Lower threshold (reversed)", "Upper threshold", "Combined Loss Function - Derivative Algorithm (Bottom View)",
            OUTPUT_DIR / "combined_loss_derivative_bottom_view.png"
        )
        print(f"  ✓ Saved derivative bottom view to: {OUTPUT_DIR / 'combined_loss_derivative_bottom_view.png'}")
    
    if "correlation" in selected_algorithms:
        print("\n" + "="*80)
        print("PROCESSING CORRELATION ALGORITHM")
        print("="*80)
        best_loss_grid, _ = grid_search_width_combinations(
            concatenated_data, concatenated_markers, show_progress=True, max_workers=None
        )
        correlation_loss_grid = best_loss_grid
        print(f"Correlation algorithm: Loss grid shape: {correlation_loss_grid.shape}")
        plot_2d_loss_heatmap(
            best_loss_grid, NEGATIVE_FRAMES_VALUES, POSITIVE_FRAMES_VALUES,
            "Negative Frames", "Positive Frames", "Combined Loss Function - Correlation Algorithm (Optimized Threshold)",
            OUTPUT_DIR / "combined_loss_correlation_optimized_heatmap.png"
        )
        print(f"  ✓ Saved correlation heatmap to: {OUTPUT_DIR / 'combined_loss_correlation_optimized_heatmap.png'}")
        plot_3d_loss_surface(
            best_loss_grid, NEGATIVE_FRAMES_VALUES, POSITIVE_FRAMES_VALUES,
            "Negative Frames", "Positive Frames", "Combined Loss Function - Correlation Algorithm (Optimized Threshold)",
            OUTPUT_DIR / "combined_loss_correlation_optimized_3d.png", show_interactive=False
        )
        print(f"  ✓ Saved correlation 3D plot to: {OUTPUT_DIR / 'combined_loss_correlation_optimized_3d.png'}")
    
    # Create combined overlay plot if multiple algorithms selected
    if len(selected_algorithms) > 1:
        print("\n" + "="*80)
        print("CREATING COMBINED OVERLAY PLOT")
        print("="*80)
        plot_combined_loss_landscapes(
            threshold_loss_grid, derivative_loss_grid, correlation_loss_grid,
            OUTPUT_DIR / "combined_precision_loss_all_algorithms.png",
            show_interactive=False
        )

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS:")
    print("="*80)
    print(f"Total annotated files processed: {len(data_files_with_annotations)}")
    print(f"Total ground truth markers: {total_markers}")
    if data_files_with_annotations:
        print(f"Average markers per file: {total_markers / len(data_files_with_annotations):.1f}")
    
    if threshold_loss_grid is not None:
        min_idx = np.unravel_index(np.argmin(threshold_loss_grid), threshold_loss_grid.shape)
        min_threshold = THRESHOLD_VALUES[min_idx[0]]
        min_deriv_mag = DERIVATIVE_MAGNITUDE_VALUES[min_idx[1]]
        min_loss = threshold_loss_grid[min_idx]
        print(f"\nThreshold algorithm:")
        print(f"  Global minimum loss: {min_loss:.2f}")
        print(f"    Threshold: {min_threshold:.2f}, Derivative Magnitude: {min_deriv_mag:.2f}")
        print(f"  NOTE: Config defaults are threshold={THRESHOLD_DEFAULT:.2f}, derivative_threshold={DERIVATIVE_THRESHOLD_DEFAULT:.2f}")
        print(f"        Consider updating config.py if these optimized values are better.")
    
    if derivative_loss_grid is not None:
        min_idx = np.unravel_index(np.argmin(derivative_loss_grid), derivative_loss_grid.shape)
        min_upper = DERIVATIVE_UPPER_VALUES[min_idx[0]]
        min_lower = DERIVATIVE_LOWER_VALUES[min_idx[1]]
        min_loss = derivative_loss_grid[min_idx]
        print(f"\nDerivative algorithm:")
        print(f"  Global minimum loss: {min_loss:.2f}")
        print(f"    Upper threshold: {min_upper:.2f}, Lower threshold: {min_lower:.2f}")
        print(f"  NOTE: Config defaults are upper={DERIVATIVE_UPPER_DEFAULT:.2f}, lower={DERIVATIVE_LOWER_DEFAULT:.2f}")
        print(f"        These should match the optimized values from loss_derivative.py.")
    
    if correlation_loss_grid is not None:
        min_idx = np.unravel_index(np.argmin(correlation_loss_grid), correlation_loss_grid.shape)
        min_neg = NEGATIVE_FRAMES_VALUES[min_idx[0]]
        min_pos = POSITIVE_FRAMES_VALUES[min_idx[1]]
        min_loss = correlation_loss_grid[min_idx]
        print(f"\nCorrelation algorithm:")
        print(f"  Global minimum loss: {min_loss:.2f}")
        print(f"    Negative frames: {int(min_neg)}, Positive frames: {int(min_pos)}")
    
    print("="*80)
    print(f"\nCompleted at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
