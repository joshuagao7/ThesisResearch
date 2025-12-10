from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# Unified color palette - car-inspired gradient scheme
PRIMARY_RED = '#C41E3A'      # Deep red
PRIMARY_ORANGE = '#FF6B35'   # Vibrant orange
PRIMARY_YELLOW = '#FFD23F'   # Golden yellow
PRIMARY_BLUE = '#0066CC'     # Electric blue

# Create gradient colormap
GRADIENT_COLORS = [PRIMARY_RED, PRIMARY_ORANGE, PRIMARY_YELLOW, PRIMARY_BLUE]
GRADIENT_CMAP = LinearSegmentedColormap.from_list('red_orange_yellow_blue', GRADIENT_COLORS, N=100)

# Set consistent font styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

import sys
# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from jump_detection.algorithms.derivative import (
    DerivativeParameters,
    _run_derivative_pipeline,
)
from jump_detection.analysis.precise import calculate_precise_jump_boundaries
from jump_detection.analysis.loss import compute_precision_loss
from jump_detection.annotations import load_annotations
from jump_detection.config import SAMPLING_RATE, DEFAULT_DATA_FILES
from jump_detection.data import load_dataset
from jump_detection.types import DetectionResult, Jump, DerivativeNames

# Ranges for derivative threshold values
DERIVATIVE_UPPER_VALUES = np.linspace(0, 100, 60)  # Upper threshold: 0-100
DERIVATIVE_LOWER_VALUES = np.linspace(-100, 0, 60)  # Lower threshold: -100-0
# Reverse lower values so 0 is at bottom-left (descending order)
DERIVATIVE_LOWER_VALUES_REVERSED = np.flip(DERIVATIVE_LOWER_VALUES)
IN_AIR_THRESHOLD = 250  # Updated to match detailedplot2.py

# Get project root for path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# Use DEFAULT_DATA_FILES from config, converting Path objects to strings
DATA_FILES = [str(f) for f in DEFAULT_DATA_FILES]

OUTPUT_DIR = PROJECT_ROOT / "results" / "plots" / "loss"
SEARCH_WINDOW = 70  # For precise jump detection


def concatenate_all_data_and_annotations(
    data_files_with_annotations: list[tuple[Path, list[int]]],
) -> tuple[np.ndarray, list[int], list[int]]:
    """Concatenate all data files and annotations into single arrays.
    
    Args:
        data_files_with_annotations: List of (data_file_path, ground_truth_markers) tuples
        
    Returns:
        Tuple of (concatenated_raw_data, concatenated_markers, file_boundaries)
        - concatenated_raw_data: All sensor data concatenated vertically
        - concatenated_markers: All annotation markers with offsets applied
        - file_boundaries: Cumulative lengths of each file (for debugging)
    """
    all_data = []
    all_markers = []
    file_boundaries = [0]  # Start indices for each file in concatenated data
    
    current_offset = 0
    
    for data_file, ground_truth_markers in data_files_with_annotations:
        # Load raw data
        raw_data = load_dataset(data_file)
        all_data.append(raw_data)
        
        # Offset markers by current position in concatenated data
        offset_markers = [m + current_offset for m in ground_truth_markers]
        all_markers.extend(offset_markers)
        
        # Update offset for next file
        current_offset += len(raw_data)
        file_boundaries.append(current_offset)
    
    # Concatenate all data vertically
    concatenated_data = np.vstack(all_data)
    
    return concatenated_data, all_markers, file_boundaries


def compute_precise_jumps_from_concatenated(
    concatenated_data: np.ndarray,
    result: DetectionResult,
) -> list[Jump]:
    """Calculate precise jump boundaries from detection result on concatenated data.
    
    Args:
        concatenated_data: Concatenated raw data from all files
        result: DetectionResult from derivative pipeline
        
    Returns:
        List of Jump objects with precise boundaries
    """
    signal = result.pooled_data if result.pooled_data is not None else concatenated_data.sum(axis=1)
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


def compute_loss_with_annotations(
    detected_jumps: list,
    ground_truth_markers: list[int],
) -> float:
    """Compute loss using ground truth annotations.
    
    Args:
        detected_jumps: List of detected Jump objects (with precise boundaries)
        ground_truth_markers: List of frame indices that should be in correct jumps
        
    Returns:
        Loss value (false_positives + false_negatives)
        
    Raises:
        ValueError: If ground_truth_markers is empty or None
    """
    if ground_truth_markers is None or len(ground_truth_markers) == 0:
        raise ValueError("Ground truth markers are required but not provided")
    
    # Compute precision loss using ground truth
    metrics = compute_precision_loss(detected_jumps, ground_truth_markers)
    return float(metrics["loss"])


def _init_progress(total: int) -> tuple[int, int, int]:
    bar_length = 40
    report_interval = max(1, total // bar_length)
    print("Progress: [", end="", flush=True)
    return 0, bar_length, report_interval


def _update_progress(count: int, total: int, bar_length: int, report_interval: int) -> None:
    if count % report_interval == 0 or count == total:
        progress = count / total
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\rProgress: [{bar}] {progress * 100:5.1f}%", end="", flush=True)


def _finalize_progress() -> None:
    print()


def compute_loss_for_parameter_combination(
    concatenated_data: np.ndarray,
    concatenated_markers: list[int],
    upper_threshold: float,
    lower_threshold: float,
) -> float:
    """Compute loss for concatenated data with given parameters.
    
    Uses the same code path as detect_derivative_jumps() but with raw data.
    
    Args:
        concatenated_data: All sensor data concatenated vertically
        concatenated_markers: All annotation markers with offsets applied
        upper_threshold: Upper derivative threshold parameter
        lower_threshold: Lower derivative threshold parameter
        
    Returns:
        Loss value (false_positives + false_negatives)
    """
    # Create parameters exactly as the public API does
    params = DerivativeParameters(
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        in_air_threshold=IN_AIR_THRESHOLD,
    )
    
    # Run derivative pipeline using the same code as detect_derivative_jumps()
    signals, jumps, metadata = _run_derivative_pipeline(concatenated_data, params)
    
    # Update metadata exactly as detect_derivative_jumps() does
    metadata.update({
        "parameters": params.as_dict(),
        "export_paths": [],  # No file exports for concatenated data
        "data_file_path": None,  # No file path for concatenated data
    })
    
    # Create DetectionResult exactly as detect_derivative_jumps() does
    pooled = signals[DerivativeNames.POOLED.value]
    result = DetectionResult(
        participant_name=None,
        sampling_rate=SAMPLING_RATE,
        raw_data=concatenated_data,
        pooled_data=pooled,
        signals=signals,
        jumps=jumps,
        metadata=metadata,
    )
    
    # Calculate precise jumps with refined boundaries
    precise_jumps = compute_precise_jumps_from_concatenated(concatenated_data, result)
    
    # Compute loss using ground truth annotations
    loss = compute_loss_with_annotations(precise_jumps, concatenated_markers)
    
    return loss


def compute_loss_for_parameter_combination_wrapper(
    args: tuple[np.ndarray, list[int], float, float],
) -> tuple[float, float, float]:
    """Wrapper function for parallel processing of parameter combinations.
    
    Args:
        args: Tuple of (concatenated_data, concatenated_markers, upper_threshold, lower_threshold)
        
    Returns:
        Tuple of (upper_threshold, lower_threshold, loss)
    """
    concatenated_data, concatenated_markers, upper_threshold, lower_threshold = args
    loss = compute_loss_for_parameter_combination(
        concatenated_data,
        concatenated_markers,
        upper_threshold,
        lower_threshold,
    )
    return upper_threshold, lower_threshold, loss


def grid_search_derivative_algorithm(
    concatenated_data: np.ndarray,
    concatenated_markers: list[int],
    show_progress: bool = False,
    max_workers: int | None = None,
) -> np.ndarray:
    """Run grid search and return grid of loss values using precise loss detection.
    
    Args:
        concatenated_data: All sensor data concatenated vertically
        concatenated_markers: All annotation markers with offsets applied
        show_progress: Whether to show progress bar
        max_workers: Maximum number of parallel workers (None = use CPU count)
        
    Returns:
        2D array of loss values [upper_threshold, lower_threshold]
    """
    total = len(DERIVATIVE_UPPER_VALUES) * len(DERIVATIVE_LOWER_VALUES)
    
    if show_progress:
        counter, bar_length, report_interval = _init_progress(total)
    
    loss_grid = np.zeros((len(DERIVATIVE_UPPER_VALUES), len(DERIVATIVE_LOWER_VALUES)), dtype=float)
    
    # Prepare all parameter combinations for parallel processing
    parameter_combinations = [
        (concatenated_data, concatenated_markers, float(upper), float(lower))
        for upper in DERIVATIVE_UPPER_VALUES
        for lower in DERIVATIVE_LOWER_VALUES
    ]
    
    # Create index mapping for results
    index_map = {}
    for i, upper in enumerate(DERIVATIVE_UPPER_VALUES):
        for j, lower in enumerate(DERIVATIVE_LOWER_VALUES):
            index_map[(float(upper), float(lower))] = (i, j)
    
    # Process parameter combinations in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_combo = {
            executor.submit(compute_loss_for_parameter_combination_wrapper, combo): combo
            for combo in parameter_combinations
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_combo):
            upper_threshold, lower_threshold, loss = future.result()
            i, j = index_map[(upper_threshold, lower_threshold)]
            loss_grid[i, j] = loss
            
            if show_progress:
                counter += 1
                _update_progress(counter, total, bar_length, report_interval)
    
    if show_progress:
        _finalize_progress()
    return loss_grid


def plot_bottom_view(
    loss_grid: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    loss_threshold: float = 5.0,
    mark_specific_point: tuple[float, float] | None = None,
) -> tuple[float, float, float]:
    """Create a 2D bottom-view (contour/heatmap) plot with highlighted optimized regions.
    
    Returns:
        Tuple of (min_lower_threshold, min_upper_threshold, min_loss) for the global minimum.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create meshgrid for 2D plotting
    X, Y = np.meshgrid(x_values, y_values)
    
    # Create contour plot
    contours = ax.contour(X, Y, loss_grid, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')
    
    # Create filled contour/heatmap
    im = ax.contourf(X, Y, loss_grid, levels=50, cmap=GRADIENT_CMAP, alpha=0.8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Loss (False Positives + False Negatives)', rotation=270, labelpad=20)
    
    # Highlight optimized regions (low loss)
    optimized_mask = loss_grid <= loss_threshold
    if optimized_mask.any():
        # Create overlay for optimized regions using contour
        ax.contour(
            X, Y, loss_grid,
            levels=[loss_threshold],
            colors=['red'],
            linewidths=3,
            linestyles='--',
        )
        # Fill optimized regions
        from numpy import ma
        optimized_overlay = ma.masked_where(~optimized_mask, np.ones_like(loss_grid))
        ax.contourf(
            X, Y, 
            optimized_overlay,
            levels=[0.5, 1.5],
            colors=['yellow'],
            alpha=0.3,
        )
        # Add text annotation
        ax.text(
            0.02, 0.98,
            f'Optimized regions (loss ≤ {loss_threshold}) shown in yellow',
            transform=ax.transAxes,
            fontsize=11,
            fontweight='bold',
            color=PRIMARY_RED,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )
    
    # Find and mark the global minimum
    min_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
    min_x = x_values[min_idx[1]]
    min_y = y_values[min_idx[0]]
    min_loss = loss_grid[min_idx]
    
    # Trace the global minimum region - highlight a small region around it
    # Find all points within a small loss threshold of the global minimum
    min_region_threshold = min_loss + 0.5  # Points within 0.5 of global minimum
    min_region_mask = loss_grid <= min_region_threshold
    if min_region_mask.any():
        # Draw a contour around the minimum region
        ax.contour(
            X, Y, loss_grid,
            levels=[min_region_threshold],
            colors=['magenta'],
            linewidths=2,
            linestyles='-',
            alpha=0.8,
        )
    
    # Mark the global minimum with a large red star
    ax.plot(min_x, min_y, 'r*', markersize=25, markeredgewidth=2, 
            markeredgecolor=PRIMARY_RED, label=f'Global minimum: Lower={min_x:.2f}, Upper={min_y:.2f}, Loss={min_loss:.2f}')
    
    # Mark specific point if provided (upper, lower)
    if mark_specific_point is not None:
        spec_upper, spec_lower = mark_specific_point
        # Find the closest grid point
        spec_lower_idx = np.argmin(np.abs(x_values - spec_lower))
        spec_upper_idx = np.argmin(np.abs(y_values - spec_upper))
        spec_x = x_values[spec_lower_idx]
        spec_y = y_values[spec_upper_idx]
        spec_loss = loss_grid[spec_upper_idx, spec_lower_idx]
        
        # Mark with a cyan circle
        ax.plot(spec_x, spec_y, 'co', markersize=15, markeredgewidth=2,
                markeredgecolor=PRIMARY_BLUE, label=f'Specified point: Lower={spec_x:.2f}, Upper={spec_y:.2f}, Loss={spec_loss:.2f}')
        
        # Add annotation line
        ax.annotate(
            f'({spec_x:.1f}, {spec_y:.1f})',
            xy=(spec_x, spec_y),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=PRIMARY_BLUE, alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
        )
    
    ax.legend(loc='upper right', fontsize=10)
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved bottom view plot to {output_path}")
    plt.close(fig)
    
    return min_x, min_y, min_loss


def plot_3d_loss(
    loss_grid: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    show_interactive: bool = False,
    mark_specific_point: tuple[float, float] | None = None,
) -> None:
    """Create a 3D surface plot of the loss values."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for 3D plotting
    X, Y = np.meshgrid(x_values, y_values)

    # Plot surface
    surf = ax.plot_surface(
        X, Y, loss_grid,
        cmap='viridis',
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
               edgecolors='darkred', linewidths=2, label=f'Global min: Lower={min_x:.2f}, Upper={min_y:.2f}, Loss={min_loss:.2f}')

    # Mark specific point if provided (upper, lower)
    if mark_specific_point is not None:
        spec_upper, spec_lower = mark_specific_point
        # Find the closest grid point
        spec_lower_idx = np.argmin(np.abs(x_values - spec_lower))
        spec_upper_idx = np.argmin(np.abs(y_values - spec_upper))
        spec_x = x_values[spec_lower_idx]
        spec_y = y_values[spec_upper_idx]
        spec_loss = loss_grid[spec_upper_idx, spec_lower_idx]
        
        ax.scatter([spec_x], [spec_y], [spec_loss], color=PRIMARY_BLUE, s=150, marker='o',
                   edgecolors='darkcyan', linewidths=2, label=f'Specified: Lower={spec_x:.2f}, Upper={spec_y:.2f}, Loss={spec_loss:.2f}')

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_zlabel('Loss (False Positives + False Negatives)', fontsize=12)
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
    
    # If show_interactive is True, display the plot for interactive rotation
    if show_interactive:
        plt.show()
    else:
        plt.close(fig)


def plot_3d_loss_mesh(
    loss_grid: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    show_interactive: bool = False,
) -> None:
    """Create a 3D mesh/wireframe plot of the loss values showing grid structure."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for 3D plotting
    X, Y = np.meshgrid(x_values, y_values)

    # Plot wireframe mesh - shows the grid structure clearly
    wire = ax.plot_wireframe(
        X, Y, loss_grid,
        color=PRIMARY_BLUE,
        linewidth=0.5,
        alpha=0.6,
    )
    
    # Also add a semi-transparent surface with visible grid lines for better visualization
    surf = ax.plot_surface(
        X, Y, loss_grid,
        cmap='viridis',
        alpha=0.3,
        linewidth=0.5,
        edgecolor='black',
        antialiased=True,
    )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_zlabel('Loss (False Positives + False Negatives)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar based on the surface
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Loss')

    # Set viewing angle
    ax.view_init(elev=30, azim=45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D mesh plot to {output_path}")
    
    # If show_interactive is True, display the plot for interactive rotation
    if show_interactive:
        plt.show()
    else:
        plt.close(fig)


def plot_3d_loss_mesh_with_water(
    loss_grid: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
    water_level: float = 3.0,
    show_interactive: bool = False,
) -> None:
    """Create a 3D mesh plot with solid blue water filling from z=0 to z=water_level, 
    then erasing water where loss < water_level (i.e., where loss surface cuts through water)."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for 3D plotting
    X, Y = np.meshgrid(x_values, y_values)

    # Plot wireframe mesh - shows the full loss surface structure
    wire = ax.plot_wireframe(
        X, Y, loss_grid,
        color=PRIMARY_BLUE,
        linewidth=0.5,
        alpha=0.4,
    )
    
    # Create solid water block from z=0 to z=water_level
    # But erase water where loss_grid < water_level (where loss surface cuts through the water)
    # So water fills from the loss surface up to water_level, but only where loss <= water_level
    
    # Create water top surface at water_level
    water_top = np.full_like(loss_grid, water_level)
    
    # Create water bottom surface - follows the loss_grid, but only where loss <= water_level
    # Where loss > water_level, no water (set to NaN)
    # Where loss <= water_level, water bottom is at the loss surface (so water fills from loss up to water_level)
    water_bottom = np.where(loss_grid <= water_level, loss_grid, np.nan)
    
    # Also create a flat bottom at z=0 for areas where loss <= water_level
    # This creates the "solid fill" effect from z=0 to z=water_level
    water_bottom_flat = np.where(loss_grid <= water_level, 0, np.nan)
    
    # Mask: only show water where loss_grid <= water_level (acceptable regions)
    water_mask = loss_grid <= water_level
    
    # Create masked water surfaces
    masked_water_top = np.where(water_mask, water_top, np.nan)
    masked_water_bottom = np.where(water_mask, water_bottom_flat, np.nan)
    
    # Plot the top surface of the water (at water_level) - solid blue
    water_top_surf = ax.plot_surface(
        X, Y, masked_water_top,
        color=PRIMARY_BLUE,
        alpha=0.8,
        linewidth=0.1,
        edgecolor=PRIMARY_BLUE,
        antialiased=True,
        shade=True,
    )
    
    # Plot the bottom surface of the water (at z=0) - solid blue
    water_bottom_surf = ax.plot_surface(
        X, Y, masked_water_bottom,
        color=PRIMARY_BLUE,
        alpha=0.8,
        linewidth=0.1,
        edgecolor=PRIMARY_BLUE,
        antialiased=True,
        shade=True,
    )
    
    # Plot the loss surface where it cuts through the water (where loss <= water_level)
    # This shows where the water gets "erased" by the loss landscape
    water_cut_surface = np.where(water_mask, loss_grid, np.nan)
    water_cut_surf = ax.plot_surface(
        X, Y, water_cut_surface,
        color=PRIMARY_YELLOW,
        alpha=0.6,
        linewidth=0.2,
        edgecolor=PRIMARY_BLUE,
        antialiased=True,
        shade=True,
    )
    
    # Also show the full loss surface with low opacity for reference
    full_surf = ax.plot_surface(
        X, Y, loss_grid,
        cmap='viridis',
        alpha=0.3,
        linewidth=0.5,
        edgecolor='gray',
        antialiased=True,
    )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_zlabel('Loss (False Positives + False Negatives)', fontsize=12)
    ax.set_title(f"{title} - Solid Water Fill (z=0 to z={water_level}, erased where loss > {water_level})", 
                 fontsize=14, fontweight='bold')
    
    # Set z-axis limits to show the full range
    ax.set_zlim(0, max(np.max(loss_grid), water_level * 1.2))

    # Add colorbar based on the full surface
    fig.colorbar(full_surf, ax=ax, shrink=0.5, aspect=5, label='Loss')

    # Set viewing angle
    ax.view_init(elev=30, azim=45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D mesh plot with water to {output_path}")
    
    # If show_interactive is True, display the plot for interactive rotation
    if show_interactive:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    print("LOSS CALCULATION AND 3D VISUALIZATION (DERIVATIVE ALGORITHM)")
    print("Using ground truth annotations for loss calculation")
    print("Loss = False Positives + False Negatives (based on precise jump boundaries)")
    print(f"Started at: {datetime.now():%Y-%m-%d %H:%M:%S}")

    # Load all data files and their annotations upfront
    print(f"\nLoading data files and annotations for {len(DATA_FILES)} participants...")
    data_files_with_annotations = []
    participant_names = []
    
    for data_file in DATA_FILES:
        file_path = Path(data_file)
        participant = file_path.parent.name if file_path.parent != file_path else "Dataset"
        file_label = file_path.stem
        participant_names.append(participant)
        
        # Load ground truth annotations - REQUIRED
        annotations = load_annotations(file_path)
        
        if annotations is None:
            print(f"\n{'='*70}")
            print("ERROR: Missing annotations file")
            print(f"{'='*70}")
            print(f"File: {file_path}")
            print(f"Participant: {participant} – {file_label}")
            print(f"Expected annotation file: {file_path}_annotations.json")
            print(f"{'='*70}")
            raise FileNotFoundError(
                f"Annotations file not found for {file_path}. "
                f"Expected: {file_path}_annotations.json"
            )
        
        ground_truth_markers = annotations.markers
        
        if ground_truth_markers is None or len(ground_truth_markers) == 0:
            print(f"\n{'='*70}")
            print("ERROR: Empty annotations")
            print(f"{'='*70}")
            print(f"File: {file_path}")
            print(f"Participant: {participant} – {file_label}")
            print(f"Annotation file exists but contains no markers.")
            print(f"{'='*70}")
            raise ValueError(
                f"Annotations file for {file_path} exists but contains no markers. "
                f"Please add ground truth markers to the annotation file."
            )
        
        data_files_with_annotations.append((file_path, ground_truth_markers))
        print(f"  ✓ {participant} – {file_label}: {len(ground_truth_markers)} markers")
    
    print(f"\nSuccessfully loaded {len(data_files_with_annotations)} participants with annotations")
    
    # Concatenate all data and annotations upfront
    print("\nConcatenating all data files and annotations...")
    concatenated_data, concatenated_markers, file_boundaries = concatenate_all_data_and_annotations(
        data_files_with_annotations
    )
    print(f"  ✓ Concatenated data shape: {concatenated_data.shape}")
    print(f"  ✓ Total markers: {len(concatenated_markers)}")
    print(f"  ✓ File boundaries: {file_boundaries}")
    
    # Grid search: compute loss for all parameter combinations
    # Parallelized across parameter combinations
    print(f"\nGrid searching {len(DERIVATIVE_UPPER_VALUES)} × {len(DERIVATIVE_LOWER_VALUES)} = {len(DERIVATIVE_UPPER_VALUES) * len(DERIVATIVE_LOWER_VALUES)} parameter combinations...")
    print("Using precise jump detection and ground truth annotations...")
    print("Parallelizing across parameter combinations...")
    
    combined_loss_grid = grid_search_derivative_algorithm(
        concatenated_data,
        concatenated_markers,
        show_progress=True,
        max_workers=None,  # Use all available CPU cores
    )
    
    # Reverse the loss grid horizontally to match reversed lower threshold axis
    # (so that 0 is at bottom-left in the plot)
    combined_loss_grid_reversed = np.flip(combined_loss_grid, axis=1)
    
    # Find global minimum coordinates (before reversing for output)
    min_idx = np.unravel_index(np.argmin(combined_loss_grid), combined_loss_grid.shape)
    min_lower_original = DERIVATIVE_LOWER_VALUES[min_idx[1]]
    min_upper = DERIVATIVE_UPPER_VALUES[min_idx[0]]
    min_loss_value = combined_loss_grid[min_idx]
    
    # Print global minimum coordinates
    print(f"\n{'='*70}")
    print("GLOBAL MINIMUM COORDINATES:")
    print(f"{'='*70}")
    print(f"Lower threshold: {min_lower_original:.4f}")
    print(f"Upper threshold: {min_upper:.4f}")
    print(f"Loss value:      {min_loss_value:.4f}")
    print(f"{'='*70}\n")
    
    # Specific point to mark: upper=24, lower=-19.2
    specific_point = (24.0, -19.2)  # (upper, lower)
    print(f"Marking specified point: Upper={specific_point[0]}, Lower={specific_point[1]}")
    
    # Save bottom view (2D contour/heatmap) with optimized regions highlighted
    min_coords = plot_bottom_view(
        combined_loss_grid_reversed,
        DERIVATIVE_LOWER_VALUES_REVERSED,
        DERIVATIVE_UPPER_VALUES,
        "Lower threshold (descending: 0 → negative)",
        "Upper threshold",
        "Combined Loss Function - Bottom View (Precise Detection with Ground Truth)",
        OUTPUT_DIR / "combined_loss_derivative_bottom_view.png",
        loss_threshold=5.0,
        mark_specific_point=specific_point,
    )
    
    # Save static PNG version (3D surface plot) with reversed axes
    plot_3d_loss(
        combined_loss_grid_reversed,
        DERIVATIVE_LOWER_VALUES_REVERSED,
        DERIVATIVE_UPPER_VALUES,
        "Lower threshold (descending: 0 → negative)",
        "Upper threshold",
        "Combined Loss Function (Precise Detection with Ground Truth)",
        OUTPUT_DIR / "combined_loss_derivative.png",
        show_interactive=False,
        mark_specific_point=specific_point,
    )
    
    # Display interactive version (rotatable)
    print("\nDisplaying interactive 3D plot (rotate to view)...")
    print("Close the plot window to continue.")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for 3D plotting with reversed lower values
    X, Y = np.meshgrid(DERIVATIVE_LOWER_VALUES_REVERSED, DERIVATIVE_UPPER_VALUES)
    
    # Plot surface
    surf = ax.plot_surface(
        X, Y, combined_loss_grid_reversed,
        cmap='viridis',
        alpha=0.9,
        linewidth=0,
        antialiased=True,
    )
    
    # Mark global minimum in the reversed grid
    min_idx_reversed = np.unravel_index(np.argmin(combined_loss_grid_reversed), combined_loss_grid_reversed.shape)
    min_x_reversed = DERIVATIVE_LOWER_VALUES_REVERSED[min_idx_reversed[1]]
    min_y_reversed = DERIVATIVE_UPPER_VALUES[min_idx_reversed[0]]
    min_loss_reversed = combined_loss_grid_reversed[min_idx_reversed]
    ax.scatter([min_x_reversed], [min_y_reversed], [min_loss_reversed], color=PRIMARY_RED, s=200, marker='*',
               edgecolors='darkred', linewidths=2, label=f'Global min: Lower={min_lower_original:.2f}, Upper={min_upper:.2f}, Loss={min_loss_value:.2f}')
    
    # Mark specific point
    spec_lower_idx_rev = np.argmin(np.abs(DERIVATIVE_LOWER_VALUES_REVERSED - specific_point[1]))
    spec_upper_idx = np.argmin(np.abs(DERIVATIVE_UPPER_VALUES - specific_point[0]))
    spec_x_rev = DERIVATIVE_LOWER_VALUES_REVERSED[spec_lower_idx_rev]
    spec_y_rev = DERIVATIVE_UPPER_VALUES[spec_upper_idx]
    spec_loss_rev = combined_loss_grid_reversed[spec_upper_idx, spec_lower_idx_rev]
    ax.scatter([spec_x_rev], [spec_y_rev], [spec_loss_rev], color=PRIMARY_BLUE, s=150, marker='o',
               edgecolors='darkcyan', linewidths=2, label=f'Specified: Lower={specific_point[1]:.2f}, Upper={specific_point[0]:.2f}, Loss={spec_loss_rev:.2f}')
    
    ax.set_xlabel("Lower threshold (descending: 0 → negative)", fontsize=12)
    ax.set_ylabel("Upper threshold", fontsize=12)
    ax.set_zlabel('Loss (False Positives + False Negatives)', fontsize=12)
    ax.set_title("Combined Loss Function (Precise Detection with Ground Truth) - Interactive (Rotate to View)", 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Loss')
    
    # Add legend
    ax.legend(loc='upper left', fontsize=9)
    
    # Set viewing angle
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()  # This makes it interactive and rotatable

    print(f"\nCompleted at: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()

