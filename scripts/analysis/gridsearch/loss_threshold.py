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
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from jump_detection.algorithms.threshold import (
    ThresholdParameters,
    _run_threshold_pipeline,
)
from jump_detection.analysis.precise import calculate_precise_jump_boundaries
from jump_detection.analysis.loss import compute_precision_loss
from jump_detection.annotations import load_annotations
from jump_detection.config import SAMPLING_RATE, DEFAULT_DATA_FILES
from jump_detection.data import load_dataset
from jump_detection.types import DetectionResult, Jump, ThresholdNames

THRESHOLD_VALUES = np.linspace(90, 300, 40)
DERIVATIVE_MAGNITUDE_VALUES = np.linspace(0, 100, 40)

# Get project root for path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

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
        result: DetectionResult from threshold pipeline
        
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
    threshold: float,
    derivative_threshold: float,
) -> float:
    """Compute loss for concatenated data with given parameters.
    
    Args:
        concatenated_data: All sensor data concatenated vertically
        concatenated_markers: All annotation markers with offsets applied
        threshold: Threshold parameter
        derivative_threshold: Derivative threshold parameter
        
    Returns:
        Loss value (false_positives + false_negatives)
    """
    params = ThresholdParameters(threshold=threshold, derivative_threshold=derivative_threshold)
    
    # Run threshold pipeline directly on concatenated data
    signals, jumps, metadata = _run_threshold_pipeline(concatenated_data, params)
    
    # Create DetectionResult for compatibility with precise jump calculation
    result = DetectionResult(
        participant_name=None,
        sampling_rate=SAMPLING_RATE,
        raw_data=concatenated_data,
        pooled_data=signals[ThresholdNames.AVERAGE.value],
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
        args: Tuple of (concatenated_data, concatenated_markers, threshold, derivative_threshold)
        
    Returns:
        Tuple of (threshold, derivative_threshold, loss)
    """
    concatenated_data, concatenated_markers, threshold, derivative_threshold = args
    loss = compute_loss_for_parameter_combination(
        concatenated_data,
        concatenated_markers,
        threshold,
        derivative_threshold,
    )
    return threshold, derivative_threshold, loss


def grid_search_threshold_algorithm(
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
        2D array of loss values [threshold, derivative_magnitude]
    """
    total = len(THRESHOLD_VALUES) * len(DERIVATIVE_MAGNITUDE_VALUES)
    
    if show_progress:
        counter, bar_length, report_interval = _init_progress(total)
    
    loss_grid = np.zeros((len(THRESHOLD_VALUES), len(DERIVATIVE_MAGNITUDE_VALUES)), dtype=float)
    
    # Prepare all parameter combinations for parallel processing
    parameter_combinations = [
        (concatenated_data, concatenated_markers, float(thresh), float(mag))
        for thresh in THRESHOLD_VALUES
        for mag in DERIVATIVE_MAGNITUDE_VALUES
    ]
    
    # Create index mapping for results
    index_map = {}
    for i, thresh in enumerate(THRESHOLD_VALUES):
        for j, mag in enumerate(DERIVATIVE_MAGNITUDE_VALUES):
            index_map[(float(thresh), float(mag))] = (i, j)
    
    # Process parameter combinations in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_combo = {
            executor.submit(compute_loss_for_parameter_combination_wrapper, combo): combo
            for combo in parameter_combinations
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_combo):
            threshold, derivative_threshold, loss = future.result()
            i, j = index_map[(threshold, derivative_threshold)]
            loss_grid[i, j] = loss
            
            if show_progress:
                counter += 1
                _update_progress(counter, total, bar_length, report_interval)
    
    if show_progress:
        _finalize_progress()
    return loss_grid


def plot_3d_loss(
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

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_zlabel('Loss (False Positives + False Negatives)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Loss')

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
        cmap=GRADIENT_CMAP,
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
        cmap=GRADIENT_CMAP,
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
    print("LOSS CALCULATION AND 3D VISUALIZATION (THRESHOLD ALGORITHM)")
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
    print(f"\nGrid searching {len(THRESHOLD_VALUES)} × {len(DERIVATIVE_MAGNITUDE_VALUES)} = {len(THRESHOLD_VALUES) * len(DERIVATIVE_MAGNITUDE_VALUES)} parameter combinations...")
    print("Using precise jump detection and ground truth annotations...")
    print("Parallelizing across parameter combinations...")
    
    combined_loss_grid = grid_search_threshold_algorithm(
        concatenated_data,
        concatenated_markers,
        show_progress=True,
        max_workers=None,  # Use all available CPU cores
    )
    
    # Transpose combined loss grid to swap axes
    combined_loss_grid_transposed = combined_loss_grid.T
    
    # Save static PNG version (3D surface plot only)
    plot_3d_loss(
        combined_loss_grid_transposed,
        THRESHOLD_VALUES,
        DERIVATIVE_MAGNITUDE_VALUES,
        "Threshold",
        "Derivative magnitude",
        "Combined Loss Function - Threshold Algorithm (Precise Detection with Ground Truth)",
        OUTPUT_DIR / "combined_loss_threshold.png",
        show_interactive=False,
    )
    
    # Display interactive version (rotatable)
    print("\nDisplaying interactive 3D plot (rotate to view)...")
    print("Close the plot window to continue.")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for 3D plotting (swapped axes)
    X, Y = np.meshgrid(THRESHOLD_VALUES, DERIVATIVE_MAGNITUDE_VALUES)
    
    # Plot surface (using transposed grid)
    surf = ax.plot_surface(
        X, Y, combined_loss_grid_transposed,
        cmap=GRADIENT_CMAP,
        alpha=0.9,
        linewidth=0,
        antialiased=True,
    )
    
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Derivative magnitude", fontsize=12)
    ax.set_zlabel('Loss (False Positives + False Negatives)', fontsize=12)
    ax.set_title("Combined Loss Function - Threshold Algorithm (Precise Detection with Ground Truth) - Interactive (Rotate to View)", 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Loss')
    
    # Set viewing angle
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()  # This makes it interactive and rotatable

    print(f"\nCompleted at: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
