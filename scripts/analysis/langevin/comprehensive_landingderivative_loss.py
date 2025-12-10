"""Comprehensive parameter optimization for landing derivative algorithm using precise loss function.

This script optimizes:
- Landing derivative threshold (landing_threshold) - positive threshold for landing detection (derivative > threshold)
- Center offset (center_offset) - frames to go left from landing
- Search window (search_window) - for precise boundary calculation
- In air threshold (in_air_threshold) - optional filtering
- Minimum and maximum flight time constraints

Uses precise jump boundaries (already computed in the algorithm) and precision loss function.
"""

from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import json
import random

import numpy as np

import sys
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from jump_detection.algorithms.landing_derivative import (
    LandingDerivativeParameters,
    _run_landing_derivative_pipeline,
)
from jump_detection.analysis.loss import compute_precision_loss
from jump_detection.annotations import load_annotations
from jump_detection.config import SAMPLING_RATE, DEFAULT_DATA_FILES
from jump_detection.data import load_dataset
from jump_detection.types import DetectionResult, LandingDerivativeNames

# Parameter ranges for grid search (if needed)
LANDING_THRESHOLD_VALUES = np.linspace(0, 100, 20)  # Landing derivative threshold: 0 to 100 (positive)
CENTER_OFFSET_VALUES = np.arange(5, 25, 2)  # Center offset: 5 to 24 frames (integers)
SEARCH_WINDOW_VALUES = np.arange(50, 100, 5)  # Search window: 50 to 95 frames (integers)
IN_AIR_THRESHOLD_VALUES = np.linspace(100, 400, 15)  # In air threshold: 100 to 400
MIN_FLIGHT_TIME_VALUES = np.linspace(0.1, 0.5, 10)  # Minimum flight time: 0.1 to 0.5 seconds
MAX_FLIGHT_TIME_VALUES = np.linspace(0.5, 1.5, 10)  # Maximum flight time: 0.5 to 1.5 seconds

# Get project root for path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Use DEFAULT_DATA_FILES from config, converting Path objects to strings
DATA_FILES = [str(f) for f in DEFAULT_DATA_FILES]

OUTPUT_DIR = PROJECT_ROOT / "results"

# Parameter bounds for Langevin sampling
PARAM_BOUNDS = {
    "landing_threshold": (0.0, 100.0),  # Positive threshold for landing (derivative > threshold)
    "center_offset": (5.0, 25.0),  # Will be converted to int
    "search_window": (50.0, 100.0),  # Will be converted to int
    "in_air_threshold": (100.0, 400.0),
    "min_flight_time": (0.1, 0.5),
    "max_flight_time": (0.5, 1.5),
}


def concatenate_all_data_and_annotations(
    data_files_with_annotations: list[tuple[Path, list[int]]],
) -> tuple[np.ndarray, list[int], list[int]]:
    """Concatenate all data files and annotations into single arrays.
    
    Args:
        data_files_with_annotations: List of (data_file_path, ground_truth_markers) tuples
        
    Returns:
        Tuple of (concatenated_raw_data, concatenated_markers, file_boundaries)
    """
    all_data = []
    all_markers = []
    file_boundaries = [0]
    
    current_offset = 0
    
    for data_file, ground_truth_markers in data_files_with_annotations:
        raw_data = load_dataset(data_file)
        all_data.append(raw_data)
        
        offset_markers = [m + current_offset for m in ground_truth_markers]
        all_markers.extend(offset_markers)
        
        current_offset += len(raw_data)
        file_boundaries.append(current_offset)
    
    concatenated_data = np.vstack(all_data)
    
    return concatenated_data, all_markers, file_boundaries


def compute_loss_for_parameter_combination(
    concatenated_data: np.ndarray,
    concatenated_markers: list[int],
    landing_threshold: float,
    center_offset: int,
    search_window: int,
    in_air_threshold: float | None,
    min_flight_time: float,
    max_flight_time: float,
) -> float:
    """Compute loss for concatenated data with given parameters.
    
    Args:
        concatenated_data: All sensor data concatenated vertically
        concatenated_markers: All annotation markers with offsets applied
        landing_threshold: Landing derivative threshold parameter (positive, derivative > threshold)
        center_offset: Center offset in frames (integer)
        search_window: Search window for precise boundaries (integer)
        in_air_threshold: In air threshold parameter (None to disable)
        min_flight_time: Minimum flight time (seconds)
        max_flight_time: Maximum flight time (seconds)
        
    Returns:
        Loss value (false_positives + false_negatives)
    """
    # Validate that min_flight_time < max_flight_time
    if min_flight_time >= max_flight_time:
        return float('inf')  # Invalid parameter combination
    
    # Validate integer parameters
    center_offset = int(round(center_offset))
    search_window = int(round(search_window))
    
    if center_offset < 1 or search_window < 1:
        return float('inf')
    
    params = LandingDerivativeParameters(
        landing_threshold=landing_threshold,
        center_offset=center_offset,
        search_window=search_window,
        in_air_threshold=in_air_threshold,
        min_flight_time=min_flight_time,
        max_flight_time=max_flight_time,
    )
    
    # Run landing derivative pipeline directly on concatenated data
    signals, jumps, metadata = _run_landing_derivative_pipeline(concatenated_data, params)
    
    # Create DetectionResult for compatibility
    result = DetectionResult(
        participant_name=None,
        sampling_rate=SAMPLING_RATE,
        raw_data=concatenated_data,
        pooled_data=signals[LandingDerivativeNames.POOLED.value],
        signals=signals,
        jumps=jumps,
        metadata=metadata,
    )
    
    # Jumps already have precise boundaries from the algorithm
    # Compute loss using ground truth annotations
    if not concatenated_markers:
        return float('inf')
    
    metrics = compute_precision_loss(jumps, concatenated_markers)
    return float(metrics["loss"])


def compute_loss_for_parameter_combination_wrapper(
    args: tuple[np.ndarray, list[int], float, int, int, float | None, float, float],
) -> tuple[float, int, int, float | None, float, float, float]:
    """Wrapper function for parallel processing of parameter combinations.
    
    Args:
        args: Tuple of (concatenated_data, concatenated_markers, landing_threshold,
                       center_offset, search_window, in_air_threshold, min_flight_time, max_flight_time)
        
    Returns:
        Tuple of (landing_threshold, center_offset, search_window, in_air_threshold,
                 min_flight_time, max_flight_time, loss)
    """
    concatenated_data, concatenated_markers, landing_threshold, center_offset, \
        search_window, in_air_threshold, min_flight_time, max_flight_time = args
    
    loss = compute_loss_for_parameter_combination(
        concatenated_data,
        concatenated_markers,
        landing_threshold,
        center_offset,
        search_window,
        in_air_threshold,
        min_flight_time,
        max_flight_time,
    )
    return landing_threshold, center_offset, search_window, in_air_threshold, min_flight_time, max_flight_time, loss


def comprehensive_grid_search(
    concatenated_data: np.ndarray,
    concatenated_markers: list[int],
    show_progress: bool = True,
    max_workers: int | None = None,
) -> list[dict]:
    """Run comprehensive parameter grid search and return all results sorted by loss.
    
    Args:
        concatenated_data: All sensor data concatenated vertically
        concatenated_markers: All annotation markers with offsets applied
        show_progress: Whether to show progress updates
        max_workers: Maximum number of parallel workers (None = use CPU count)
        
    Returns:
        List of dictionaries with parameter combinations and their losses, sorted by loss
    """
    # Generate all parameter combinations
    all_combinations = list(product(
        LANDING_THRESHOLD_VALUES,
        CENTER_OFFSET_VALUES,
        SEARCH_WINDOW_VALUES,
        IN_AIR_THRESHOLD_VALUES,
        MIN_FLIGHT_TIME_VALUES,
        MAX_FLIGHT_TIME_VALUES,
    ))
    
    # Filter out invalid combinations (min_flight_time >= max_flight_time)
    valid_combinations = [
        (landing, center, search, in_air, min_time, max_time)
        for landing, center, search, in_air, min_time, max_time in all_combinations
        if min_time < max_time
    ]
    
    total = len(valid_combinations)
    print(f"Total parameter combinations to test: {total:,}")
    print(f"  Landing threshold: {len(LANDING_THRESHOLD_VALUES)} values")
    print(f"  Center offset: {len(CENTER_OFFSET_VALUES)} values")
    print(f"  Search window: {len(SEARCH_WINDOW_VALUES)} values")
    print(f"  In air threshold: {len(IN_AIR_THRESHOLD_VALUES)} values")
    print(f"  Min flight time: {len(MIN_FLIGHT_TIME_VALUES)} values")
    print(f"  Max flight time: {len(MAX_FLIGHT_TIME_VALUES)} values")
    
    results = []
    completed = 0
    
    # Prepare all parameter combinations for parallel processing
    parameter_combinations = [
        (concatenated_data, concatenated_markers, float(landing), int(center),
         int(search), float(in_air), float(min_time), float(max_time))
        for landing, center, search, in_air, min_time, max_time in valid_combinations
    ]
    
    # Process parameter combinations in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_combo = {
            executor.submit(compute_loss_for_parameter_combination_wrapper, combo): combo
            for combo in parameter_combinations
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_combo):
            try:
                landing_threshold, center_offset, search_window, in_air_threshold, \
                    min_flight_time, max_flight_time, loss = future.result()
                
                results.append({
                    "landing_threshold": float(landing_threshold),
                    "center_offset": int(center_offset),
                    "search_window": int(search_window),
                    "in_air_threshold": float(in_air_threshold) if in_air_threshold is not None else None,
                    "min_flight_time": float(min_flight_time),
                    "max_flight_time": float(max_flight_time),
                    "loss": float(loss),
                })
                
                completed += 1
                if show_progress and completed % max(1, total // 100) == 0:
                    progress = completed / total * 100
                    print(f"Progress: {completed:,}/{total:,} ({progress:.1f}%)", end='\r')
            except Exception as e:
                print(f"\nError processing combination: {e}")
                continue
    
    if show_progress:
        print(f"\nCompleted: {completed:,}/{total:,} combinations")
    
    # Sort by loss (ascending)
    results.sort(key=lambda x: x["loss"])
    
    return results


def print_optimized_results(results: list[dict], top_n: int = 20) -> None:
    """Print the top N optimized results.
    
    Args:
        results: List of result dictionaries sorted by loss
        top_n: Number of top results to print
    """
    print(f"\n{'='*120}")
    print(f"TOP {min(top_n, len(results))} OPTIMIZED RESULTS")
    print(f"{'='*120}")
    print(f"{'Rank':<6} {'Landing':<10} {'Center':<10} {'Search':<10} {'In Air':<10} {'Min Time':<10} {'Max Time':<10} {'Loss':<10}")
    print(f"{'-'*120}")
    
    for i, result in enumerate(results[:top_n], 1):
        in_air_str = f"{result['in_air_threshold']:.4f}" if result['in_air_threshold'] is not None else "None"
        print(
            f"{i:<6} "
            f"{result['landing_threshold']:<10.4f} "
            f"{result['center_offset']:<10} "
            f"{result['search_window']:<10} "
            f"{in_air_str:<10} "
            f"{result['min_flight_time']:<10.4f} "
            f"{result['max_flight_time']:<10.4f} "
            f"{result['loss']:<10.4f}"
        )
    
    print(f"{'='*120}")
    
    if results:
        best = results[0]
        in_air_str = f"{best['in_air_threshold']:.6f}" if best['in_air_threshold'] is not None else "None"
        print(f"\nBEST RESULT:")
        print(f"  Landing threshold:   {best['landing_threshold']:.6f} (derivative > threshold)")
        print(f"  Center offset:       {best['center_offset']}")
        print(f"  Search window:       {best['search_window']}")
        print(f"  In air threshold:    {in_air_str}")
        print(f"  Min flight time:     {best['min_flight_time']:.6f} seconds")
        print(f"  Max flight time:      {best['max_flight_time']:.6f} seconds")
        print(f"  Loss:                 {best['loss']:.6f}")
        print(f"{'='*120}\n")


def random_initial_parameters() -> dict[str, float]:
    """Generate random initial parameters within bounds.
    
    Returns:
        Dictionary with parameter values
    """
    return {
        "landing_threshold": random.uniform(*PARAM_BOUNDS["landing_threshold"]),
        "center_offset": random.uniform(*PARAM_BOUNDS["center_offset"]),
        "search_window": random.uniform(*PARAM_BOUNDS["search_window"]),
        "in_air_threshold": random.uniform(*PARAM_BOUNDS["in_air_threshold"]),
        "min_flight_time": random.uniform(*PARAM_BOUNDS["min_flight_time"]),
        "max_flight_time": random.uniform(*PARAM_BOUNDS["max_flight_time"]),
    }


def perturb_parameters(
    params: dict[str, float],
    param_to_perturb: str,
    step_sizes: dict[str, float] | None = None,
) -> dict[str, float]:
    """Perturb a single parameter using Langevin-style noise.
    
    Args:
        params: Current parameter dictionary
        param_to_perturb: Name of the parameter to perturb
        step_sizes: Dictionary of step sizes for each parameter (default: 5% of range)
        
    Returns:
        New parameter dictionary with perturbations
    """
    if step_sizes is None:
        # Default step sizes: 5% of parameter range
        step_sizes = {
            "landing_threshold": (PARAM_BOUNDS["landing_threshold"][1] - PARAM_BOUNDS["landing_threshold"][0]) * 0.05,
            "center_offset": (PARAM_BOUNDS["center_offset"][1] - PARAM_BOUNDS["center_offset"][0]) * 0.05,
            "search_window": (PARAM_BOUNDS["search_window"][1] - PARAM_BOUNDS["search_window"][0]) * 0.05,
            "in_air_threshold": (PARAM_BOUNDS["in_air_threshold"][1] - PARAM_BOUNDS["in_air_threshold"][0]) * 0.05,
            "min_flight_time": (PARAM_BOUNDS["min_flight_time"][1] - PARAM_BOUNDS["min_flight_time"][0]) * 0.05,
            "max_flight_time": (PARAM_BOUNDS["max_flight_time"][1] - PARAM_BOUNDS["max_flight_time"][0]) * 0.05,
        }
    
    new_params = params.copy()
    
    # Perturb only the specified parameter
    if param_to_perturb in new_params:
        # Add Gaussian noise scaled by step size
        noise = np.random.normal(0, step_sizes[param_to_perturb])
        new_value = new_params[param_to_perturb] + noise
        
        # Clip to bounds
        min_val, max_val = PARAM_BOUNDS[param_to_perturb]
        new_params[param_to_perturb] = np.clip(new_value, min_val, max_val)
    
    # Ensure min_flight_time < max_flight_time
    if new_params["min_flight_time"] >= new_params["max_flight_time"]:
        # Adjust to maintain constraint
        mid_point = (new_params["min_flight_time"] + new_params["max_flight_time"]) / 2
        new_params["min_flight_time"] = mid_point - 0.05
        new_params["max_flight_time"] = mid_point + 0.05
        # Clip again to ensure within bounds
        new_params["min_flight_time"] = np.clip(
            new_params["min_flight_time"],
            *PARAM_BOUNDS["min_flight_time"]
        )
        new_params["max_flight_time"] = np.clip(
            new_params["max_flight_time"],
            *PARAM_BOUNDS["max_flight_time"]
        )
    
    return new_params


def langevin_sampling(
    concatenated_data: np.ndarray,
    concatenated_markers: list[int],
    n_iterations: int = 1000,
    initial_params: dict[str, float] | None = None,
    temperature: float = 1.0,
    temperature_decay: float = 0.9995,
    step_sizes: dict[str, float] | None = None,
    show_progress: bool = True,
) -> tuple[list[dict], dict]:
    """Langevin-inspired sampling strategy for parameter optimization.
    
    Uses Metropolis-Hastings acceptance criterion with temperature schedule.
    Parameters are perturbed sequentially, cycling through all parameters.
    
    Args:
        concatenated_data: All sensor data concatenated vertically
        concatenated_markers: All annotation markers with offsets applied
        n_iterations: Number of sampling iterations
        initial_params: Starting parameters (if None, random initialization)
        temperature: Initial temperature for acceptance probability
        temperature_decay: Temperature decay factor per iteration
        step_sizes: Step sizes for each parameter (default: 5% of range)
        show_progress: Whether to show progress updates
        
    Returns:
        Tuple of (all_accepted_results, best_result_dict)
    """
    # Initialize parameters
    if initial_params is None:
        current_params = random_initial_parameters()
    else:
        current_params = initial_params.copy()
    
    # Compute initial loss
    current_loss = compute_loss_for_parameter_combination(
        concatenated_data,
        concatenated_markers,
        current_params["landing_threshold"],
        int(round(current_params["center_offset"])),
        int(round(current_params["search_window"])),
        current_params["in_air_threshold"],
        current_params["min_flight_time"],
        current_params["max_flight_time"],
    )
    
    best_params = current_params.copy()
    best_loss = current_loss
    
    all_results = [{
        "landing_threshold": float(current_params["landing_threshold"]),
        "center_offset": int(round(current_params["center_offset"])),
        "search_window": int(round(current_params["search_window"])),
        "in_air_threshold": float(current_params["in_air_threshold"]) if current_params["in_air_threshold"] is not None else None,
        "min_flight_time": float(current_params["min_flight_time"]),
        "max_flight_time": float(current_params["max_flight_time"]),
        "loss": float(current_loss),
        "accepted": True,
        "iteration": 0,
        "temperature": float(temperature),
    }]
    
    current_temp = temperature
    accepted_count = 1
    
    if show_progress:
        print(f"Starting Langevin sampling with {n_iterations} iterations")
        print(f"Initial parameters: {current_params}")
        print(f"Initial loss: {current_loss:.4f}")
        print(f"Initial temperature: {temperature:.4f}")
        print()
    
    # Get parameter names for sequential cycling
    param_names = list(current_params.keys())
    
    for iteration in range(1, n_iterations + 1):
        # Select parameter to perturb sequentially (cycle through all parameters)
        param_to_perturb = param_names[iteration % len(param_names)]
        
        # Perturb the selected parameter
        proposed_params = perturb_parameters(
            current_params,
            param_to_perturb=param_to_perturb,
            step_sizes=step_sizes,
        )
        
        # Compute loss for proposed parameters
        proposed_loss = compute_loss_for_parameter_combination(
            concatenated_data,
            concatenated_markers,
            proposed_params["landing_threshold"],
            int(round(proposed_params["center_offset"])),
            int(round(proposed_params["search_window"])),
            proposed_params["in_air_threshold"],
            proposed_params["min_flight_time"],
            proposed_params["max_flight_time"],
        )
        
        # Metropolis-Hastings acceptance criterion
        # Accept if loss decreases, or with probability exp(-(new_loss - old_loss) / temperature)
        loss_diff = proposed_loss - current_loss
        accept = False
        
        if loss_diff <= 0:
            # Always accept if loss decreases
            accept = True
        elif current_temp > 0:
            # Accept with probability based on temperature
            accept_prob = np.exp(-loss_diff / current_temp)
            accept = random.random() < accept_prob
        
        if accept:
            current_params = proposed_params
            current_loss = proposed_loss
            accepted_count += 1
            
            # Update best if this is better
            if proposed_loss < best_loss:
                best_params = proposed_params.copy()
                best_loss = proposed_loss
        
        # Record result (accepted or rejected) - convert to native Python types
        all_results.append({
            "landing_threshold": float(proposed_params["landing_threshold"]),
            "center_offset": int(round(proposed_params["center_offset"])),
            "search_window": int(round(proposed_params["search_window"])),
            "in_air_threshold": float(proposed_params["in_air_threshold"]) if proposed_params["in_air_threshold"] is not None else None,
            "min_flight_time": float(proposed_params["min_flight_time"]),
            "max_flight_time": float(proposed_params["max_flight_time"]),
            "loss": float(proposed_loss),
            "accepted": bool(accept),
            "iteration": int(iteration),
            "temperature": float(current_temp),
        })
        
        # Decay temperature
        current_temp *= temperature_decay
        
        # Progress update
        if show_progress and (iteration % max(1, n_iterations // 100) == 0 or iteration == n_iterations):
            progress = iteration / n_iterations * 100
            acceptance_rate = accepted_count / iteration * 100
            print(
                f"Iteration {iteration:5d}/{n_iterations} | "
                f"Loss: {current_loss:7.4f} | "
                f"Best: {best_loss:7.4f} | "
                f"Temp: {current_temp:.4f} | "
                f"Accept: {acceptance_rate:5.1f}% | "
                f"Progress: {progress:5.1f}%",
                end='\r'
            )
    
    if show_progress:
        print()  # New line after progress
    
    best_result = {
        "landing_threshold": float(best_params["landing_threshold"]),
        "center_offset": int(round(best_params["center_offset"])),
        "search_window": int(round(best_params["search_window"])),
        "in_air_threshold": float(best_params["in_air_threshold"]) if best_params["in_air_threshold"] is not None else None,
        "min_flight_time": float(best_params["min_flight_time"]),
        "max_flight_time": float(best_params["max_flight_time"]),
        "loss": float(best_loss),
        "total_iterations": int(n_iterations),
        "accepted_count": int(accepted_count),
        "acceptance_rate": float(accepted_count / n_iterations),
    }
    
    return all_results, best_result


def convert_to_json_serializable(obj):
    """Recursively convert numpy types and other non-serializable types to JSON-compatible types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    # Check for numpy scalars using base classes (compatible with NumPy 1.x and 2.x)
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Try to convert numpy scalar types by checking type name
    elif type(obj).__module__ == 'numpy':
        # Try converting to Python native types
        try:
            if np.issubdtype(type(obj), np.integer):
                return int(obj)
            elif np.issubdtype(type(obj), np.floating):
                return float(obj)
            elif np.issubdtype(type(obj), np.bool_):
                return bool(obj)
        except (TypeError, AttributeError):
            pass
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    
    return obj


def save_results_to_json(results: list[dict], output_path: Path) -> None:
    """Save all results to a JSON file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "total_combinations": len(results),
        "parameter_ranges": {
            "landing_threshold": {
                "min": float(np.min(LANDING_THRESHOLD_VALUES)),
                "max": float(np.max(LANDING_THRESHOLD_VALUES)),
                "count": len(LANDING_THRESHOLD_VALUES),
            },
            "center_offset": {
                "min": int(np.min(CENTER_OFFSET_VALUES)),
                "max": int(np.max(CENTER_OFFSET_VALUES)),
                "count": len(CENTER_OFFSET_VALUES),
            },
            "search_window": {
                "min": int(np.min(SEARCH_WINDOW_VALUES)),
                "max": int(np.max(SEARCH_WINDOW_VALUES)),
                "count": len(SEARCH_WINDOW_VALUES),
            },
            "in_air_threshold": {
                "min": float(np.min(IN_AIR_THRESHOLD_VALUES)),
                "max": float(np.max(IN_AIR_THRESHOLD_VALUES)),
                "count": len(IN_AIR_THRESHOLD_VALUES),
            },
            "min_flight_time": {
                "min": float(np.min(MIN_FLIGHT_TIME_VALUES)),
                "max": float(np.max(MIN_FLIGHT_TIME_VALUES)),
                "count": len(MIN_FLIGHT_TIME_VALUES),
            },
            "max_flight_time": {
                "min": float(np.min(MAX_FLIGHT_TIME_VALUES)),
                "max": float(np.max(MAX_FLIGHT_TIME_VALUES)),
                "count": len(MAX_FLIGHT_TIME_VALUES),
            },
        },
        "results": results,
    }
    
    # Convert all numpy types to native Python types
    output_data = convert_to_json_serializable(output_data)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved results to {output_path}")


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive parameter optimization for landing derivative algorithm"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["grid", "langevin", "both"],
        default="langevin",
        help="Optimization method: grid search, Langevin sampling, or both"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of iterations for Langevin sampling (default: 10000)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=10.0,
        help="Initial temperature for Langevin sampling (default: 10.0)"
    )
    parser.add_argument(
        "--temp-decay",
        type=float,
        default=0.9995,
        help="Temperature decay factor per iteration (default: 0.9995)"
    )
    parser.add_argument(
        "--initial-params",
        type=str,
        default=None,
        help="JSON file with initial parameters for Langevin sampling"
    )
    
    args = parser.parse_args()
    
    print("COMPREHENSIVE PARAMETER OPTIMIZATION (LANDING DERIVATIVE ALGORITHM)")
    print("Using precise jump boundaries and precision loss function")
    print(f"Method: {args.method}")
    print(f"Started at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    
    # Load all data files and their annotations upfront
    print(f"\nLoading data files and annotations for {len(DATA_FILES)} participants...")
    data_files_with_annotations = []
    participant_names = []
    
    for data_file in DATA_FILES:
        # Resolve path relative to project root
        file_path = Path(data_file)
        if not file_path.is_absolute():
            file_path = PROJECT_ROOT / file_path
        file_path = file_path.resolve()
        
        participant = file_path.parent.name if file_path.parent != file_path else "Dataset"
        file_label = file_path.stem
        participant_names.append(participant)
        
        # Load ground truth annotations - REQUIRED
        annotations = load_annotations(file_path)
        
        if annotations is None:
            print(f"  ⚠ Skipping {participant} – {file_label}: No annotation file found")
            continue
        
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
    
    # Load initial parameters if provided
    initial_params = None
    if args.initial_params:
        with open(args.initial_params, 'r') as f:
            initial_params = json.load(f)
        print(f"\nLoaded initial parameters from {args.initial_params}")
    
    all_results = []
    
    # Run grid search if requested
    if args.method in ["grid", "both"]:
        print("\n" + "="*100)
        print("Starting comprehensive parameter grid search...")
        print("="*100)
        
        grid_results = comprehensive_grid_search(
            concatenated_data,
            concatenated_markers,
            show_progress=True,
            max_workers=None,  # Use all available CPU cores
        )
        
        all_results.extend(grid_results)
        
        # Print optimized results
        print_optimized_results(grid_results, top_n=50)
        
        # Save grid search results
        output_path = OUTPUT_DIR / "comprehensive_landingderivative_loss_grid_results.json"
        save_results_to_json(grid_results, output_path)
    
    # Run Langevin sampling if requested
    if args.method in ["langevin", "both"]:
        print("\n" + "="*100)
        print("Starting Langevin sampling optimization...")
        print("="*100)
        
        langevin_all, langevin_best = langevin_sampling(
            concatenated_data,
            concatenated_markers,
            n_iterations=args.iterations,
            initial_params=initial_params,
            temperature=args.temperature,
            temperature_decay=args.temp_decay,
            show_progress=True,
        )
        
        # Extract only accepted results for comparison
        accepted_results = [
            {k: v for k, v in r.items() if k != "accepted" and k != "iteration" and k != "temperature"}
            for r in langevin_all
            if r.get("accepted", False)
        ]
        
        # Sort by loss
        accepted_results.sort(key=lambda x: x["loss"])
        all_results.extend(accepted_results)
        
        # Print Langevin results
        print(f"\n{'='*100}")
        print("LANGEVIN SAMPLING RESULTS")
        print(f"{'='*100}")
        print(f"Total iterations: {args.iterations}")
        print(f"Accepted samples: {len(accepted_results)}")
        print(f"Acceptance rate: {len(accepted_results) / args.iterations * 100:.2f}%")
        print(f"\nBEST RESULT FROM LANGEVIN SAMPLING:")
        in_air_str = f"{langevin_best['in_air_threshold']:.6f}" if langevin_best['in_air_threshold'] is not None else "None"
        print(f"  Landing threshold:   {langevin_best['landing_threshold']:.6f} (derivative > threshold)")
        print(f"  Center offset:       {langevin_best['center_offset']}")
        print(f"  Search window:       {langevin_best['search_window']}")
        print(f"  In air threshold:    {in_air_str}")
        print(f"  Min flight time:     {langevin_best['min_flight_time']:.6f} seconds")
        print(f"  Max flight time:      {langevin_best['max_flight_time']:.6f} seconds")
        print(f"  Loss:                 {langevin_best['loss']:.6f}")
        print(f"{'='*100}")
        
        # Print top accepted results
        print(f"\nTOP {min(20, len(accepted_results))} ACCEPTED RESULTS:")
        print_optimized_results(accepted_results, top_n=20)
        
        # Save Langevin results
        langevin_output = OUTPUT_DIR / "comprehensive_landingderivative_loss_langevin_results.json"
        langevin_output.parent.mkdir(parents=True, exist_ok=True)
        
        langevin_output_data = {
            "timestamp": datetime.now().isoformat(),
            "method": "langevin_sampling",
            "n_iterations": int(args.iterations),
            "initial_temperature": float(args.temperature),
            "temperature_decay": float(args.temp_decay),
            "initial_params": convert_to_json_serializable(initial_params) if initial_params else None,
            "best_result": convert_to_json_serializable(langevin_best),
            "all_iterations": convert_to_json_serializable(langevin_all),
            "accepted_results": convert_to_json_serializable(accepted_results),
        }
        
        with open(langevin_output, 'w') as f:
            json.dump(langevin_output_data, f, indent=2)
        
        print(f"Saved Langevin sampling results to {langevin_output}")
    
    # If both methods were run, combine and show overall best
    if args.method == "both" and all_results:
        print(f"\n{'='*100}")
        print("COMBINED RESULTS (Grid Search + Langevin Sampling)")
        print(f"{'='*100}")
        all_results.sort(key=lambda x: x["loss"])
        print_optimized_results(all_results, top_n=20)
        
        combined_output = OUTPUT_DIR / "comprehensive_landingderivative_loss_combined_results.json"
        save_results_to_json(all_results, combined_output)
    
    print(f"\nCompleted at: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()

