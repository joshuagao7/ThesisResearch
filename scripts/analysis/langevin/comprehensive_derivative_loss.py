"""Comprehensive 5-parameter grid search for derivative algorithm using precise loss function.

This script sweeps through:
- High derivative threshold (upper_threshold)
- Low derivative threshold (lower_threshold)
- In air threshold (in_air_threshold)
- Minimum physics based constraint (min_flight_time)
- Maximum physics based constraint (max_flight_time)

Uses precise jump boundaries and precision loss function.
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

# Parameter ranges for grid search
UPPER_THRESHOLD_VALUES = np.linspace(0, 100, 20)  # High derivative threshold: 0 to 100
LOWER_THRESHOLD_VALUES = np.linspace(-100, 0, 20)  # Low derivative threshold: -100 to 0
IN_AIR_THRESHOLD_VALUES = np.linspace(100, 400, 15)  # In air threshold: 100 to 400
MIN_FLIGHT_TIME_VALUES = np.linspace(0.1, 0.5, 10)  # Minimum flight time: 0.1 to 0.5 seconds
MAX_FLIGHT_TIME_VALUES = np.linspace(0.5, 1.5, 10)  # Maximum flight time: 0.5 to 1.5 seconds

# Get project root for path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Use DEFAULT_DATA_FILES from config, converting Path objects to strings
DATA_FILES = [str(f) for f in DEFAULT_DATA_FILES]

OUTPUT_DIR = PROJECT_ROOT / "results"
SEARCH_WINDOW = 70  # For precise jump detection

# Parameter bounds for Langevin sampling
PARAM_BOUNDS = {
    "upper_threshold": (0.0, 100.0),
    "lower_threshold": (-100.0, 0.0),
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
        precise_data = calculate_precise_jump_boundaries(
            signal, jump.center, SEARCH_WINDOW
        )
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


def compute_loss_for_parameter_combination(
    concatenated_data: np.ndarray,
    concatenated_markers: list[int],
    upper_threshold: float,
    lower_threshold: float,
    in_air_threshold: float,
    min_flight_time: float,
    max_flight_time: float,
) -> float:
    """Compute loss for concatenated data with given parameters.
    
    Args:
        concatenated_data: All sensor data concatenated vertically
        concatenated_markers: All annotation markers with offsets applied
        upper_threshold: Upper derivative threshold parameter
        lower_threshold: Lower derivative threshold parameter
        in_air_threshold: In air threshold parameter
        min_flight_time: Minimum flight time (seconds)
        max_flight_time: Maximum flight time (seconds)
        
    Returns:
        Loss value (false_positives + false_negatives)
    """
    # Validate that min_flight_time < max_flight_time
    if min_flight_time >= max_flight_time:
        return float('inf')  # Invalid parameter combination
    
    params = DerivativeParameters(
        upper_threshold=upper_threshold,
        lower_threshold=lower_threshold,
        in_air_threshold=in_air_threshold,
        min_flight_time=min_flight_time,
        max_flight_time=max_flight_time,
    )
    
    # Run derivative pipeline directly on concatenated data
    signals, jumps, metadata = _run_derivative_pipeline(concatenated_data, params)
    
    # Create DetectionResult for compatibility with precise jump calculation
    result = DetectionResult(
        participant_name=None,
        sampling_rate=SAMPLING_RATE,
        raw_data=concatenated_data,
        pooled_data=signals[DerivativeNames.POOLED.value],
        signals=signals,
        jumps=jumps,
        metadata=metadata,
    )
    
    # Calculate precise jumps with refined boundaries
    precise_jumps = compute_precise_jumps_from_concatenated(concatenated_data, result)
    
    # Compute loss using ground truth annotations
    if not concatenated_markers:
        return float('inf')
    
    metrics = compute_precision_loss(precise_jumps, concatenated_markers)
    return float(metrics["loss"])


def compute_loss_for_parameter_combination_wrapper(
    args: tuple[np.ndarray, list[int], float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Wrapper function for parallel processing of parameter combinations.
    
    Args:
        args: Tuple of (concatenated_data, concatenated_markers, upper_threshold, 
                       lower_threshold, in_air_threshold, min_flight_time, max_flight_time)
        
    Returns:
        Tuple of (upper_threshold, lower_threshold, in_air_threshold, 
                 min_flight_time, max_flight_time, loss)
    """
    concatenated_data, concatenated_markers, upper_threshold, lower_threshold, \
        in_air_threshold, min_flight_time, max_flight_time = args
    
    loss = compute_loss_for_parameter_combination(
        concatenated_data,
        concatenated_markers,
        upper_threshold,
        lower_threshold,
        in_air_threshold,
        min_flight_time,
        max_flight_time,
    )
    return upper_threshold, lower_threshold, in_air_threshold, min_flight_time, max_flight_time, loss


def comprehensive_grid_search(
    concatenated_data: np.ndarray,
    concatenated_markers: list[int],
    show_progress: bool = True,
    max_workers: int | None = None,
) -> list[dict]:
    """Run comprehensive 5-parameter grid search and return all results sorted by loss.
    
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
        UPPER_THRESHOLD_VALUES,
        LOWER_THRESHOLD_VALUES,
        IN_AIR_THRESHOLD_VALUES,
        MIN_FLIGHT_TIME_VALUES,
        MAX_FLIGHT_TIME_VALUES,
    ))
    
    # Filter out invalid combinations (min_flight_time >= max_flight_time)
    valid_combinations = [
        (upper, lower, in_air, min_time, max_time)
        for upper, lower, in_air, min_time, max_time in all_combinations
        if min_time < max_time
    ]
    
    total = len(valid_combinations)
    print(f"Total parameter combinations to test: {total:,}")
    print(f"  Upper threshold: {len(UPPER_THRESHOLD_VALUES)} values")
    print(f"  Lower threshold: {len(LOWER_THRESHOLD_VALUES)} values")
    print(f"  In air threshold: {len(IN_AIR_THRESHOLD_VALUES)} values")
    print(f"  Min flight time: {len(MIN_FLIGHT_TIME_VALUES)} values")
    print(f"  Max flight time: {len(MAX_FLIGHT_TIME_VALUES)} values")
    
    results = []
    completed = 0
    
    # Prepare all parameter combinations for parallel processing
    parameter_combinations = [
        (concatenated_data, concatenated_markers, float(upper), float(lower),
         float(in_air), float(min_time), float(max_time))
        for upper, lower, in_air, min_time, max_time in valid_combinations
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
                upper_threshold, lower_threshold, in_air_threshold, \
                    min_flight_time, max_flight_time, loss = future.result()
                
                results.append({
                    "upper_threshold": float(upper_threshold),
                    "lower_threshold": float(lower_threshold),
                    "in_air_threshold": float(in_air_threshold),
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
    print(f"\n{'='*100}")
    print(f"TOP {min(top_n, len(results))} OPTIMIZED RESULTS")
    print(f"{'='*100}")
    print(f"{'Rank':<6} {'Upper':<10} {'Lower':<10} {'In Air':<10} {'Min Time':<10} {'Max Time':<10} {'Loss':<10}")
    print(f"{'-'*100}")
    
    for i, result in enumerate(results[:top_n], 1):
        print(
            f"{i:<6} "
            f"{result['upper_threshold']:<10.4f} "
            f"{result['lower_threshold']:<10.4f} "
            f"{result['in_air_threshold']:<10.4f} "
            f"{result['min_flight_time']:<10.4f} "
            f"{result['max_flight_time']:<10.4f} "
            f"{result['loss']:<10.4f}"
        )
    
    print(f"{'='*100}")
    
    if results:
        best = results[0]
        print(f"\nBEST RESULT:")
        print(f"  Upper threshold:     {best['upper_threshold']:.6f}")
        print(f"  Lower threshold:     {best['lower_threshold']:.6f}")
        print(f"  In air threshold:    {best['in_air_threshold']:.6f}")
        print(f"  Min flight time:     {best['min_flight_time']:.6f} seconds")
        print(f"  Max flight time:      {best['max_flight_time']:.6f} seconds")
        print(f"  Loss:                 {best['loss']:.6f}")
        print(f"{'='*100}\n")


def random_initial_parameters() -> dict[str, float]:
    """Generate random initial parameters within bounds.
    
    Returns:
        Dictionary with parameter values
    """
    return {
        "upper_threshold": random.uniform(*PARAM_BOUNDS["upper_threshold"]),
        "lower_threshold": random.uniform(*PARAM_BOUNDS["lower_threshold"]),
        "in_air_threshold": random.uniform(*PARAM_BOUNDS["in_air_threshold"]),
        "min_flight_time": random.uniform(*PARAM_BOUNDS["min_flight_time"]),
        "max_flight_time": random.uniform(*PARAM_BOUNDS["max_flight_time"]),
    }


def perturb_parameters(
    params: dict[str, float],
    step_sizes: dict[str, float] | None = None,
    perturbation_prob: float = 0.3,
) -> dict[str, float]:
    """Randomly perturb parameters using Langevin-style noise.
    
    Args:
        params: Current parameter dictionary
        step_sizes: Dictionary of step sizes for each parameter (default: 5% of range)
        perturbation_prob: Probability of perturbing each parameter independently
        
    Returns:
        New parameter dictionary with perturbations
    """
    if step_sizes is None:
        # Default step sizes: 5% of parameter range
        step_sizes = {
            "upper_threshold": (PARAM_BOUNDS["upper_threshold"][1] - PARAM_BOUNDS["upper_threshold"][0]) * 0.05,
            "lower_threshold": (PARAM_BOUNDS["lower_threshold"][1] - PARAM_BOUNDS["lower_threshold"][0]) * 0.05,
            "in_air_threshold": (PARAM_BOUNDS["in_air_threshold"][1] - PARAM_BOUNDS["in_air_threshold"][0]) * 0.05,
            "min_flight_time": (PARAM_BOUNDS["min_flight_time"][1] - PARAM_BOUNDS["min_flight_time"][0]) * 0.05,
            "max_flight_time": (PARAM_BOUNDS["max_flight_time"][1] - PARAM_BOUNDS["max_flight_time"][0]) * 0.05,
        }
    
    new_params = params.copy()
    
    # Perturb each parameter independently with given probability
    for param_name in new_params:
        if random.random() < perturbation_prob:
            # Add Gaussian noise scaled by step size
            noise = np.random.normal(0, step_sizes[param_name])
            new_value = new_params[param_name] + noise
            
            # Clip to bounds
            min_val, max_val = PARAM_BOUNDS[param_name]
            new_params[param_name] = np.clip(new_value, min_val, max_val)
    
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
    perturbation_prob: float = 0.3,
    show_progress: bool = True,
) -> tuple[list[dict], dict]:
    """Langevin-inspired sampling strategy for parameter optimization.
    
    Uses Metropolis-Hastings acceptance criterion with temperature schedule.
    
    Args:
        concatenated_data: All sensor data concatenated vertically
        concatenated_markers: All annotation markers with offsets applied
        n_iterations: Number of sampling iterations
        initial_params: Starting parameters (if None, random initialization)
        temperature: Initial temperature for acceptance probability
        temperature_decay: Temperature decay factor per iteration
        step_sizes: Step sizes for each parameter (default: 5% of range)
        perturbation_prob: Probability of perturbing each parameter
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
        current_params["upper_threshold"],
        current_params["lower_threshold"],
        current_params["in_air_threshold"],
        current_params["min_flight_time"],
        current_params["max_flight_time"],
    )
    
    best_params = current_params.copy()
    best_loss = current_loss
    
    all_results = [{
        "upper_threshold": float(current_params["upper_threshold"]),
        "lower_threshold": float(current_params["lower_threshold"]),
        "in_air_threshold": float(current_params["in_air_threshold"]),
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
    
    for iteration in range(1, n_iterations + 1):
        # Perturb parameters
        proposed_params = perturb_parameters(
            current_params,
            step_sizes=step_sizes,
            perturbation_prob=perturbation_prob,
        )
        
        # Compute loss for proposed parameters
        proposed_loss = compute_loss_for_parameter_combination(
            concatenated_data,
            concatenated_markers,
            proposed_params["upper_threshold"],
            proposed_params["lower_threshold"],
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
            "upper_threshold": float(proposed_params["upper_threshold"]),
            "lower_threshold": float(proposed_params["lower_threshold"]),
            "in_air_threshold": float(proposed_params["in_air_threshold"]),
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
        "upper_threshold": float(best_params["upper_threshold"]),
        "lower_threshold": float(best_params["lower_threshold"]),
        "in_air_threshold": float(best_params["in_air_threshold"]),
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
            "upper_threshold": {
                "min": float(np.min(UPPER_THRESHOLD_VALUES)),
                "max": float(np.max(UPPER_THRESHOLD_VALUES)),
                "count": len(UPPER_THRESHOLD_VALUES),
            },
            "lower_threshold": {
                "min": float(np.min(LOWER_THRESHOLD_VALUES)),
                "max": float(np.max(LOWER_THRESHOLD_VALUES)),
                "count": len(LOWER_THRESHOLD_VALUES),
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
        description="Comprehensive parameter optimization for derivative algorithm"
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
    
    print("COMPREHENSIVE 5-PARAMETER OPTIMIZATION (DERIVATIVE ALGORITHM)")
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
        print("Starting comprehensive 5-parameter grid search...")
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
        output_path = OUTPUT_DIR / "comprehensive_loss_grid_results.json"
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
        print(f"  Upper threshold:     {langevin_best['upper_threshold']:.6f}")
        print(f"  Lower threshold:     {langevin_best['lower_threshold']:.6f}")
        print(f"  In air threshold:    {langevin_best['in_air_threshold']:.6f}")
        print(f"  Min flight time:     {langevin_best['min_flight_time']:.6f} seconds")
        print(f"  Max flight time:      {langevin_best['max_flight_time']:.6f} seconds")
        print(f"  Loss:                 {langevin_best['loss']:.6f}")
        print(f"{'='*100}")
        
        # Print top accepted results
        print(f"\nTOP {min(20, len(accepted_results))} ACCEPTED RESULTS:")
        print_optimized_results(accepted_results, top_n=20)
        
        # Save Langevin results
        langevin_output = OUTPUT_DIR / "comprehensive_loss_langevin_results.json"
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
        
        combined_output = OUTPUT_DIR / "comprehensive_loss_combined_results.json"
        save_results_to_json(all_results, combined_output)
    
    print(f"\nCompleted at: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()

