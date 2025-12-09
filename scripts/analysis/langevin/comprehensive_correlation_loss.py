"""Comprehensive 5-parameter Langevin sampling for correlation algorithm using precise loss function.

This script optimizes:
- Buffer size (buffer_size)
- Negative frames (negative_frames)
- Zero frames (zero_frames)
- Positive frames (positive_frames)
- Correlation threshold (correlation_threshold)

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

from jump_detection.algorithms.correlation import (
    CorrelationParameters,
    _run_correlation_pipeline,
)
from jump_detection.analysis.precise import calculate_precise_jump_boundaries
from jump_detection.analysis.loss import compute_precision_loss
from jump_detection.annotations import load_annotations
from jump_detection.config import (
    SAMPLING_RATE,
    DEFAULT_DATA_FILES,
    CORRELATION_BUFFER_SIZE_DEFAULT,
    CORRELATION_NEGATIVE_FRAMES_DEFAULT,
    CORRELATION_ZERO_FRAMES_DEFAULT,
    CORRELATION_POSITIVE_FRAMES_DEFAULT,
    CORRELATION_THRESHOLD_DEFAULT,
)
from jump_detection.data import load_dataset
from jump_detection.types import DetectionResult, Jump

# Parameter ranges for grid search (if needed)
BUFFER_SIZE_VALUES = np.arange(10, 100, 5, dtype=int)  # Buffer size: 10 to 95
NEGATIVE_FRAMES_VALUES = np.arange(5, 50, 2, dtype=int)  # Negative frames: 5 to 49
ZERO_FRAMES_VALUES = np.arange(1, 30, 1, dtype=int)  # Zero frames: 1 to 29
POSITIVE_FRAMES_VALUES = np.arange(1, 30, 1, dtype=int)  # Positive frames: 1 to 29
CORRELATION_THRESHOLD_VALUES = np.linspace(0.0, 500.0, 40)  # Correlation threshold: 0 to 500

# Get project root for path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Use DEFAULT_DATA_FILES from config, converting Path objects to strings
DATA_FILES = [str(f) for f in DEFAULT_DATA_FILES]

OUTPUT_DIR = PROJECT_ROOT / "results"
SEARCH_WINDOW = 70  # For precise jump detection

# Parameter bounds for Langevin sampling
PARAM_BOUNDS = {
    "buffer_size": (10.0, 100.0),
    "negative_frames": (5.0, 50.0),
    "zero_frames": (1.0, 30.0),
    "positive_frames": (1.0, 30.0),
    "correlation_threshold": (0.0, 500.0),
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
        result: DetectionResult from correlation pipeline
        
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
    buffer_size: int,
    negative_frames: int,
    zero_frames: int,
    positive_frames: int,
    correlation_threshold: float,
) -> float:
    """Compute loss for concatenated data with given parameters.
    
    Args:
        concatenated_data: All sensor data concatenated vertically
        concatenated_markers: All annotation markers with offsets applied
        buffer_size: Buffer size parameter
        negative_frames: Negative frames parameter
        zero_frames: Zero frames parameter
        positive_frames: Positive frames parameter
        correlation_threshold: Correlation threshold parameter
        
    Returns:
        Loss value (false_positives + false_negatives)
    """
    # Validate that frame counts don't exceed buffer size
    total_frames = negative_frames + zero_frames + positive_frames
    if total_frames > buffer_size or buffer_size <= 0 or negative_frames <= 0 or zero_frames <= 0 or positive_frames <= 0:
        return float('inf')  # Invalid parameter combination
    
    params = CorrelationParameters(
        buffer_size=int(buffer_size),
        negative_frames=int(negative_frames),
        zero_frames=int(zero_frames),
        positive_frames=int(positive_frames),
        correlation_threshold=correlation_threshold,
    )
    
    # Run correlation pipeline directly on concatenated data
    signals, jumps, metadata = _run_correlation_pipeline(concatenated_data, params)
    
    # Create DetectionResult for compatibility with precise jump calculation
    result = DetectionResult(
        participant_name=None,
        sampling_rate=SAMPLING_RATE,
        raw_data=concatenated_data,
        pooled_data=signals.get("pooled", concatenated_data.sum(axis=1)),
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
    args: tuple[np.ndarray, list[int], int, int, int, int, float],
) -> tuple[int, int, int, int, float, float]:
    """Wrapper function for parallel processing of parameter combinations.
    
    Args:
        args: Tuple of (concatenated_data, concatenated_markers, buffer_size,
                       negative_frames, zero_frames, positive_frames, correlation_threshold)
        
    Returns:
        Tuple of (buffer_size, negative_frames, zero_frames, positive_frames,
                 correlation_threshold, loss)
    """
    concatenated_data, concatenated_markers, buffer_size, negative_frames, \
        zero_frames, positive_frames, correlation_threshold = args
    
    loss = compute_loss_for_parameter_combination(
        concatenated_data,
        concatenated_markers,
        buffer_size,
        negative_frames,
        zero_frames,
        positive_frames,
        correlation_threshold,
    )
    return buffer_size, negative_frames, zero_frames, positive_frames, correlation_threshold, loss


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
        BUFFER_SIZE_VALUES,
        NEGATIVE_FRAMES_VALUES,
        ZERO_FRAMES_VALUES,
        POSITIVE_FRAMES_VALUES,
        CORRELATION_THRESHOLD_VALUES,
    ))
    
    # Filter out invalid combinations
    valid_combinations = [
        (buf, neg, zero, pos, thresh)
        for buf, neg, zero, pos, thresh in all_combinations
        if neg + zero + pos <= buf and buf > 0 and neg > 0 and zero > 0 and pos > 0
    ]
    
    total = len(valid_combinations)
    
    if show_progress:
        print(f"Total parameter combinations to test: {total:,}")
        print(f"  Buffer size: {len(BUFFER_SIZE_VALUES)} values")
        print(f"  Negative frames: {len(NEGATIVE_FRAMES_VALUES)} values")
        print(f"  Zero frames: {len(ZERO_FRAMES_VALUES)} values")
        print(f"  Positive frames: {len(POSITIVE_FRAMES_VALUES)} values")
        print(f"  Correlation threshold: {len(CORRELATION_THRESHOLD_VALUES)} values")
    
    results = []
    completed = 0
    
    # Prepare all parameter combinations for parallel processing
    parameter_combinations = [
        (concatenated_data, concatenated_markers, int(buf), int(neg),
         int(zero), int(pos), float(thresh))
        for buf, neg, zero, pos, thresh in valid_combinations
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
                buffer_size, negative_frames, zero_frames, positive_frames, \
                    correlation_threshold, loss = future.result()
                
                results.append({
                    "buffer_size": int(buffer_size),
                    "negative_frames": int(negative_frames),
                    "zero_frames": int(zero_frames),
                    "positive_frames": int(positive_frames),
                    "correlation_threshold": float(correlation_threshold),
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
    print(f"{'Rank':<6} {'Buffer':<10} {'Neg':<8} {'Zero':<8} {'Pos':<8} {'Threshold':<12} {'Loss':<10}")
    print(f"{'-'*100}")
    
    for i, result in enumerate(results[:top_n], 1):
        print(
            f"{i:<6} "
            f"{result['buffer_size']:<10} "
            f"{result['negative_frames']:<8} "
            f"{result['zero_frames']:<8} "
            f"{result['positive_frames']:<8} "
            f"{result['correlation_threshold']:<12.4f} "
            f"{result['loss']:<10.4f}"
        )
    
    print(f"{'='*100}")
    
    if results:
        best = results[0]
        print(f"\nBEST RESULT:")
        print(f"  Buffer size:          {best['buffer_size']:.6f}")
        print(f"  Negative frames:      {best['negative_frames']:.6f}")
        print(f"  Zero frames:          {best['zero_frames']:.6f}")
        print(f"  Positive frames:      {best['positive_frames']:.6f}")
        print(f"  Correlation threshold: {best['correlation_threshold']:.6f}")
        print(f"  Loss:                 {best['loss']:.6f}")
        print(f"{'='*100}\n")


def random_initial_parameters() -> dict[str, float]:
    """Generate random initial parameters within bounds.
    
    Returns:
        Dictionary with parameter values
    """
    buffer_size = random.uniform(*PARAM_BOUNDS["buffer_size"])
    negative_frames = random.uniform(*PARAM_BOUNDS["negative_frames"])
    zero_frames = random.uniform(*PARAM_BOUNDS["zero_frames"])
    positive_frames = random.uniform(*PARAM_BOUNDS["positive_frames"])
    
    # Ensure frame counts don't exceed buffer size
    total_frames = negative_frames + zero_frames + positive_frames
    if total_frames > buffer_size:
        # Scale down proportionally
        scale = buffer_size / total_frames
        negative_frames *= scale
        zero_frames *= scale
        positive_frames *= scale
    
    return {
        "buffer_size": float(buffer_size),
        "negative_frames": float(negative_frames),
        "zero_frames": float(zero_frames),
        "positive_frames": float(positive_frames),
        "correlation_threshold": random.uniform(*PARAM_BOUNDS["correlation_threshold"]),
    }


def default_initial_parameters() -> dict[str, float]:
    """Generate initial parameters from defaults.
    
    Returns:
        Dictionary with default parameter values
    """
    return {
        "buffer_size": float(CORRELATION_BUFFER_SIZE_DEFAULT),
        "negative_frames": float(CORRELATION_NEGATIVE_FRAMES_DEFAULT),
        "zero_frames": float(CORRELATION_ZERO_FRAMES_DEFAULT),
        "positive_frames": float(CORRELATION_POSITIVE_FRAMES_DEFAULT),
        "correlation_threshold": float(CORRELATION_THRESHOLD_DEFAULT),
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
            "buffer_size": (PARAM_BOUNDS["buffer_size"][1] - PARAM_BOUNDS["buffer_size"][0]) * 0.05,
            "negative_frames": (PARAM_BOUNDS["negative_frames"][1] - PARAM_BOUNDS["negative_frames"][0]) * 0.05,
            "zero_frames": (PARAM_BOUNDS["zero_frames"][1] - PARAM_BOUNDS["zero_frames"][0]) * 0.05,
            "positive_frames": (PARAM_BOUNDS["positive_frames"][1] - PARAM_BOUNDS["positive_frames"][0]) * 0.05,
            "correlation_threshold": (PARAM_BOUNDS["correlation_threshold"][1] - PARAM_BOUNDS["correlation_threshold"][0]) * 0.05,
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
    
    # Ensure frame counts don't exceed buffer size
    total_frames = new_params["negative_frames"] + new_params["zero_frames"] + new_params["positive_frames"]
    if total_frames > new_params["buffer_size"]:
        # Scale down proportionally
        scale = new_params["buffer_size"] / total_frames
        new_params["negative_frames"] *= scale
        new_params["zero_frames"] *= scale
        new_params["positive_frames"] *= scale
    
    # Ensure minimum values
    new_params["negative_frames"] = max(1.0, new_params["negative_frames"])
    new_params["zero_frames"] = max(1.0, new_params["zero_frames"])
    new_params["positive_frames"] = max(1.0, new_params["positive_frames"])
    new_params["buffer_size"] = max(10.0, new_params["buffer_size"])
    
    return new_params


def langevin_sampling(
    concatenated_data: np.ndarray,
    concatenated_markers: list[int],
    n_iterations: int = 10000,
    initial_params: dict[str, float] | None = None,
    temperature: float = 10.0,
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
        initial_params: Starting parameters (if None, uses defaults)
        temperature: Initial temperature for acceptance probability
        temperature_decay: Temperature decay factor per iteration
        step_sizes: Step sizes for each parameter (default: 5% of range)
        perturbation_prob: Probability of perturbing each parameter
        show_progress: Whether to show progress updates
        
    Returns:
        Tuple of (all_results, best_result_dict)
    """
    # Initialize parameters
    if initial_params is None:
        current_params = default_initial_parameters()
    else:
        current_params = initial_params.copy()
    
    # Compute initial loss
    current_loss = compute_loss_for_parameter_combination(
        concatenated_data,
        concatenated_markers,
        current_params["buffer_size"],
        current_params["negative_frames"],
        current_params["zero_frames"],
        current_params["positive_frames"],
        current_params["correlation_threshold"],
    )
    
    best_params = current_params.copy()
    best_loss = current_loss
    
    all_results = [{
        "buffer_size": float(current_params["buffer_size"]),
        "negative_frames": float(current_params["negative_frames"]),
        "zero_frames": float(current_params["zero_frames"]),
        "positive_frames": float(current_params["positive_frames"]),
        "correlation_threshold": float(current_params["correlation_threshold"]),
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
            proposed_params["buffer_size"],
            proposed_params["negative_frames"],
            proposed_params["zero_frames"],
            proposed_params["positive_frames"],
            proposed_params["correlation_threshold"],
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
            "buffer_size": float(proposed_params["buffer_size"]),
            "negative_frames": float(proposed_params["negative_frames"]),
            "zero_frames": float(proposed_params["zero_frames"]),
            "positive_frames": float(proposed_params["positive_frames"]),
            "correlation_threshold": float(proposed_params["correlation_threshold"]),
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
                f"Iteration {iteration}/{n_iterations} ({progress:.1f}%) | "
                f"Loss: {current_loss:.4f} | "
                f"Best: {best_loss:.4f} | "
                f"Acceptance: {acceptance_rate:.1f}% | "
                f"Temp: {current_temp:.4f}"
            )
    
    best_result = {
        "buffer_size": float(best_params["buffer_size"]),
        "negative_frames": float(best_params["negative_frames"]),
        "zero_frames": float(best_params["zero_frames"]),
        "positive_frames": float(best_params["positive_frames"]),
        "correlation_threshold": float(best_params["correlation_threshold"]),
        "loss": float(best_loss),
    }
    
    return all_results, best_result


def save_results_to_json(results: list[dict], output_path: Path) -> None:
    """Save results to JSON file.
    
    Args:
        results: List of result dictionaries
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "method": "grid_search",
        "total_results": len(results),
        "results": results,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(results)} results to {output_path}")


def convert_to_json_serializable(obj: object) -> object:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def main():
    """Main function to run comprehensive correlation algorithm optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive parameter optimization for correlation algorithm"
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
        help="Temperature decay rate per iteration (default: 0.9995)"
    )
    parser.add_argument(
        "--initial-params",
        type=str,
        default=None,
        help="Path to JSON file with initial parameters"
    )
    
    args = parser.parse_args()
    
    print("COMPREHENSIVE PARAMETER OPTIMIZATION (CORRELATION ALGORITHM)")
    print("="*100)
    print(f"Started at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print()
    
    # Load all data files and annotations
    print("Loading data files and annotations...")
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
    
    # Run Langevin sampling
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
    print(f"  Buffer size:          {langevin_best['buffer_size']:.6f}")
    print(f"  Negative frames:      {langevin_best['negative_frames']:.6f}")
    print(f"  Zero frames:          {langevin_best['zero_frames']:.6f}")
    print(f"  Positive frames:      {langevin_best['positive_frames']:.6f}")
    print(f"  Correlation threshold: {langevin_best['correlation_threshold']:.6f}")
    print(f"  Loss:                 {langevin_best['loss']:.6f}")
    print(f"{'='*100}")
    
    # Print top accepted results
    print(f"\nTOP {min(20, len(accepted_results))} ACCEPTED RESULTS:")
    print_optimized_results(accepted_results, top_n=20)
    
    # Save Langevin results
    langevin_output = OUTPUT_DIR / "comprehensive_correlation_loss_langevin_results.json"
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
    
    print(f"\nCompleted at: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()

