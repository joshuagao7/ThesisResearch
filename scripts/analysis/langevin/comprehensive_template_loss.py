"""Comprehensive parameter optimization for template-based algorithm using Langevin sampling.

This script optimizes all parameters for the template algorithm:
- Takeoff template: 50 time weights + n_sensors sensor weights + threshold
- Landing template: 50 time weights + n_sensors sensor weights + threshold
- Time constraints: min_flight_time, max_flight_time
- Optional: in_air_threshold

Total: ~200 parameters (50+48+50+48+4 = 200)

Uses precise jump boundaries and precision loss function.
"""

from pathlib import Path
from datetime import datetime
import json
import random

import numpy as np

import sys
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from jump_detection.algorithms.template import (
    TemplateParameters,
    _run_template_pipeline,
    _create_default_template_parameters,
)
from jump_detection.analysis.precise import calculate_precise_jump_boundaries
from jump_detection.analysis.loss import compute_precision_loss
from jump_detection.annotations import load_annotations
from jump_detection.config import SAMPLING_RATE, DEFAULT_DATA_FILES
from jump_detection.data import load_dataset
from jump_detection.types import DetectionResult, Jump, TemplateNames

# Get project root for path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Use DEFAULT_DATA_FILES from config, converting Path objects to strings
DATA_FILES = [str(f) for f in DEFAULT_DATA_FILES]

OUTPUT_DIR = PROJECT_ROOT / "results"
SEARCH_WINDOW = 70  # For precise jump detection

TEMPLATE_SIZE = 50
DEFAULT_N_SENSORS = 48  # Will be adjusted based on actual data

# Parameter bounds for Langevin sampling
# We'll use bounds for individual weights
TIME_WEIGHT_BOUNDS = (-2.0, 2.0)  # Bounds for each time weight
SENSOR_WEIGHT_BOUNDS = (-2.0, 2.0)  # Bounds for each sensor weight
THRESHOLD_BOUNDS = (0.0, 1.0)  # Correlation threshold (normalized)
FLIGHT_TIME_BOUNDS = (0.1, 1.5)  # Min/max flight time


def concatenate_all_data_and_annotations(
    data_files_with_annotations: list[tuple[Path, list[int]]],
) -> tuple[np.ndarray, list[int], list[int]]:
    """Concatenate all data files and annotations into single arrays."""
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
    """Calculate precise jump boundaries from detection result on concatenated data."""
    signal = result.pooled_data if result.pooled_data is not None else concatenated_data.sum(axis=1)
    precise_jumps = []
    
    for jump in result.jumps:
        precise_data = calculate_precise_jump_boundaries(signal, jump.center, SEARCH_WINDOW)
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


def params_dict_to_template_params(params: dict[str, object], n_sensors: int) -> TemplateParameters:
    """Convert flat parameter dictionary to TemplateParameters."""
    # Extract arrays
    takeoff_time_weights = np.array(params["takeoff_time_weights"])
    takeoff_sensor_weights = np.array(params["takeoff_sensor_weights"])
    landing_time_weights = np.array(params["landing_time_weights"])
    landing_sensor_weights = np.array(params["landing_sensor_weights"])
    
    # Adjust sensor weights to match actual sensor count
    if len(takeoff_sensor_weights) != n_sensors:
        if len(takeoff_sensor_weights) < n_sensors:
            padded = np.zeros(n_sensors)
            padded[:len(takeoff_sensor_weights)] = takeoff_sensor_weights
            takeoff_sensor_weights = padded
        else:
            takeoff_sensor_weights = takeoff_sensor_weights[:n_sensors]
    
    if len(landing_sensor_weights) != n_sensors:
        if len(landing_sensor_weights) < n_sensors:
            padded = np.zeros(n_sensors)
            padded[:len(landing_sensor_weights)] = landing_sensor_weights
            landing_sensor_weights = padded
        else:
            landing_sensor_weights = landing_sensor_weights[:n_sensors]
    
    return TemplateParameters(
        takeoff_time_weights=takeoff_time_weights,
        takeoff_sensor_weights=takeoff_sensor_weights,
        takeoff_threshold=float(params["takeoff_threshold"]),
        landing_time_weights=landing_time_weights,
        landing_sensor_weights=landing_sensor_weights,
        landing_threshold=float(params["landing_threshold"]),
        min_flight_time=float(params["min_flight_time"]),
        max_flight_time=float(params["max_flight_time"]),
        in_air_threshold=float(params.get("in_air_threshold", 0)) if params.get("in_air_threshold", 0) > 0 else None,
    )


def compute_loss_for_parameter_combination(
    concatenated_data: np.ndarray,
    concatenated_markers: list[int],
    params: dict[str, object],
) -> float:
    """Compute loss for concatenated data with given parameters."""
    # Validate constraints
    if params["min_flight_time"] >= params["max_flight_time"]:
        return float('inf')
    
    try:
        n_sensors = concatenated_data.shape[1]
        template_params = params_dict_to_template_params(params, n_sensors)
        
        # Run template pipeline directly on concatenated data
        signals, jumps, metadata = _run_template_pipeline(concatenated_data, template_params)
        
        # Create DetectionResult for compatibility with precise jump calculation
        result = DetectionResult(
            participant_name=None,
            sampling_rate=SAMPLING_RATE,
            raw_data=concatenated_data,
            pooled_data=signals[TemplateNames.POOLED.value],
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
    except Exception as e:
        import traceback
        if not hasattr(compute_loss_for_parameter_combination, '_error_count'):
            compute_loss_for_parameter_combination._error_count = 0
        if compute_loss_for_parameter_combination._error_count < 3:
            print(f"\nError in compute_loss_for_parameter_combination: {e}")
            print(traceback.format_exc())
            compute_loss_for_parameter_combination._error_count += 1
        return float('inf')


def default_initial_parameters(n_sensors: int) -> dict[str, object]:
    """Generate initial parameters from default template parameters."""
    default_template = _create_default_template_parameters()
    
    # Adjust sensor weights to match actual sensor count
    takeoff_sensor = default_template.takeoff_sensor_weights
    landing_sensor = default_template.landing_sensor_weights
    
    if len(takeoff_sensor) != n_sensors:
        if len(takeoff_sensor) < n_sensors:
            padded = np.ones(n_sensors) / n_sensors
            padded[:len(takeoff_sensor)] = takeoff_sensor
            takeoff_sensor = padded
        else:
            takeoff_sensor = takeoff_sensor[:n_sensors]
    
    if len(landing_sensor) != n_sensors:
        if len(landing_sensor) < n_sensors:
            padded = np.ones(n_sensors) / n_sensors
            padded[:len(landing_sensor)] = landing_sensor
            landing_sensor = padded
        else:
            landing_sensor = landing_sensor[:n_sensors]
    
    return {
        "takeoff_time_weights": default_template.takeoff_time_weights.tolist(),
        "takeoff_sensor_weights": takeoff_sensor.tolist(),
        "takeoff_threshold": default_template.takeoff_threshold,
        "landing_time_weights": default_template.landing_time_weights.tolist(),
        "landing_sensor_weights": landing_sensor.tolist(),
        "landing_threshold": default_template.landing_threshold,
        "min_flight_time": default_template.min_flight_time,
        "max_flight_time": default_template.max_flight_time,
        "in_air_threshold": 0.0,  # Disabled by default
    }


def perturb_parameters(
    params: dict[str, object],
    param_to_perturb: str,
    n_sensors: int,
    step_size: float = 0.05,
) -> dict[str, object]:
    """Perturb a single parameter group using Langevin-style noise."""
    new_params = params.copy()
    
    # Perturb time weights
    if param_to_perturb == "takeoff_time_weights":
        takeoff_time = np.array(new_params["takeoff_time_weights"])
        noise = np.random.normal(0, step_size, size=takeoff_time.shape)
        takeoff_time = np.clip(takeoff_time + noise, *TIME_WEIGHT_BOUNDS)
        new_params["takeoff_time_weights"] = takeoff_time.tolist()
    
    if param_to_perturb == "landing_time_weights":
        landing_time = np.array(new_params["landing_time_weights"])
        noise = np.random.normal(0, step_size, size=landing_time.shape)
        landing_time = np.clip(landing_time + noise, *TIME_WEIGHT_BOUNDS)
        new_params["landing_time_weights"] = landing_time.tolist()
    
    # Perturb sensor weights
    if param_to_perturb == "takeoff_sensor_weights":
        takeoff_sensor = np.array(new_params["takeoff_sensor_weights"])
        if len(takeoff_sensor) < n_sensors:
            padded = np.zeros(n_sensors)
            padded[:len(takeoff_sensor)] = takeoff_sensor
            takeoff_sensor = padded
        else:
            takeoff_sensor = takeoff_sensor[:n_sensors]
        noise = np.random.normal(0, step_size, size=takeoff_sensor.shape)
        takeoff_sensor = np.clip(takeoff_sensor + noise, *SENSOR_WEIGHT_BOUNDS)
        new_params["takeoff_sensor_weights"] = takeoff_sensor.tolist()
    
    if param_to_perturb == "landing_sensor_weights":
        landing_sensor = np.array(new_params["landing_sensor_weights"])
        if len(landing_sensor) < n_sensors:
            padded = np.zeros(n_sensors)
            padded[:len(landing_sensor)] = landing_sensor
            landing_sensor = padded
        else:
            landing_sensor = landing_sensor[:n_sensors]
        noise = np.random.normal(0, step_size, size=landing_sensor.shape)
        landing_sensor = np.clip(landing_sensor + noise, *SENSOR_WEIGHT_BOUNDS)
        new_params["landing_sensor_weights"] = landing_sensor.tolist()
    
    # Perturb thresholds
    if param_to_perturb == "takeoff_threshold":
        noise = np.random.normal(0, step_size)
        new_params["takeoff_threshold"] = float(np.clip(
            new_params["takeoff_threshold"] + noise, *THRESHOLD_BOUNDS
        ))
    
    if param_to_perturb == "landing_threshold":
        noise = np.random.normal(0, step_size)
        new_params["landing_threshold"] = float(np.clip(
            new_params["landing_threshold"] + noise, *THRESHOLD_BOUNDS
        ))
    
    # Perturb flight times
    if param_to_perturb == "min_flight_time":
        noise = np.random.normal(0, 0.02)
        new_params["min_flight_time"] = float(np.clip(
            new_params["min_flight_time"] + noise, *FLIGHT_TIME_BOUNDS
        ))
    
    if param_to_perturb == "max_flight_time":
        noise = np.random.normal(0, 0.02)
        new_params["max_flight_time"] = float(np.clip(
            new_params["max_flight_time"] + noise, *FLIGHT_TIME_BOUNDS
        ))
    
    # Ensure constraints
    if new_params["min_flight_time"] >= new_params["max_flight_time"]:
        mid_point = (new_params["min_flight_time"] + new_params["max_flight_time"]) / 2
        new_params["min_flight_time"] = mid_point - 0.05
        new_params["max_flight_time"] = mid_point + 0.05
        new_params["min_flight_time"] = np.clip(new_params["min_flight_time"], *FLIGHT_TIME_BOUNDS)
        new_params["max_flight_time"] = np.clip(new_params["max_flight_time"], *FLIGHT_TIME_BOUNDS)
    
    return new_params


def langevin_sampling(
    concatenated_data: np.ndarray,
    concatenated_markers: list[int],
    n_iterations: int = 1000,
    initial_params: dict[str, object] | None = None,
    temperature: float = 1.0,
    temperature_decay: float = 0.9995,
    step_size: float = 0.05,
    show_progress: bool = True,
) -> tuple[list[dict], dict]:
    """Langevin-inspired sampling strategy for parameter optimization.
    
    Parameters are perturbed sequentially, cycling through all parameter groups.
    """
    n_sensors = concatenated_data.shape[1]
    
    # Initialize parameters
    if initial_params is None:
        current_params = default_initial_parameters(n_sensors)
    else:
        current_params = initial_params.copy()
    
    # Compute initial loss
    current_loss = compute_loss_for_parameter_combination(
        concatenated_data,
        concatenated_markers,
        current_params,
    )
    
    best_params = current_params.copy()
    best_loss = current_loss
    
    all_results = [{
        **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in current_params.items()},
        "loss": float(current_loss),
        "accepted": True,
        "iteration": 0,
        "temperature": float(temperature),
    }]
    
    current_temp = temperature
    accepted_count = 1
    
    if show_progress:
        print(f"Starting Langevin sampling with {n_iterations} iterations")
        print(f"Initial loss: {current_loss:.4f}")
        print(f"Initial temperature: {temperature:.4f}")
        print(f"Template size: {TEMPLATE_SIZE}, Sensors: {n_sensors}")
        print(f"Total parameters: ~{TEMPLATE_SIZE * 2 + n_sensors * 2 + 4}")
        print()
    
    # Define parameter groups to cycle through sequentially
    param_groups = [
        "takeoff_time_weights",
        "landing_time_weights",
        "takeoff_sensor_weights",
        "landing_sensor_weights",
        "takeoff_threshold",
        "landing_threshold",
        "min_flight_time",
        "max_flight_time",
    ]
    
    for iteration in range(1, n_iterations + 1):
        # Select parameter group to perturb sequentially (cycle through all groups)
        param_to_perturb = param_groups[iteration % len(param_groups)]
        
        # Perturb the selected parameter group
        proposed_params = perturb_parameters(
            current_params,
            param_to_perturb=param_to_perturb,
            n_sensors=n_sensors,
            step_size=step_size,
        )
        
        # Compute loss for proposed parameters
        proposed_loss = compute_loss_for_parameter_combination(
            concatenated_data,
            concatenated_markers,
            proposed_params,
        )
        
        # Metropolis-Hastings acceptance criterion
        loss_diff = proposed_loss - current_loss
        accept = False
        
        if loss_diff <= 0:
            accept = True
        elif current_temp > 0:
            accept_prob = np.exp(-loss_diff / current_temp)
            accept = random.random() < accept_prob
        
        if accept:
            current_params = proposed_params
            current_loss = proposed_loss
            accepted_count += 1
            
            if proposed_loss < best_loss:
                best_params = proposed_params.copy()
                best_loss = proposed_loss
        
        # Record result
        all_results.append({
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in proposed_params.items()},
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
        print()
    
    best_result = {
        **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in best_params.items()},
        "loss": float(best_loss),
        "total_iterations": int(n_iterations),
        "accepted_count": int(accepted_count),
        "acceptance_rate": float(accepted_count / n_iterations),
    }
    
    return all_results, best_result


def convert_to_json_serializable(obj):
    """Recursively convert numpy types to JSON-compatible types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif type(obj).__module__ == 'numpy':
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


def print_best_result(best_result: dict) -> None:
    """Print the best result in a readable format."""
    print(f"\n{'='*100}")
    print("BEST RESULT FROM LANGEVIN SAMPLING")
    print(f"{'='*100}")
    
    print("\nTakeoff Template:")
    takeoff_time = np.array(best_result["takeoff_time_weights"])
    takeoff_sensor = np.array(best_result["takeoff_sensor_weights"])
    print(f"  Time weights: min={takeoff_time.min():.4f}, max={takeoff_time.max():.4f}, mean={takeoff_time.mean():.4f}")
    print(f"  Sensor weights: min={takeoff_sensor.min():.4f}, max={takeoff_sensor.max():.4f}, mean={takeoff_sensor.mean():.4f}")
    print(f"  Threshold: {best_result['takeoff_threshold']:.6f}")
    
    print("\nLanding Template:")
    landing_time = np.array(best_result["landing_time_weights"])
    landing_sensor = np.array(best_result["landing_sensor_weights"])
    print(f"  Time weights: min={landing_time.min():.4f}, max={landing_time.max():.4f}, mean={landing_time.mean():.4f}")
    print(f"  Sensor weights: min={landing_sensor.min():.4f}, max={landing_sensor.max():.4f}, mean={landing_sensor.mean():.4f}")
    print(f"  Threshold: {best_result['landing_threshold']:.6f}")
    
    print("\nTime Constraints:")
    print(f"  Min flight time: {best_result['min_flight_time']:.6f} seconds")
    print(f"  Max flight time: {best_result['max_flight_time']:.6f} seconds")
    
    print(f"\nLoss: {best_result['loss']:.6f}")
    print(f"Total iterations: {best_result['total_iterations']}")
    print(f"Accepted count: {best_result['accepted_count']}")
    print(f"Acceptance rate: {best_result['acceptance_rate']:.2f}%")
    print(f"{'='*100}\n")


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive parameter optimization for template algorithm"
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
    
    print("COMPREHENSIVE PARAMETER OPTIMIZATION (TEMPLATE ALGORITHM)")
    print("Using precise jump boundaries and precision loss function")
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
    print(f"  ✓ Number of sensors: {concatenated_data.shape[1]}")
    
    # Load initial parameters if provided
    initial_params = None
    if args.initial_params:
        with open(args.initial_params, 'r') as f:
            initial_params = json.load(f)
        print(f"\nLoaded initial parameters from {args.initial_params}")
    
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
    
    # Extract only accepted results
    accepted_results = [
        {k: v for k, v in r.items() if k not in ["accepted", "iteration", "temperature"]}
        for r in langevin_all
        if r.get("accepted", False)
    ]
    
    # Sort by loss
    accepted_results.sort(key=lambda x: x["loss"])
    
    # Print results
    print_best_result(langevin_best)
    
    # Save results
    langevin_output = OUTPUT_DIR / "comprehensive_template_loss_langevin_results.json"
    langevin_output.parent.mkdir(parents=True, exist_ok=True)
    
    langevin_output_data = {
        "timestamp": datetime.now().isoformat(),
        "method": "langevin_sampling",
        "n_iterations": int(args.iterations),
        "initial_temperature": float(args.temperature),
        "temperature_decay": float(args.temp_decay),
        "initial_params": convert_to_json_serializable(initial_params) if initial_params else None,
        "best_result": convert_to_json_serializable(langevin_best),
        "all_iterations": convert_to_json_serializable(langevin_all[:1000]),  # Save first 1000 iterations
        "accepted_results": convert_to_json_serializable(accepted_results[:100]),  # Top 100
    }
    
    with open(langevin_output, 'w') as f:
        json.dump(langevin_output_data, f, indent=2)
    
    print(f"Saved Langevin sampling results to {langevin_output}")
    print(f"\nCompleted at: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()


