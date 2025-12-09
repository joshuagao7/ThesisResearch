"""Comprehensive parameter optimization for ensemble voting algorithm using Langevin sampling.

This script optimizes all parameters for the ensemble algorithm:
- All individual algorithm parameters (threshold, derivative, correlation, hybrid)
- Weights for each condition (~11 weights)
- Final score threshold
- Ensemble physics constraints

Uses precise jump boundaries and precision loss function.
Focuses on Langevin sampling due to large parameter space (~30-35 parameters).
"""

from pathlib import Path
from datetime import datetime
import json
import random

import numpy as np

import sys
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from jump_detection.algorithms.ensemble import (
    EnsembleParameters,
    _run_ensemble_pipeline,
    _create_default_ensemble_parameters,
)
from jump_detection.algorithms.threshold import ThresholdParameters
from jump_detection.algorithms.derivative import DerivativeParameters
from jump_detection.algorithms.correlation import CorrelationParameters
from jump_detection.algorithms.hybrid import HybridParameters
from jump_detection.analysis.precise import calculate_precise_jump_boundaries
from jump_detection.analysis.loss import compute_precision_loss
from jump_detection.annotations import load_annotations
from jump_detection.config import SAMPLING_RATE, DEFAULT_DATA_FILES
from jump_detection.data import load_dataset
from jump_detection.types import DetectionResult, Jump, EnsembleNames

# Get project root for path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Use DEFAULT_DATA_FILES from config, converting Path objects to strings
DATA_FILES = [str(f) for f in DEFAULT_DATA_FILES]

OUTPUT_DIR = PROJECT_ROOT / "results"
SEARCH_WINDOW = 70  # For precise jump detection

# Parameter bounds for Langevin sampling
PARAM_BOUNDS = {
    # Threshold algorithm parameters
    "threshold_threshold": (100.0, 400.0),
    "threshold_derivative_threshold": (0.0, 100.0),
    "threshold_derivative_window": (1, 50),
    "threshold_min_flight_time": (0.1, 0.5),
    "threshold_max_flight_time": (0.5, 1.5),
    
    # Derivative algorithm parameters
    "derivative_upper_threshold": (0.0, 100.0),
    "derivative_lower_threshold": (-100.0, 0.0),
    "derivative_in_air_threshold": (100.0, 400.0),
    "derivative_min_flight_time": (0.1, 0.5),
    "derivative_max_flight_time": (0.5, 1.5),
    
    # Correlation algorithm parameters
    "correlation_buffer_size": (10, 100),
    "correlation_negative_frames": (5, 50),
    "correlation_zero_frames": (1, 30),
    "correlation_positive_frames": (1, 30),
    "correlation_threshold": (0.0, 500.0),
    
    # Hybrid algorithm parameters
    "hybrid_takeoff_threshold": (100.0, 400.0),
    "hybrid_landing_derivative_threshold": (0.0, 100.0),
    "hybrid_in_air_threshold": (100.0, 400.0),
    "hybrid_min_flight_time": (0.1, 0.5),
    "hybrid_max_flight_time": (0.5, 1.5),
    
    # Ensemble parameters
    "ensemble_min_flight_time": (0.1, 0.5),
    "ensemble_max_flight_time": (0.5, 1.5),
    "score_threshold": (0.0, 20.0),
    
    # Weights (one for each condition)
    "weight_threshold_mask": (-5.0, 5.0),
    "weight_derivative_binary": (-5.0, 5.0),
    "weight_physics_filtered": (-5.0, 5.0),
    "weight_upper_mask": (-5.0, 5.0),
    "weight_lower_mask": (-5.0, 5.0),
    "weight_derivative_in_air": (-5.0, 5.0),
    "weight_correlation_above_threshold": (-5.0, 5.0),
    "weight_takeoff_mask": (-5.0, 5.0),
    "weight_landing_mask": (-5.0, 5.0),
    "weight_hybrid_in_air": (-5.0, 5.0),
}

# Condition names for weights
CONDITION_NAMES = [
    "threshold_mask",
    "derivative_binary",
    "physics_filtered",
    "upper_mask",
    "lower_mask",
    "derivative_in_air",
    "correlation_above_threshold",
    "takeoff_mask",
    "landing_mask",
    "hybrid_in_air",
]


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


def params_dict_to_ensemble_params(params: dict[str, float]) -> EnsembleParameters:
    """Convert flat parameter dictionary to EnsembleParameters."""
    # Extract weights
    weights = {}
    for condition_name in CONDITION_NAMES:
        weight_key = f"weight_{condition_name}"
        weights[condition_name] = params.get(weight_key, 1.0)
    
    # Create individual algorithm parameters
    threshold_params = ThresholdParameters(
        threshold=params["threshold_threshold"],
        derivative_threshold=params["threshold_derivative_threshold"],
        min_flight_time=params["threshold_min_flight_time"],
        max_flight_time=params["threshold_max_flight_time"],
    )
    
    derivative_params = DerivativeParameters(
        upper_threshold=params["derivative_upper_threshold"],
        lower_threshold=params["derivative_lower_threshold"],
        in_air_threshold=params["derivative_in_air_threshold"],
        min_flight_time=params["derivative_min_flight_time"],
        max_flight_time=params["derivative_max_flight_time"],
    )
    
    correlation_params = CorrelationParameters(
        buffer_size=int(params["correlation_buffer_size"]),
        negative_frames=int(params["correlation_negative_frames"]),
        zero_frames=int(params["correlation_zero_frames"]),
        positive_frames=int(params["correlation_positive_frames"]),
        correlation_threshold=params["correlation_threshold"],
    )
    
    hybrid_params = HybridParameters(
        takeoff_threshold=params["hybrid_takeoff_threshold"],
        landing_derivative_threshold=params["hybrid_landing_derivative_threshold"],
        in_air_threshold=params["hybrid_in_air_threshold"],
        min_flight_time=params["hybrid_min_flight_time"],
        max_flight_time=params["hybrid_max_flight_time"],
    )
    
    return EnsembleParameters(
        threshold_params=threshold_params,
        derivative_params=derivative_params,
        correlation_params=correlation_params,
        hybrid_params=hybrid_params,
        weights=weights,
        score_threshold=params["score_threshold"],
        min_flight_time=params["ensemble_min_flight_time"],
        max_flight_time=params["ensemble_max_flight_time"],
    )


def compute_loss_for_parameter_combination(
    concatenated_data: np.ndarray,
    concatenated_markers: list[int],
    params: dict[str, float],
) -> float:
    """Compute loss for concatenated data with given parameters.
    
    Args:
        concatenated_data: All sensor data concatenated vertically
        concatenated_markers: All annotation markers with offsets applied
        params: Dictionary of all parameter values
        
    Returns:
        Loss value (false_positives + false_negatives)
    """
    # Validate constraints
    if params["threshold_min_flight_time"] >= params["threshold_max_flight_time"]:
        return float('inf')
    if params["derivative_min_flight_time"] >= params["derivative_max_flight_time"]:
        return float('inf')
    if params["hybrid_min_flight_time"] >= params["hybrid_max_flight_time"]:
        return float('inf')
    if params["ensemble_min_flight_time"] >= params["ensemble_max_flight_time"]:
        return float('inf')
    
    # Validate correlation buffer size
    buffer_size = int(params["correlation_buffer_size"])
    neg_frames = int(params["correlation_negative_frames"])
    zero_frames = int(params["correlation_zero_frames"])
    pos_frames = int(params["correlation_positive_frames"])
    if neg_frames + zero_frames + pos_frames != buffer_size:
        return float('inf')
    
    try:
        ensemble_params = params_dict_to_ensemble_params(params)
        
        # Run ensemble pipeline directly on concatenated data
        signals, jumps, metadata = _run_ensemble_pipeline(concatenated_data, ensemble_params)
        
        # Create DetectionResult for compatibility with precise jump calculation
        result = DetectionResult(
            participant_name=None,
            sampling_rate=SAMPLING_RATE,
            raw_data=concatenated_data,
            pooled_data=signals[EnsembleNames.POOLED.value],
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
        # Log the error for debugging, but return infinity
        # Only log first few errors to avoid spam
        import traceback
        if not hasattr(compute_loss_for_parameter_combination, '_error_count'):
            compute_loss_for_parameter_combination._error_count = 0
        if compute_loss_for_parameter_combination._error_count < 3:
            print(f"\nError in compute_loss_for_parameter_combination: {e}")
            print(traceback.format_exc())
            compute_loss_for_parameter_combination._error_count += 1
        return float('inf')


def default_initial_parameters() -> dict[str, float]:
    """Generate initial parameters from default ensemble parameters."""
    from jump_detection.algorithms.ensemble import _create_default_ensemble_parameters
    
    default_ensemble = _create_default_ensemble_parameters()
    
    params = {
        # Threshold algorithm
        "threshold_threshold": default_ensemble.threshold_params.threshold,
        "threshold_derivative_threshold": default_ensemble.threshold_params.derivative_threshold,
        "threshold_derivative_window": 10.0,  # Default value from config, not part of ThresholdParameters
        "threshold_min_flight_time": default_ensemble.threshold_params.min_flight_time,
        "threshold_max_flight_time": default_ensemble.threshold_params.max_flight_time,
        
        # Derivative algorithm
        "derivative_upper_threshold": default_ensemble.derivative_params.upper_threshold,
        "derivative_lower_threshold": default_ensemble.derivative_params.lower_threshold,
        "derivative_in_air_threshold": default_ensemble.derivative_params.in_air_threshold,
        "derivative_min_flight_time": default_ensemble.derivative_params.min_flight_time,
        "derivative_max_flight_time": default_ensemble.derivative_params.max_flight_time,
        
        # Correlation algorithm
        "correlation_buffer_size": float(default_ensemble.correlation_params.buffer_size),
        "correlation_negative_frames": float(default_ensemble.correlation_params.negative_frames),
        "correlation_zero_frames": float(default_ensemble.correlation_params.zero_frames),
        "correlation_positive_frames": float(default_ensemble.correlation_params.positive_frames),
        "correlation_threshold": default_ensemble.correlation_params.correlation_threshold,
        
        # Hybrid algorithm
        "hybrid_takeoff_threshold": default_ensemble.hybrid_params.takeoff_threshold,
        "hybrid_landing_derivative_threshold": default_ensemble.hybrid_params.landing_derivative_threshold,
        "hybrid_in_air_threshold": default_ensemble.hybrid_params.in_air_threshold,
        "hybrid_min_flight_time": default_ensemble.hybrid_params.min_flight_time,
        "hybrid_max_flight_time": default_ensemble.hybrid_params.max_flight_time,
        
        # Ensemble parameters
        "ensemble_min_flight_time": default_ensemble.min_flight_time,
        "ensemble_max_flight_time": default_ensemble.max_flight_time,
        "score_threshold": default_ensemble.score_threshold,
    }
    
    # Add weights
    for condition_name in CONDITION_NAMES:
        weight_key = f"weight_{condition_name}"
        params[weight_key] = default_ensemble.weights.get(condition_name, 1.0)
    
    return params


def random_initial_parameters() -> dict[str, float]:
    """Generate initial parameters starting from reasonable defaults with small random perturbations."""
    from jump_detection.config import (
        THRESHOLD_DEFAULT,
        DERIVATIVE_THRESHOLD_DEFAULT,
        DERIVATIVE_WINDOW_DEFAULT,
        DERIVATIVE_UPPER_DEFAULT,
        DERIVATIVE_LOWER_DEFAULT,
        IN_AIR_THRESHOLD_DEFAULT,
        CORRELATION_BUFFER_SIZE_DEFAULT,
        CORRELATION_NEGATIVE_FRAMES_DEFAULT,
        CORRELATION_ZERO_FRAMES_DEFAULT,
        CORRELATION_POSITIVE_FRAMES_DEFAULT,
        CORRELATION_THRESHOLD_DEFAULT,
        MIN_FLIGHT_TIME_S,
        MAX_FLIGHT_TIME_S,
    )
    
    # Start with optimized defaults, then add small random perturbations
    params = {
        # Threshold algorithm - use defaults with small noise
        "threshold_threshold": THRESHOLD_DEFAULT + random.uniform(-20, 20),
        "threshold_derivative_threshold": DERIVATIVE_THRESHOLD_DEFAULT + random.uniform(-5, 5),
        "threshold_derivative_window": float(DERIVATIVE_WINDOW_DEFAULT + random.randint(-2, 2)),
        "threshold_min_flight_time": MIN_FLIGHT_TIME_S + random.uniform(-0.05, 0.05),
        "threshold_max_flight_time": MAX_FLIGHT_TIME_S + random.uniform(-0.1, 0.1),
        
        # Derivative algorithm - use defaults with small noise
        "derivative_upper_threshold": DERIVATIVE_UPPER_DEFAULT + random.uniform(-3, 3),
        "derivative_lower_threshold": DERIVATIVE_LOWER_DEFAULT + random.uniform(-3, 3),
        "derivative_in_air_threshold": IN_AIR_THRESHOLD_DEFAULT + random.uniform(-20, 20),
        "derivative_min_flight_time": MIN_FLIGHT_TIME_S + random.uniform(-0.05, 0.05),
        "derivative_max_flight_time": MAX_FLIGHT_TIME_S + random.uniform(-0.1, 0.1),
        
        # Correlation algorithm - use defaults with small noise
        "correlation_buffer_size": float(CORRELATION_BUFFER_SIZE_DEFAULT + random.randint(-4, 4)),
        "correlation_negative_frames": float(CORRELATION_NEGATIVE_FRAMES_DEFAULT + random.randint(-2, 2)),
        "correlation_zero_frames": float(CORRELATION_ZERO_FRAMES_DEFAULT + random.randint(-1, 1)),
        "correlation_positive_frames": float(CORRELATION_POSITIVE_FRAMES_DEFAULT + random.randint(-1, 1)),
        "correlation_threshold": CORRELATION_THRESHOLD_DEFAULT + random.uniform(-20, 20),
        
        # Hybrid algorithm - use defaults with small noise
        "hybrid_takeoff_threshold": THRESHOLD_DEFAULT + random.uniform(-20, 20),
        "hybrid_landing_derivative_threshold": 15.0 + random.uniform(-3, 3),
        "hybrid_in_air_threshold": IN_AIR_THRESHOLD_DEFAULT + random.uniform(-20, 20),
        "hybrid_min_flight_time": MIN_FLIGHT_TIME_S + random.uniform(-0.05, 0.05),
        "hybrid_max_flight_time": MAX_FLIGHT_TIME_S + random.uniform(-0.1, 0.1),
        
        # Ensemble parameters
        "ensemble_min_flight_time": MIN_FLIGHT_TIME_S + random.uniform(-0.05, 0.05),
        "ensemble_max_flight_time": MAX_FLIGHT_TIME_S + random.uniform(-0.1, 0.1),
        "score_threshold": 5.0 + random.uniform(-1, 1),  # Start near default
        
        # Weights - start with small positive values, allow some to be negative
        "weight_threshold_mask": random.uniform(0.5, 2.0),
        "weight_derivative_binary": random.uniform(0.5, 2.0),
        "weight_physics_filtered": random.uniform(0.5, 2.0),
        "weight_upper_mask": random.uniform(0.5, 2.0),
        "weight_lower_mask": random.uniform(0.5, 2.0),
        "weight_derivative_in_air": random.uniform(0.5, 2.0),
        "weight_correlation_above_threshold": random.uniform(0.5, 2.0),
        "weight_takeoff_mask": random.uniform(0.5, 2.0),
        "weight_landing_mask": random.uniform(0.5, 2.0),
        "weight_hybrid_in_air": random.uniform(0.5, 2.0),
    }
    
    # Clip all parameters to bounds
    for param_name, value in params.items():
        if param_name in PARAM_BOUNDS:
            min_val, max_val = PARAM_BOUNDS[param_name]
            params[param_name] = np.clip(value, min_val, max_val)
    
    # Ensure correlation buffer size constraint
    buffer_size = int(params["correlation_buffer_size"])
    neg_frames = int(params["correlation_negative_frames"])
    zero_frames = int(params["correlation_zero_frames"])
    pos_frames = int(params["correlation_positive_frames"])
    
    # Adjust to satisfy constraint
    total = neg_frames + zero_frames + pos_frames
    if total != buffer_size:
        # Scale proportionally
        if total > 0:
            scale = buffer_size / total
            params["correlation_negative_frames"] = float(int(neg_frames * scale))
            params["correlation_zero_frames"] = float(int(zero_frames * scale))
            params["correlation_positive_frames"] = float(buffer_size - int(neg_frames * scale) - int(zero_frames * scale))
        else:
            # Default split
            params["correlation_negative_frames"] = float(buffer_size // 2)
            params["correlation_zero_frames"] = float(buffer_size // 4)
            params["correlation_positive_frames"] = float(buffer_size - buffer_size // 2 - buffer_size // 4)
    
    # Ensure flight time constraints
    for prefix in ["threshold", "derivative", "hybrid", "ensemble"]:
        min_key = f"{prefix}_min_flight_time"
        max_key = f"{prefix}_max_flight_time"
        if params[min_key] >= params[max_key]:
            mid_point = (params[min_key] + params[max_key]) / 2
            params[min_key] = mid_point - 0.05
            params[max_key] = mid_point + 0.05
            params[min_key] = np.clip(params[min_key], *PARAM_BOUNDS[min_key])
            params[max_key] = np.clip(params[max_key], *PARAM_BOUNDS[max_key])
    
    return params


def perturb_parameters(
    params: dict[str, float],
    step_sizes: dict[str, float] | None = None,
    perturbation_prob: float = 0.3,
) -> dict[str, float]:
    """Randomly perturb parameters using Langevin-style noise."""
    if step_sizes is None:
        # Default step sizes: 5% of parameter range
        step_sizes = {}
        for param_name, (min_val, max_val) in PARAM_BOUNDS.items():
            step_sizes[param_name] = (max_val - min_val) * 0.05
    
    new_params = params.copy()
    
    # Perturb each parameter independently with given probability
    for param_name in new_params:
        if random.random() < perturbation_prob:
            if "window" in param_name or "frames" in param_name or "buffer_size" in param_name:
                # Integer parameters: use discrete steps
                step = max(1, int(step_sizes[param_name]))
                noise = random.randint(-step, step)
                new_value = new_params[param_name] + noise
            else:
                # Float parameters: use Gaussian noise
                noise = np.random.normal(0, step_sizes[param_name])
                new_value = new_params[param_name] + noise
            
            # Clip to bounds
            min_val, max_val = PARAM_BOUNDS[param_name]
            new_params[param_name] = np.clip(new_value, min_val, max_val)
    
    # Ensure constraints
    # Flight time constraints
    for prefix in ["threshold", "derivative", "hybrid", "ensemble"]:
        min_key = f"{prefix}_min_flight_time"
        max_key = f"{prefix}_max_flight_time"
        if new_params[min_key] >= new_params[max_key]:
            mid_point = (new_params[min_key] + new_params[max_key]) / 2
            new_params[min_key] = mid_point - 0.05
            new_params[max_key] = mid_point + 0.05
            new_params[min_key] = np.clip(new_params[min_key], *PARAM_BOUNDS[min_key])
            new_params[max_key] = np.clip(new_params[max_key], *PARAM_BOUNDS[max_key])
    
    # Correlation buffer size constraint
    buffer_size = int(new_params["correlation_buffer_size"])
    neg_frames = int(new_params["correlation_negative_frames"])
    zero_frames = int(new_params["correlation_zero_frames"])
    pos_frames = int(new_params["correlation_positive_frames"])
    
    total = neg_frames + zero_frames + pos_frames
    if total != buffer_size:
        # Adjust to satisfy constraint
        if total > 0:
            scale = buffer_size / total
            new_params["correlation_negative_frames"] = float(int(neg_frames * scale))
            new_params["correlation_zero_frames"] = float(int(zero_frames * scale))
            new_params["correlation_positive_frames"] = float(buffer_size - int(neg_frames * scale) - int(zero_frames * scale))
        else:
            new_params["correlation_negative_frames"] = float(buffer_size // 2)
            new_params["correlation_zero_frames"] = float(buffer_size // 4)
            new_params["correlation_positive_frames"] = float(buffer_size - buffer_size // 2 - buffer_size // 4)
    
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
    """Langevin-inspired sampling strategy for parameter optimization."""
    # Initialize parameters
    if initial_params is None:
        # Start from defaults with small perturbations for better starting point
        current_params = default_initial_parameters()
        # Add small random perturbations
        for param_name in current_params:
            if param_name in PARAM_BOUNDS:
                min_val, max_val = PARAM_BOUNDS[param_name]
                if "window" in param_name or "frames" in param_name or "buffer_size" in param_name:
                    # Integer parameters: small integer perturbation
                    noise = random.randint(-2, 2)
                    current_params[param_name] = float(np.clip(current_params[param_name] + noise, min_val, max_val))
                else:
                    # Float parameters: small percentage perturbation (5%)
                    range_size = max_val - min_val
                    noise = random.uniform(-range_size * 0.05, range_size * 0.05)
                    current_params[param_name] = float(np.clip(current_params[param_name] + noise, min_val, max_val))
        
        # Ensure constraints are still satisfied
        # Correlation buffer size constraint
        buffer_size = int(current_params["correlation_buffer_size"])
        neg_frames = int(current_params["correlation_negative_frames"])
        zero_frames = int(current_params["correlation_zero_frames"])
        pos_frames = int(current_params["correlation_positive_frames"])
        total = neg_frames + zero_frames + pos_frames
        if total != buffer_size:
            if total > 0:
                scale = buffer_size / total
                current_params["correlation_negative_frames"] = float(int(neg_frames * scale))
                current_params["correlation_zero_frames"] = float(int(zero_frames * scale))
                current_params["correlation_positive_frames"] = float(buffer_size - int(neg_frames * scale) - int(zero_frames * scale))
            else:
                current_params["correlation_negative_frames"] = float(buffer_size // 2)
                current_params["correlation_zero_frames"] = float(buffer_size // 4)
                current_params["correlation_positive_frames"] = float(buffer_size - buffer_size // 2 - buffer_size // 4)
        
        # Flight time constraints
        for prefix in ["threshold", "derivative", "hybrid", "ensemble"]:
            min_key = f"{prefix}_min_flight_time"
            max_key = f"{prefix}_max_flight_time"
            if current_params[min_key] >= current_params[max_key]:
                mid_point = (current_params[min_key] + current_params[max_key]) / 2
                current_params[min_key] = mid_point - 0.05
                current_params[max_key] = mid_point + 0.05
                current_params[min_key] = np.clip(current_params[min_key], *PARAM_BOUNDS[min_key])
                current_params[max_key] = np.clip(current_params[max_key], *PARAM_BOUNDS[max_key])
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
        **{k: float(v) for k, v in current_params.items()},
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
        print(f"Total parameters: {len(current_params)}")
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
            **{k: float(v) for k, v in proposed_params.items()},
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
        **{k: float(v) for k, v in best_params.items()},
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
    
    print("\nThreshold Algorithm Parameters:")
    print(f"  threshold: {best_result['threshold_threshold']:.6f}")
    print(f"  derivative_threshold: {best_result['threshold_derivative_threshold']:.6f}")
    print(f"  derivative_window: {int(best_result['threshold_derivative_window'])}")
    print(f"  min_flight_time: {best_result['threshold_min_flight_time']:.6f}")
    print(f"  max_flight_time: {best_result['threshold_max_flight_time']:.6f}")
    
    print("\nDerivative Algorithm Parameters:")
    print(f"  upper_threshold: {best_result['derivative_upper_threshold']:.6f}")
    print(f"  lower_threshold: {best_result['derivative_lower_threshold']:.6f}")
    print(f"  in_air_threshold: {best_result['derivative_in_air_threshold']:.6f}")
    print(f"  min_flight_time: {best_result['derivative_min_flight_time']:.6f}")
    print(f"  max_flight_time: {best_result['derivative_max_flight_time']:.6f}")
    
    print("\nCorrelation Algorithm Parameters:")
    print(f"  buffer_size: {int(best_result['correlation_buffer_size'])}")
    print(f"  negative_frames: {int(best_result['correlation_negative_frames'])}")
    print(f"  zero_frames: {int(best_result['correlation_zero_frames'])}")
    print(f"  positive_frames: {int(best_result['correlation_positive_frames'])}")
    print(f"  correlation_threshold: {best_result['correlation_threshold']:.6f}")
    
    print("\nHybrid Algorithm Parameters:")
    print(f"  takeoff_threshold: {best_result['hybrid_takeoff_threshold']:.6f}")
    print(f"  landing_derivative_threshold: {best_result['hybrid_landing_derivative_threshold']:.6f}")
    print(f"  in_air_threshold: {best_result['hybrid_in_air_threshold']:.6f}")
    print(f"  min_flight_time: {best_result['hybrid_min_flight_time']:.6f}")
    print(f"  max_flight_time: {best_result['hybrid_max_flight_time']:.6f}")
    
    print("\nEnsemble Parameters:")
    print(f"  score_threshold: {best_result['score_threshold']:.6f}")
    print(f"  min_flight_time: {best_result['ensemble_min_flight_time']:.6f}")
    print(f"  max_flight_time: {best_result['ensemble_max_flight_time']:.6f}")
    
    print("\nWeights:")
    for condition_name in CONDITION_NAMES:
        weight_key = f"weight_{condition_name}"
        print(f"  {condition_name}: {best_result.get(weight_key, 0.0):.6f}")
    
    print(f"\nLoss: {best_result['loss']:.6f}")
    print(f"Total iterations: {best_result['total_iterations']}")
    print(f"Accepted count: {best_result['accepted_count']}")
    print(f"Acceptance rate: {best_result['acceptance_rate']:.2f}%")
    print(f"{'='*100}\n")


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive parameter optimization for ensemble algorithm"
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
    
    print("COMPREHENSIVE PARAMETER OPTIMIZATION (ENSEMBLE ALGORITHM)")
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
    langevin_output = OUTPUT_DIR / "comprehensive_ensemble_loss_langevin_results.json"
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
        "accepted_results": convert_to_json_serializable(accepted_results[:100]),  # Top 100
    }
    
    with open(langevin_output, 'w') as f:
        json.dump(langevin_output_data, f, indent=2)
    
    print(f"Saved Langevin sampling results to {langevin_output}")
    print(f"\nCompleted at: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()

