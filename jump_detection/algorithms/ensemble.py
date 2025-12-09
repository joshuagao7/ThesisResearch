"""Ensemble voting-based jump detection pipeline combining all algorithms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..config import (
    DEFAULT_DATA_FILES,
    EXTRACTED_JUMPS_ENSEMBLE_DIR,
    MAX_FLIGHT_TIME_S,
    MIN_FLIGHT_TIME_S,
    SAMPLING_RATE,
)
from ..data import JumpWindowExport, load_dataset
from ..types import (
    DetectionResult,
    DerivativeNames,
    EnsembleNames,
    HybridNames,
    Jump,
    ThresholdNames,
)
from .correlation import CorrelationParameters, _run_correlation_pipeline
from .derivative import DerivativeParameters, _run_derivative_pipeline
from .hybrid import HybridParameters, _run_hybrid_pipeline
from .threshold import ThresholdParameters, _run_threshold_pipeline, _apply_physics_constraints, _extract_jumps


@dataclass(slots=True)
class EnsembleParameters:
    """Parameters for ensemble voting algorithm."""
    
    # Individual algorithm parameters
    threshold_params: ThresholdParameters
    derivative_params: DerivativeParameters
    correlation_params: CorrelationParameters
    hybrid_params: HybridParameters
    
    # Weights for each condition
    weights: dict[str, float]
    
    # Final score threshold
    score_threshold: float
    
    # Physics constraints for final jump mask
    min_flight_time: float = MIN_FLIGHT_TIME_S
    max_flight_time: float = MAX_FLIGHT_TIME_S
    
    def as_dict(self) -> dict[str, object]:
        return {
            "threshold_params": self.threshold_params.as_dict(),
            "derivative_params": self.derivative_params.as_dict(),
            "correlation_params": self.correlation_params.as_dict(),
            "hybrid_params": self.hybrid_params.as_dict(),
            "weights": self.weights,
            "score_threshold": self.score_threshold,
            "min_flight_time": self.min_flight_time,
            "max_flight_time": self.max_flight_time,
        }


def detect_ensemble_jumps(
    data_file_path: str | Path,
    participant_name: Optional[str] = None,
    params: Optional[EnsembleParameters] = None,
    *,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> DetectionResult:
    """Detect jumps using ensemble voting from all algorithms."""
    if params is None:
        # Create default parameters with default weights
        params = _create_default_ensemble_parameters()
    
    raw_data = load_dataset(data_file_path)
    
    signals, jumps, metadata = _run_ensemble_pipeline(raw_data, params)
    
    export_paths: list[Path] = []
    if save_windows and jumps:
        exporter = JumpWindowExport(
            output_dir=(output_dir or EXTRACTED_JUMPS_ENSEMBLE_DIR),
            sampling_rate=SAMPLING_RATE,
        )
        export_paths = exporter.save(raw_data, jumps, participant_name)
    
    metadata.update(
        {
            "parameters": params.as_dict(),
            "export_paths": [str(path) for path in export_paths],
            "data_file_path": str(data_file_path),
        }
    )
    
    pooled = signals[EnsembleNames.POOLED.value]
    result = DetectionResult(
        participant_name=participant_name,
        sampling_rate=SAMPLING_RATE,
        raw_data=raw_data,
        pooled_data=pooled,
        signals=signals,
        jumps=jumps,
        metadata=metadata,
    )
    return result


def detect_ensemble_jumps_with_params(
    data_file_path: str | Path,
    weights: dict[str, float],
    score_threshold: float,
    threshold_params: Optional[ThresholdParameters] = None,
    derivative_params: Optional[DerivativeParameters] = None,
    correlation_params: Optional[CorrelationParameters] = None,
    hybrid_params: Optional[HybridParameters] = None,
    participant_name: Optional[str] = None,
    *,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> DetectionResult:
    """Detect jumps with custom weights and threshold."""
    from .correlation import CorrelationParameters as DefaultCorrelationParams
    from .derivative import DerivativeParameters as DefaultDerivativeParams
    from .hybrid import HybridParameters as DefaultHybridParams
    from .threshold import ThresholdParameters as DefaultThresholdParams
    
    params = EnsembleParameters(
        threshold_params=threshold_params or DefaultThresholdParams(),
        derivative_params=derivative_params or DefaultDerivativeParams(),
        correlation_params=correlation_params or DefaultCorrelationParams(),
        hybrid_params=hybrid_params or DefaultHybridParams(),
        weights=weights,
        score_threshold=score_threshold,
    )
    return detect_ensemble_jumps(
        data_file_path,
        participant_name,
        params,
        save_windows=save_windows,
        output_dir=output_dir,
    )


def process_all_ensemble_participants(
    data_files: Optional[Sequence[str | Path]] = None,
    *,
    params: Optional[EnsembleParameters] = None,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> list[DetectionResult]:
    """Process all participants with ensemble algorithm."""
    results: list[DetectionResult] = []
    for data_file in data_files or DEFAULT_DATA_FILES:
        path = Path(data_file)
        participant = path.parent.name if path.parent != path else path.stem
        result = detect_ensemble_jumps(
            path,
            participant,
            params=params,
            save_windows=save_windows,
            output_dir=output_dir,
        )
        results.append(result)
    return results


def _create_default_ensemble_parameters() -> EnsembleParameters:
    """Create default ensemble parameters with equal weights."""
    from .correlation import CorrelationParameters
    from .derivative import DerivativeParameters
    from .hybrid import HybridParameters
    from .threshold import ThresholdParameters
    
    # Default weights (all equal)
    default_weights = {
        # Threshold conditions
        "threshold_mask": 1.0,
        "derivative_binary": 1.0,
        "physics_filtered": 1.0,
        # Derivative conditions
        "upper_mask": 1.0,
        "lower_mask": 1.0,
        "derivative_in_air": 1.0,
        # Correlation condition
        "correlation_above_threshold": 1.0,
        # Hybrid conditions
        "takeoff_mask": 1.0,
        "landing_mask": 1.0,
        "hybrid_in_air": 1.0,
    }
    
    return EnsembleParameters(
        threshold_params=ThresholdParameters(),
        derivative_params=DerivativeParameters(),
        correlation_params=CorrelationParameters(),
        hybrid_params=HybridParameters(),
        weights=default_weights,
        score_threshold=5.0,  # Default threshold
    )


def _run_ensemble_pipeline(
    raw_data: np.ndarray, parameters: EnsembleParameters
) -> tuple[dict[str, np.ndarray], list[Jump], dict[str, object]]:
    """Run ensemble voting pipeline."""
    # Run all 4 algorithms
    threshold_signals, _, _ = _run_threshold_pipeline(raw_data, parameters.threshold_params)
    derivative_signals, _, _ = _run_derivative_pipeline(raw_data, parameters.derivative_params)
    correlation_signals, _, _ = _run_correlation_pipeline(raw_data, parameters.correlation_params)
    hybrid_signals, _, _ = _run_hybrid_pipeline(raw_data, parameters.hybrid_params)
    
    # Extract pooled signal (use from any algorithm, they should all be the same)
    pooled = derivative_signals[DerivativeNames.POOLED.value]
    n_frames = len(pooled)
    
    # Extract condition signals
    condition_signals: dict[str, np.ndarray] = {}
    
    # Threshold algorithm conditions
    condition_signals["threshold_mask"] = threshold_signals[ThresholdNames.THRESHOLD_MASK.value]
    condition_signals["derivative_binary"] = threshold_signals[ThresholdNames.DERIVATIVE_BINARY.value]
    condition_signals["physics_filtered"] = threshold_signals[ThresholdNames.PHYSICS_FILTERED.value]
    
    # Derivative algorithm conditions
    condition_signals["upper_mask"] = derivative_signals[DerivativeNames.UPPER_MASK.value]
    condition_signals["lower_mask"] = derivative_signals[DerivativeNames.LOWER_MASK.value]
    condition_signals["derivative_in_air"] = derivative_signals[DerivativeNames.IN_AIR.value]
    
    # Correlation algorithm condition
    correlation_signal = correlation_signals["correlation"]
    correlation_above_threshold = (
        (correlation_signal > parameters.correlation_params.correlation_threshold)
        & ~np.isnan(correlation_signal)
    ).astype(int)
    condition_signals["correlation_above_threshold"] = correlation_above_threshold
    
    # Hybrid algorithm conditions
    condition_signals["takeoff_mask"] = hybrid_signals[HybridNames.TAKEOFF_MASK.value]
    condition_signals["landing_mask"] = hybrid_signals[HybridNames.LANDING_MASK.value]
    condition_signals["hybrid_in_air"] = hybrid_signals[HybridNames.IN_AIR.value]
    
    # Compute weighted score at each frame
    score = np.zeros(n_frames, dtype=float)
    for condition_name, condition_signal in condition_signals.items():
        weight = parameters.weights.get(condition_name, 0.0)
        # Ensure condition_signal is the right length
        if len(condition_signal) == n_frames:
            score += weight * condition_signal.astype(float)
    
    # Create jump mask from thresholded score
    jump_mask = (score >= parameters.score_threshold).astype(int)
    
    # Apply physics constraints
    physics_filtered_mask = _apply_physics_constraints(
        jump_mask,
        parameters.min_flight_time,
        parameters.max_flight_time,
    )
    
    # Extract jumps from mask
    jumps = _extract_jumps(physics_filtered_mask)
    
    # Build signals dictionary
    signals = {
        EnsembleNames.RAW_DATA.value: raw_data,
        EnsembleNames.POOLED.value: pooled,
        EnsembleNames.SCORE.value: score,
        EnsembleNames.JUMP_MASK.value: jump_mask,
        EnsembleNames.PHYSICS_FILTERED_MASK.value: physics_filtered_mask,
    }
    
    # Add all condition signals
    for condition_name, condition_signal in condition_signals.items():
        signals[f"condition_{condition_name}"] = condition_signal
    
    # Add individual algorithm signals for reference
    signals["threshold_average"] = threshold_signals[ThresholdNames.AVERAGE.value]
    signals["threshold_derivative"] = threshold_signals[ThresholdNames.DERIVATIVE.value]
    signals["derivative_derivative"] = derivative_signals[DerivativeNames.DERIVATIVE.value]
    signals["correlation_signal"] = correlation_signal
    signals["hybrid_derivative"] = hybrid_signals[HybridNames.DERIVATIVE.value]
    
    metadata = {
        "total_conditions": len(condition_signals),
        "active_conditions": sum(1 for name in condition_signals.keys() if parameters.weights.get(name, 0.0) != 0.0),
        "min_flight_frames": int(parameters.min_flight_time * SAMPLING_RATE),
        "max_flight_frames": int(parameters.max_flight_time * SAMPLING_RATE),
    }
    
    return signals, jumps, metadata

