"""Template-based jump detection pipeline using learnable templates for takeoff and landing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..config import (
    DEFAULT_DATA_FILES,
    EXTRACTED_JUMPS_TEMPLATE_DIR,
    IN_AIR_THRESHOLD_DEFAULT,
    MAX_FLIGHT_TIME_S,
    MIN_FLIGHT_TIME_S,
    SAMPLING_RATE,
)
from ..data import JumpWindowExport, load_dataset
from ..types import DetectionResult, Jump, TemplateNames
from .threshold import _apply_physics_constraints, _extract_jumps


@dataclass(slots=True)
class TemplateParameters:
    """Parameters for template-based jump detection."""
    
    # Takeoff template: factorized as time_weights × sensor_weights
    takeoff_time_weights: np.ndarray  # Shape: (template_size,)
    takeoff_sensor_weights: np.ndarray  # Shape: (n_sensors,)
    takeoff_threshold: float
    
    # Landing template: factorized as time_weights × sensor_weights
    landing_time_weights: np.ndarray  # Shape: (template_size,)
    landing_sensor_weights: np.ndarray  # Shape: (n_sensors,)
    landing_threshold: float
    
    # Time constraints
    min_flight_time: float = MIN_FLIGHT_TIME_S
    max_flight_time: float = MAX_FLIGHT_TIME_S
    
    # Optional in-air validation
    in_air_threshold: Optional[float] = None
    
    def as_dict(self) -> dict[str, object]:
        return {
            "takeoff_time_weights": self.takeoff_time_weights.tolist(),
            "takeoff_sensor_weights": self.takeoff_sensor_weights.tolist(),
            "takeoff_threshold": self.takeoff_threshold,
            "landing_time_weights": self.landing_time_weights.tolist(),
            "landing_sensor_weights": self.landing_sensor_weights.tolist(),
            "landing_threshold": self.landing_threshold,
            "min_flight_time": self.min_flight_time,
            "max_flight_time": self.max_flight_time,
            "in_air_threshold": self.in_air_threshold,
        }


def detect_template_jumps(
    data_file_path: str | Path,
    participant_name: Optional[str] = None,
    params: Optional[TemplateParameters] = None,
    *,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> DetectionResult:
    """Detect jumps using template matching."""
    if params is None:
        params = _create_default_template_parameters()
    
    raw_data = load_dataset(data_file_path)
    
    signals, jumps, metadata = _run_template_pipeline(raw_data, params)
    
    export_paths: list[Path] = []
    if save_windows and jumps:
        exporter = JumpWindowExport(
            output_dir=(output_dir or EXTRACTED_JUMPS_TEMPLATE_DIR),
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
    
    pooled = signals[TemplateNames.POOLED.value]
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


def detect_template_jumps_with_params(
    data_file_path: str | Path,
    takeoff_time_weights: np.ndarray,
    takeoff_sensor_weights: np.ndarray,
    takeoff_threshold: float,
    landing_time_weights: np.ndarray,
    landing_sensor_weights: np.ndarray,
    landing_threshold: float,
    participant_name: Optional[str] = None,
    *,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> DetectionResult:
    """Detect jumps with custom template parameters."""
    params = TemplateParameters(
        takeoff_time_weights=takeoff_time_weights,
        takeoff_sensor_weights=takeoff_sensor_weights,
        takeoff_threshold=takeoff_threshold,
        landing_time_weights=landing_time_weights,
        landing_sensor_weights=landing_sensor_weights,
        landing_threshold=landing_threshold,
    )
    return detect_template_jumps(
        data_file_path,
        participant_name,
        params,
        save_windows=save_windows,
        output_dir=output_dir,
    )


def process_all_template_participants(
    data_files: Optional[Sequence[str | Path]] = None,
    *,
    params: Optional[TemplateParameters] = None,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> list[DetectionResult]:
    """Process all participants with template algorithm."""
    results: list[DetectionResult] = []
    for data_file in data_files or DEFAULT_DATA_FILES:
        path = Path(data_file)
        participant = path.parent.name if path.parent != path else path.stem
        result = detect_template_jumps(
            path,
            participant,
            params=params,
            save_windows=save_windows,
            output_dir=output_dir,
        )
        results.append(result)
    return results


def _create_default_template_parameters() -> TemplateParameters:
    """Create default template parameters with reasonable initial values."""
    template_size = 50
    n_sensors = 48  # Default, will be adjusted based on actual data
    
    # Takeoff template: emphasize early frames (weight decreases over time)
    takeoff_time_weights = np.linspace(1.0, 0.3, template_size)
    takeoff_sensor_weights = np.ones(n_sensors) / n_sensors  # Equal weights initially
    
    # Landing template: emphasize late frames (weight increases over time)
    landing_time_weights = np.linspace(0.3, 1.0, template_size)
    landing_sensor_weights = np.ones(n_sensors) / n_sensors  # Equal weights initially
    
    return TemplateParameters(
        takeoff_time_weights=takeoff_time_weights,
        takeoff_sensor_weights=takeoff_sensor_weights,
        takeoff_threshold=0.5,
        landing_time_weights=landing_time_weights,
        landing_sensor_weights=landing_sensor_weights,
        landing_threshold=0.5,
    )


def _build_template(
    time_weights: np.ndarray, sensor_weights: np.ndarray
) -> np.ndarray:
    """Build template from factorized weights: template[i, j] = time_weights[i] * sensor_weights[j]."""
    return np.outer(time_weights, sensor_weights)


def _compute_template_correlation(
    data: np.ndarray, template: np.ndarray, position: int
) -> float:
    """Compute normalized correlation between template and data window at given position.
    
    Uses normalized dot product (cosine similarity).
    """
    template_size = template.shape[0]
    n_sensors = template.shape[1]
    
    # Extract window from data
    if position + template_size > len(data):
        return 0.0
    
    window = data[position : position + template_size, :]
    
    if window.shape[1] != n_sensors:
        # Adjust template if sensor count doesn't match
        if window.shape[1] < n_sensors:
            template = template[:, : window.shape[1]]
        else:
            # Pad template with zeros
            padded_template = np.zeros((template_size, window.shape[1]))
            padded_template[:, :n_sensors] = template
            template = padded_template
        n_sensors = window.shape[1]
    
    # Compute normalized dot product (cosine similarity)
    template_flat = template.flatten()
    window_flat = window.flatten()
    
    template_norm = np.linalg.norm(template_flat)
    window_norm = np.linalg.norm(window_flat)
    
    if template_norm == 0 or window_norm == 0:
        return 0.0
    
    correlation = np.dot(template_flat, window_flat) / (template_norm * window_norm)
    return float(correlation)


def _run_template_pipeline(
    raw_data: np.ndarray, parameters: TemplateParameters
) -> tuple[dict[str, np.ndarray], list[Jump], dict[str, object]]:
    """Run template-based detection pipeline."""
    n_frames, n_sensors = raw_data.shape
    pooled = raw_data.sum(axis=1)
    
    # Adjust sensor weights if needed
    takeoff_sensor_weights = parameters.takeoff_sensor_weights
    landing_sensor_weights = parameters.landing_sensor_weights
    
    if len(takeoff_sensor_weights) != n_sensors:
        # Adjust to match actual sensor count
        if len(takeoff_sensor_weights) < n_sensors:
            # Pad with zeros
            padded = np.zeros(n_sensors)
            padded[: len(takeoff_sensor_weights)] = takeoff_sensor_weights
            takeoff_sensor_weights = padded
        else:
            # Truncate
            takeoff_sensor_weights = takeoff_sensor_weights[:n_sensors]
    
    if len(landing_sensor_weights) != n_sensors:
        if len(landing_sensor_weights) < n_sensors:
            padded = np.zeros(n_sensors)
            padded[: len(landing_sensor_weights)] = landing_sensor_weights
            landing_sensor_weights = padded
        else:
            landing_sensor_weights = landing_sensor_weights[:n_sensors]
    
    # Build templates
    template_size = len(parameters.takeoff_time_weights)
    takeoff_template = _build_template(parameters.takeoff_time_weights, takeoff_sensor_weights)
    landing_template = _build_template(parameters.landing_time_weights, landing_sensor_weights)
    
    # Compute correlations at each position
    takeoff_correlations = np.zeros(n_frames)
    landing_correlations = np.zeros(n_frames)
    
    for i in range(n_frames - template_size + 1):
        takeoff_correlations[i] = _compute_template_correlation(raw_data, takeoff_template, i)
        landing_correlations[i] = _compute_template_correlation(raw_data, landing_template, i)
    
    # Threshold correlations to find events
    takeoff_mask = (takeoff_correlations >= parameters.takeoff_threshold).astype(int)
    landing_mask = (landing_correlations >= parameters.landing_threshold).astype(int)
    
    # Pair takeoff and landing events
    takeoff_landing_pairs = _pair_takeoff_landing(
        takeoff_mask,
        landing_mask,
        min_time=parameters.min_flight_time,
        max_time=parameters.max_flight_time,
    )
    
    # Optionally filter by in-air validation
    valid_pairs = takeoff_landing_pairs
    if parameters.in_air_threshold is not None:
        in_air = (pooled < parameters.in_air_threshold).astype(int)
        valid_pairs = _filter_pairs_with_flight(takeoff_landing_pairs, in_air)
    
    # Create Jump objects
    jumps = [
        Jump(
            start=pair.takeoff_idx,
            end=pair.landing_idx,
            center=(pair.takeoff_idx + pair.landing_idx) // 2,
            duration=pair.landing_idx - pair.takeoff_idx,
            time_diff=pair.time_diff,
        )
        for pair in valid_pairs
    ]
    
    # Build signals dictionary
    signals = {
        TemplateNames.RAW_DATA.value: raw_data,
        TemplateNames.POOLED.value: pooled,
        TemplateNames.TAKEOFF_CORRELATION.value: takeoff_correlations,
        TemplateNames.LANDING_CORRELATION.value: landing_correlations,
        TemplateNames.TAKEOFF_MASK.value: takeoff_mask,
        TemplateNames.LANDING_MASK.value: landing_mask,
    }
    
    metadata = {
        "template_size": template_size,
        "n_sensors": n_sensors,
        "total_takeoff_events": int(takeoff_mask.sum()),
        "total_landing_events": int(landing_mask.sum()),
        "total_pairs": len(takeoff_landing_pairs),
        "valid_pairs": len(valid_pairs),
    }
    
    return signals, jumps, metadata


@dataclass(slots=True)
class _TakeoffLandingPair:
    takeoff_idx: int
    landing_idx: int
    time_diff: float


def _pair_takeoff_landing(
    takeoff_mask: np.ndarray,
    landing_mask: np.ndarray,
    *,
    min_time: float,
    max_time: float,
) -> list[_TakeoffLandingPair]:
    """Pair takeoff and landing events with time constraints."""
    min_frames = int(min_time * SAMPLING_RATE)
    max_frames = int(max_time * SAMPLING_RATE)

    pairs: list[_TakeoffLandingPair] = []

    current_takeoff: int | None = None

    for idx in range(len(takeoff_mask)):
        if current_takeoff is None:
            if takeoff_mask[idx] == 1:
                current_takeoff = idx
        else:
            # If we find a new takeoff before finding a landing, update takeoff
            if takeoff_mask[idx] == 1:
                current_takeoff = idx
            elif landing_mask[idx] == 1:
                frame_diff = idx - current_takeoff
                if min_frames <= frame_diff <= max_frames:
                    pairs.append(
                        _TakeoffLandingPair(
                            takeoff_idx=current_takeoff,
                            landing_idx=idx,
                            time_diff=frame_diff / SAMPLING_RATE,
                        )
                    )
                current_takeoff = None

    return pairs


def _filter_pairs_with_flight(
    pairs: list[_TakeoffLandingPair], in_air: np.ndarray
) -> list[_TakeoffLandingPair]:
    """Filter pairs to ensure person was in air between takeoff and landing."""
    return [
        pair
        for pair in pairs
        if in_air[pair.takeoff_idx : pair.landing_idx + 1].any()
    ]

