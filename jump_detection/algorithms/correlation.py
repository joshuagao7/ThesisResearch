"""Correlation-based jump detection pipeline using template matching on derivative signal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..config import (
    CORRELATION_BUFFER_SIZE_DEFAULT,
    CORRELATION_NEGATIVE_FRAMES_DEFAULT,
    CORRELATION_POSITIVE_FRAMES_DEFAULT,
    CORRELATION_THRESHOLD_DEFAULT,
    CORRELATION_ZERO_FRAMES_DEFAULT,
    DEFAULT_DATA_FILES,
    EXTRACTED_JUMPS_CORRELATION_DIR,
    MAX_FLIGHT_TIME_S,
    MIN_FLIGHT_TIME_S,
    SAMPLING_RATE,
)
from ..data import JumpWindowExport, load_dataset
from ..types import DetectionResult, Jump


@dataclass(slots=True)
class CorrelationParameters:
    buffer_size: int = CORRELATION_BUFFER_SIZE_DEFAULT
    negative_frames: int = CORRELATION_NEGATIVE_FRAMES_DEFAULT
    zero_frames: int = CORRELATION_ZERO_FRAMES_DEFAULT
    positive_frames: int = CORRELATION_POSITIVE_FRAMES_DEFAULT
    correlation_threshold: float = CORRELATION_THRESHOLD_DEFAULT
    min_flight_time: float = MIN_FLIGHT_TIME_S
    max_flight_time: float = MAX_FLIGHT_TIME_S

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def detect_correlation_jumps(
    data_file_path: str | Path,
    participant_name: Optional[str] = None,
    params: Optional[CorrelationParameters] = None,
    *,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> DetectionResult:
    """Detect jumps using correlation-based template matching on derivative signal.
    
    Args:
        data_file_path: Path to data file
        participant_name: Optional participant name
        params: CorrelationParameters instance (uses defaults if None)
        save_windows: Whether to save extracted jump windows
        output_dir: Output directory for saved windows
        
    Returns:
        DetectionResult with detected jumps
    """
    parameters = params or CorrelationParameters()
    raw_data = load_dataset(data_file_path)

    signals, jumps, metadata = _run_correlation_pipeline(raw_data, parameters)

    export_paths: list[Path] = []
    if save_windows and jumps:
        exporter = JumpWindowExport(
            output_dir=(output_dir or EXTRACTED_JUMPS_CORRELATION_DIR),
            sampling_rate=SAMPLING_RATE,
        )
        export_paths = exporter.save(raw_data, jumps, participant_name)

    metadata.update(
        {
            "parameters": parameters.as_dict(),
            "export_paths": [str(path) for path in export_paths],
            "data_file_path": str(data_file_path),
        }
    )

    pooled = signals.get("pooled", raw_data.sum(axis=1))
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


def detect_correlation_jumps_with_params(
    data_file_path: str | Path,
    buffer_size: int,
    negative_frames: int,
    zero_frames: int,
    positive_frames: int,
    correlation_threshold: float,
    participant_name: Optional[str] = None,
    *,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> DetectionResult:
    """Detect jumps using correlation algorithm with explicit parameters.
    
    Args:
        data_file_path: Path to data file
        buffer_size: Buffer size parameter
        negative_frames: Negative frames parameter
        zero_frames: Zero frames parameter
        positive_frames: Positive frames parameter
        correlation_threshold: Correlation threshold parameter
        participant_name: Optional participant name
        save_windows: Whether to save extracted jump windows
        output_dir: Output directory for saved windows
        
    Returns:
        DetectionResult with detected jumps
    """
    params = CorrelationParameters(
        buffer_size=buffer_size,
        negative_frames=negative_frames,
        zero_frames=zero_frames,
        positive_frames=positive_frames,
        correlation_threshold=correlation_threshold,
    )
    return detect_correlation_jumps(
        data_file_path,
        participant_name,
        params,
        save_windows=save_windows,
        output_dir=output_dir,
    )


def process_all_participants_correlation(
    data_files: Optional[Sequence[str | Path]] = None,
    *,
    params: Optional[CorrelationParameters] = None,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> list[DetectionResult]:
    """Process all data files with correlation algorithm.
    
    Args:
        data_files: List of data file paths (uses DEFAULT_DATA_FILES if None)
        params: CorrelationParameters instance (uses defaults if None)
        save_windows: Whether to save extracted jump windows
        output_dir: Output directory for saved windows
        
    Returns:
        List of DetectionResult objects
    """
    if data_files is None:
        data_files = DEFAULT_DATA_FILES
    
    results = []
    for data_file in data_files:
        result = detect_correlation_jumps(
            data_file,
            params=params,
            save_windows=save_windows,
            output_dir=output_dir,
        )
        results.append(result)
    return results


def _run_correlation_pipeline(
    raw_data: np.ndarray, parameters: CorrelationParameters
) -> tuple[dict[str, np.ndarray], list[Jump], dict[str, object]]:
    """Run correlation-based jump detection pipeline.
    
    Creates a template buffer encoding jump signature:
    - Negative frames (preparation phase)
    - Zero frames (transition phase)
    - Positive frames (jump phase)
    
    Then performs template matching on the derivative signal.
    
    Args:
        raw_data: Raw sensor data (n_frames, n_sensors)
        parameters: CorrelationParameters instance
        
    Returns:
        Tuple of (signals_dict, jumps_list, metadata_dict)
    """
    pooled = raw_data.sum(axis=1)
    derivative = np.gradient(pooled)
    
    # Create template buffer
    # Template structure: [negative_frames] + [zero_frames] + [positive_frames]
    template = np.zeros(parameters.buffer_size)
    
    # Negative values for preparation phase
    template[:parameters.negative_frames] = -1.0
    
    # Zero values for transition phase
    start_zero = parameters.negative_frames
    end_zero = start_zero + parameters.zero_frames
    template[start_zero:end_zero] = 0.0
    
    # Positive values for jump phase
    start_positive = end_zero
    end_positive = min(start_positive + parameters.positive_frames, parameters.buffer_size)
    template[start_positive:end_positive] = 1.0
    
    # Perform correlation (template matching) on derivative signal
    correlation_scores = np.zeros(len(derivative))
    
    for i in range(len(derivative) - parameters.buffer_size + 1):
        window = derivative[i:i + parameters.buffer_size]
        correlation = np.dot(template, window)
        correlation_scores[i + parameters.buffer_size // 2] = correlation
    
    # Find peaks above threshold
    above_threshold = correlation_scores > parameters.correlation_threshold
    
    # Find jump centers (local maxima above threshold)
    jump_centers = []
    min_frames = int(parameters.min_flight_time * SAMPLING_RATE)
    max_frames = int(parameters.max_flight_time * SAMPLING_RATE)
    
    i = 0
    while i < len(above_threshold):
        if above_threshold[i]:
            # Find the local maximum in this region
            start_idx = i
            while i < len(above_threshold) and above_threshold[i]:
                i += 1
            end_idx = i
            
            # Find peak in this region
            peak_idx = start_idx + np.argmax(correlation_scores[start_idx:end_idx])
            jump_centers.append(peak_idx)
        else:
            i += 1
    
    # Convert centers to jumps with flight time constraints
    jumps = []
    for center in jump_centers:
        # Estimate takeoff and landing around the center
        # Look for negative derivative (takeoff) before center
        # and positive derivative (landing) after center
        
        search_window = max_frames // 2
        start_search = max(0, center - search_window)
        end_search = min(len(derivative), center + search_window)
        
        # Find takeoff (negative peak before center)
        takeoff_candidates = []
        for j in range(start_search, center):
            if derivative[j] < -10:  # Significant negative derivative
                takeoff_candidates.append((j, derivative[j]))
        
        # Find landing (positive peak after center)
        landing_candidates = []
        for j in range(center, end_search):
            if derivative[j] > 10:  # Significant positive derivative
                landing_candidates.append((j, derivative[j]))
        
        if takeoff_candidates and landing_candidates:
            # Use the most negative takeoff and most positive landing
            takeoff_idx = min(takeoff_candidates, key=lambda x: x[1])[0]
            landing_idx = max(landing_candidates, key=lambda x: x[1])[0]
            
            flight_duration = landing_idx - takeoff_idx
            flight_time = flight_duration / SAMPLING_RATE
            
            # Check if flight time is within constraints
            if min_frames <= flight_duration <= max_frames:
                jumps.append(
                    Jump(
                        start=takeoff_idx,
                        end=landing_idx,
                        center=center,
                        duration=flight_duration,
                        time_diff=flight_time,
                    )
                )
        elif takeoff_candidates:
            # Only takeoff found, estimate landing
            takeoff_idx = min(takeoff_candidates, key=lambda x: x[1])[0]
            estimated_landing = takeoff_idx + (min_frames + max_frames) // 2
            if estimated_landing < len(derivative):
                jumps.append(
                    Jump(
                        start=takeoff_idx,
                        end=estimated_landing,
                        center=center,
                        duration=estimated_landing - takeoff_idx,
                        time_diff=(estimated_landing - takeoff_idx) / SAMPLING_RATE,
                    )
                )
        elif landing_candidates:
            # Only landing found, estimate takeoff
            landing_idx = max(landing_candidates, key=lambda x: x[1])[0]
            estimated_takeoff = landing_idx - (min_frames + max_frames) // 2
            if estimated_takeoff >= 0:
                jumps.append(
                    Jump(
                        start=estimated_takeoff,
                        end=landing_idx,
                        center=center,
                        duration=landing_idx - estimated_takeoff,
                        time_diff=(landing_idx - estimated_takeoff) / SAMPLING_RATE,
                    )
                )
    
    # Create signals dictionary
    signals = {
        "raw_data": raw_data,
        "pooled": pooled,
        "derivative": derivative,
        "correlation_scores": correlation_scores,
        "above_threshold": above_threshold.astype(int),
    }
    
    metadata = {
        "total_correlation_peaks": len(jump_centers),
        "valid_jumps": len(jumps),
        "template_summary": {
            "negative_frames": parameters.negative_frames,
            "zero_frames": parameters.zero_frames,
            "positive_frames": parameters.positive_frames,
        },
    }
    
    return signals, jumps, metadata

