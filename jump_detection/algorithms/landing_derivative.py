"""Landing-only derivative-based jump detection pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..analysis.precise import calculate_precise_jump_boundaries
from ..config import (
    DEFAULT_DATA_FILES,
    DERIVATIVE_UPPER_DEFAULT,
    EXTRACTED_JUMPS_LANDING_DERIVATIVE_DIR,
    IN_AIR_THRESHOLD_DEFAULT,
    LANDING_DERIVATIVE_CENTER_OFFSET_DEFAULT,
    LANDING_DERIVATIVE_SEARCH_WINDOW_DEFAULT,
    MAX_FLIGHT_TIME_S,
    MIN_FLIGHT_TIME_S,
    SAMPLING_RATE,
)
from ..data import JumpWindowExport, load_dataset
from ..types import DetectionResult, Jump, LandingDerivativeNames


@dataclass(slots=True)
class LandingDerivativeParameters:
    landing_threshold: float = DERIVATIVE_UPPER_DEFAULT  # Positive threshold for landing detection
    center_offset: int = LANDING_DERIVATIVE_CENTER_OFFSET_DEFAULT
    search_window: int = LANDING_DERIVATIVE_SEARCH_WINDOW_DEFAULT
    in_air_threshold: Optional[float] = IN_AIR_THRESHOLD_DEFAULT
    min_flight_time: float = MIN_FLIGHT_TIME_S
    max_flight_time: float = MAX_FLIGHT_TIME_S

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def detect_landing_derivative_jumps(
    data_file_path: str | Path,
    participant_name: Optional[str] = None,
    params: Optional[LandingDerivativeParameters] = None,
    *,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> DetectionResult:
    parameters = params or LandingDerivativeParameters()
    raw_data = load_dataset(data_file_path)

    signals, jumps, metadata = _run_landing_derivative_pipeline(raw_data, parameters)

    export_paths: list[Path] = []
    if save_windows and jumps:
        exporter = JumpWindowExport(
            output_dir=(output_dir or EXTRACTED_JUMPS_LANDING_DERIVATIVE_DIR),
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

    pooled = signals[LandingDerivativeNames.POOLED.value]
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


def detect_landing_derivative_jumps_with_params(
    data_file_path: str | Path,
    landing_threshold: float,
    center_offset: int = LANDING_DERIVATIVE_CENTER_OFFSET_DEFAULT,
    participant_name: Optional[str] = None,
    *,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> DetectionResult:
    params = LandingDerivativeParameters(
        landing_threshold=landing_threshold,
        center_offset=center_offset,
    )
    return detect_landing_derivative_jumps(
        data_file_path,
        participant_name,
        params,
        save_windows=save_windows,
        output_dir=output_dir,
    )


def process_all_landing_derivative_participants(
    data_files: Optional[Sequence[str | Path]] = None,
    *,
    params: Optional[LandingDerivativeParameters] = None,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> list[DetectionResult]:
    results: list[DetectionResult] = []
    for data_file in data_files or DEFAULT_DATA_FILES:
        path = Path(data_file)
        participant = path.parent.name if path.parent != path else path.stem
        result = detect_landing_derivative_jumps(
            path,
            participant,
            params=params,
            save_windows=save_windows,
            output_dir=output_dir,
        )
        results.append(result)
    return results


def _run_landing_derivative_pipeline(
    raw_data: np.ndarray, parameters: LandingDerivativeParameters
) -> tuple[dict[str, np.ndarray], list[Jump], dict[str, object]]:
    pooled = raw_data.sum(axis=1)
    derivative = np.gradient(pooled)

    # Detect landings: positive derivative above threshold (landing = going up)
    landing_mask = (derivative > parameters.landing_threshold).astype(int)

    # Find landing indices
    landing_indices = _detect_landings(landing_mask)

    # For each landing, calculate center and get precise boundaries
    jumps: list[Jump] = []
    valid_landing_indices: list[int] = []

    for landing_idx in landing_indices:
        # Calculate rough center: 10 frames to the left
        center = max(0, landing_idx - parameters.center_offset)

        # Get precise jump boundaries
        try:
            precise_data = calculate_precise_jump_boundaries(
                pooled, center, parameters.search_window
            )
            precise_start = precise_data["precise_start"]
            precise_end = precise_data["precise_end"]
            precise_center = precise_data["precise_center"]
            precise_duration = precise_data["precise_duration"]

            # Optional: Filter by flight time constraints
            flight_time = precise_duration / SAMPLING_RATE
            if (
                parameters.min_flight_time
                <= flight_time
                <= parameters.max_flight_time
            ):
                jumps.append(
                    Jump(
                        start=precise_start,
                        end=precise_end,
                        center=precise_center,
                        duration=precise_duration,
                        time_diff=flight_time,
                    )
                )
                valid_landing_indices.append(landing_idx)
        except (ValueError, IndexError):
            # Fallback: use rough boundaries if precise calculation fails
            rough_start = max(0, center - parameters.search_window)
            rough_end = min(len(pooled), center + parameters.search_window)
            rough_duration = rough_end - rough_start
            flight_time = rough_duration / SAMPLING_RATE

            if (
                parameters.min_flight_time
                <= flight_time
                <= parameters.max_flight_time
            ):
                jumps.append(
                    Jump(
                        start=rough_start,
                        end=rough_end,
                        center=center,
                        duration=rough_duration,
                        time_diff=flight_time,
                    )
                )
                valid_landing_indices.append(landing_idx)

    # Optional: Filter by in-air threshold if specified
    if parameters.in_air_threshold is not None:
        in_air = (pooled < parameters.in_air_threshold).astype(int)
        filtered_jumps: list[Jump] = []
        for jump in jumps:
            # Check if there's an in-air period during the jump
            if in_air[jump.start : jump.end + 1].any():
                filtered_jumps.append(jump)
        jumps = filtered_jumps
    else:
        in_air = np.zeros_like(pooled, dtype=int)

    # Create indicator arrays
    landing_indicator = np.zeros_like(pooled)
    for idx in landing_indices:
        landing_indicator[idx] = 1

    valid_landing_indicator = np.zeros_like(pooled)
    for idx in valid_landing_indices:
        valid_landing_indicator[idx] = 1

    signals = {
        LandingDerivativeNames.RAW_DATA.value: raw_data,
        LandingDerivativeNames.POOLED.value: pooled,
        LandingDerivativeNames.DERIVATIVE.value: derivative,
        LandingDerivativeNames.LANDING_MASK.value: landing_mask,
        LandingDerivativeNames.LANDING_INDICATOR.value: landing_indicator,
        LandingDerivativeNames.IN_AIR.value: in_air,
    }

    metadata = {
        "total_landing_events": len(landing_indices),
        "valid_jumps": len(jumps),
        "center_offset": parameters.center_offset,
        "search_window": parameters.search_window,
        "landing_threshold": parameters.landing_threshold,
    }

    return signals, jumps, metadata


def _detect_landings(
    landing_mask: np.ndarray, min_separation: int = 20
) -> list[int]:
    """Find landing indices with minimum separation between detections."""
    landing_indices: list[int] = []
    last_landing: int | None = None

    for idx in range(len(landing_mask)):
        if landing_mask[idx] == 1:
            if last_landing is None or (idx - last_landing) >= min_separation:
                landing_indices.append(idx)
                last_landing = idx

    return landing_indices

