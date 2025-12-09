"""Threshold-based jump detection pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ..config import (
    DERIVATIVE_THRESHOLD_DEFAULT,
    EXTRACTION_WINDOW_DEFAULT,
    EXTRACTED_JUMPS_DIR,
    MAX_FLIGHT_TIME_S,
    MIN_FLIGHT_TIME_S,
    SAMPLING_RATE,
    THRESHOLD_DEFAULT,
    DEFAULT_DATA_FILES,
)
from ..data import JumpWindowExport, load_dataset
from ..types import DetectionResult, Jump, ThresholdNames


@dataclass(slots=True)
class ThresholdParameters:
    threshold: float = THRESHOLD_DEFAULT
    derivative_threshold: float = DERIVATIVE_THRESHOLD_DEFAULT
    min_flight_time: float = MIN_FLIGHT_TIME_S
    max_flight_time: float = MAX_FLIGHT_TIME_S
    extraction_window: int = EXTRACTION_WINDOW_DEFAULT

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def detect_threshold_jumps(
    data_file_path: str | Path,
    participant_name: Optional[str] = None,
    params: Optional[ThresholdParameters] = None,
    *,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> DetectionResult:
    parameters = params or ThresholdParameters()
    raw_data = load_dataset(data_file_path)

    signals, jumps, metadata = _run_threshold_pipeline(raw_data, parameters)

    export_paths: list[Path] = []
    if save_windows and jumps:
        exporter = JumpWindowExport(
            output_dir=(output_dir or EXTRACTED_JUMPS_DIR),
            window_size=parameters.extraction_window,
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

    average = signals[ThresholdNames.AVERAGE.value]
    result = DetectionResult(
        participant_name=participant_name,
        sampling_rate=SAMPLING_RATE,
        raw_data=raw_data,
        pooled_data=average,
        signals=signals,
        jumps=jumps,
        metadata=metadata,
    )
    return result


def detect_threshold_jumps_with_params(
    data_file_path: str | Path,
    threshold: float,
    derivative_threshold: float,
    participant_name: Optional[str] = None,
    *,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> DetectionResult:
    params = ThresholdParameters(
        threshold=threshold,
        derivative_threshold=derivative_threshold,
    )
    return detect_threshold_jumps(
        data_file_path,
        participant_name,
        params,
        save_windows=save_windows,
        output_dir=output_dir,
    )


def process_all_threshold_participants(
    data_files: Optional[Sequence[str | Path]] = None,
    *,
    params: Optional[ThresholdParameters] = None,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> list[DetectionResult]:
    results: list[DetectionResult] = []
    for data_file in data_files or DEFAULT_DATA_FILES:
        path = Path(data_file)
        participant = path.parent.name if path.parent != path else path.stem
        result = detect_threshold_jumps(
            path,
            participant,
            params=params,
            save_windows=save_windows,
            output_dir=output_dir,
        )
        results.append(result)
    return results


def _run_threshold_pipeline(
    raw_data: np.ndarray, parameters: ThresholdParameters
) -> tuple[dict[str, np.ndarray], list[Jump], dict[str, object]]:
    average = raw_data.sum(axis=1)
    threshold_mask = (average < parameters.threshold).astype(int)

    derivative = np.gradient(average)
    derivative_binary = (
        (derivative > parameters.derivative_threshold)
        | (derivative < -parameters.derivative_threshold)
    ).astype(int)

    physics_filtered = _apply_physics_constraints(
        threshold_mask,
        parameters.min_flight_time,
        parameters.max_flight_time,
    )

    jumps = _extract_jumps(physics_filtered)

    signals = {
        ThresholdNames.RAW_DATA.value: raw_data,
        ThresholdNames.AVERAGE.value: average,
        ThresholdNames.THRESHOLD_MASK.value: threshold_mask,
        ThresholdNames.PHYSICS_FILTERED.value: physics_filtered,
        ThresholdNames.DERIVATIVE.value: derivative,
        ThresholdNames.DERIVATIVE_BINARY.value: derivative_binary,
    }

    metadata = {
        "min_flight_frames": int(parameters.min_flight_time * SAMPLING_RATE),
        "max_flight_frames": int(parameters.max_flight_time * SAMPLING_RATE),
    }

    return signals, jumps, metadata


def _apply_physics_constraints(
    mask: np.ndarray, min_time: float, max_time: float
) -> np.ndarray:
    min_frames = int(min_time * SAMPLING_RATE)
    max_frames = int(max_time * SAMPLING_RATE)
    filtered = mask.copy()

    for start, end in _iter_segments(mask):
        length = end - start
        if length < min_frames or length > max_frames:
            filtered[start:end] = 0

    return filtered


def _extract_jumps(mask: np.ndarray) -> list[Jump]:
    return [
        Jump(start=start, end=end, center=(start + end) // 2, duration=end - start)
        for start, end in _iter_segments(mask)
    ]


def _iter_segments(mask: np.ndarray) -> Iterable[tuple[int, int]]:
    in_segment = False
    start = 0
    for idx, value in enumerate(mask):
        if value == 1 and not in_segment:
            in_segment = True
            start = idx
        elif value == 0 and in_segment:
            in_segment = False
            yield start, idx
    if in_segment:
        yield start, len(mask)


