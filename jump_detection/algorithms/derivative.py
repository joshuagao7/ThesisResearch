"""Derivative-based jump detection pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ..config import (
    DEFAULT_DATA_FILES,
    DERIVATIVE_LOWER_DEFAULT,
    DERIVATIVE_UPPER_DEFAULT,
    EXTRACTED_JUMPS_DERIVATIVE_DIR,
    IN_AIR_THRESHOLD_DEFAULT,
    MAX_FLIGHT_TIME_S,
    MIN_FLIGHT_TIME_S,
    SAMPLING_RATE,
)
from ..data import JumpWindowExport, load_dataset
from ..types import DerivativeNames, DetectionResult, Jump


@dataclass(slots=True)
class DerivativeParameters:
    upper_threshold: float = DERIVATIVE_UPPER_DEFAULT
    lower_threshold: float = DERIVATIVE_LOWER_DEFAULT
    in_air_threshold: float = IN_AIR_THRESHOLD_DEFAULT
    min_flight_time: float = MIN_FLIGHT_TIME_S
    max_flight_time: float = MAX_FLIGHT_TIME_S

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def detect_derivative_jumps(
    data_file_path: str | Path,
    participant_name: Optional[str] = None,
    params: Optional[DerivativeParameters] = None,
    *,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> DetectionResult:
    parameters = params or DerivativeParameters()
    raw_data = load_dataset(data_file_path)

    signals, jumps, metadata = _run_derivative_pipeline(raw_data, parameters)

    export_paths: list[Path] = []
    if save_windows and jumps:
        exporter = JumpWindowExport(
            output_dir=(output_dir or EXTRACTED_JUMPS_DERIVATIVE_DIR),
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

    pooled = signals[DerivativeNames.POOLED.value]
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


def detect_derivative_jumps_with_params(
    data_file_path: str | Path,
    derivative_upper_threshold: float,
    derivative_lower_threshold: float,
    participant_name: Optional[str] = None,
    *,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> DetectionResult:
    params = DerivativeParameters(
        upper_threshold=derivative_upper_threshold,
        lower_threshold=derivative_lower_threshold,
    )
    return detect_derivative_jumps(
        data_file_path,
        participant_name,
        params,
        save_windows=save_windows,
        output_dir=output_dir,
    )


def process_all_derivative_participants(
    data_files: Optional[Sequence[str | Path]] = None,
    *,
    params: Optional[DerivativeParameters] = None,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> list[DetectionResult]:
    results: list[DetectionResult] = []
    for data_file in data_files or DEFAULT_DATA_FILES:
        path = Path(data_file)
        participant = path.parent.name if path.parent != path else path.stem
        result = detect_derivative_jumps(
            path,
            participant,
            params=params,
            save_windows=save_windows,
            output_dir=output_dir,
        )
        results.append(result)
    return results


def _run_derivative_pipeline(
    raw_data: np.ndarray, parameters: DerivativeParameters
) -> tuple[dict[str, np.ndarray], list[Jump], dict[str, object]]:
    pooled = raw_data.sum(axis=1)
    derivative = np.gradient(pooled)

    upper_mask = (derivative > parameters.upper_threshold).astype(int)
    lower_mask = (derivative < parameters.lower_threshold).astype(int)

    derivative_pairs = _pair_threshold_crossings(
        lower_mask,
        upper_mask,
        min_time=parameters.min_flight_time,
        max_time=parameters.max_flight_time,
    )

    in_air = (pooled < parameters.in_air_threshold).astype(int)
    valid_pairs = _filter_pairs_with_flight(derivative_pairs, in_air)

    jumps = [
        Jump(
            start=pair.lower_idx,
            end=pair.upper_idx,
            center=(pair.lower_idx + pair.upper_idx) // 2,
            duration=pair.upper_idx - pair.lower_idx,
            time_diff=pair.time_diff,
        )
        for pair in valid_pairs
        if pair.upper_idx > pair.lower_idx  # Filter out zero-duration jumps
    ]

    pair_indicator = np.zeros_like(pooled)
    for pair in derivative_pairs:
        pair_indicator[[pair.lower_idx, pair.upper_idx]] = 1

    valid_pair_indicator = np.zeros_like(pooled)
    for pair in valid_pairs:
        valid_pair_indicator[[pair.lower_idx, pair.upper_idx]] = 1

    signals = {
        DerivativeNames.RAW_DATA.value: raw_data,
        DerivativeNames.POOLED.value: pooled,
        DerivativeNames.DERIVATIVE.value: derivative,
        DerivativeNames.UPPER_MASK.value: upper_mask,
        DerivativeNames.LOWER_MASK.value: lower_mask,
        DerivativeNames.PAIR_INDICATOR.value: pair_indicator,
        DerivativeNames.IN_AIR.value: in_air,
        DerivativeNames.VALID_PAIR_INDICATOR.value: valid_pair_indicator,
    }

    metadata = {
        "total_upper_crossings": int(upper_mask.sum()),
        "total_lower_crossings": int(lower_mask.sum()),
        "total_pairs": len(derivative_pairs),
        "valid_pairs": len(valid_pairs),
    }

    return signals, jumps, metadata


@dataclass(slots=True)
class _DerivativePair:
    lower_idx: int
    upper_idx: int
    time_diff: float


def _pair_threshold_crossings(
    lower_mask: np.ndarray,
    upper_mask: np.ndarray,
    *,
    min_time: float,
    max_time: float,
) -> list[_DerivativePair]:
    min_frames = int(min_time * SAMPLING_RATE)
    max_frames = int(max_time * SAMPLING_RATE)

    pairs: list[_DerivativePair] = []

    current_lower: int | None = None

    for idx in range(len(lower_mask)):
        if current_lower is None:
            if lower_mask[idx] == 1:
                current_lower = idx
        else:
            if lower_mask[idx] == 1:
                current_lower = idx
            elif upper_mask[idx] == 1:
                frame_diff = idx - current_lower
                # Ensure we have a positive duration (upper_idx > lower_idx)
                if frame_diff > 0 and min_frames <= frame_diff <= max_frames:
                    pairs.append(
                        _DerivativePair(
                            lower_idx=current_lower,
                            upper_idx=idx,
                            time_diff=frame_diff / SAMPLING_RATE,
                        )
                    )
                current_lower = None

    return pairs


def _filter_pairs_with_flight(
    pairs: Iterable[_DerivativePair], in_air: np.ndarray
) -> list[_DerivativePair]:
    return [pair for pair in pairs if in_air[pair.lower_idx : pair.upper_idx + 1].any()]


