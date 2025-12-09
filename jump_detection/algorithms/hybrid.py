"""Hybrid-based jump detection pipeline combining threshold takeoff with derivative landing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ..config import (
    DEFAULT_DATA_FILES,
    EXTRACTED_JUMPS_HYBRID_DIR,
    IN_AIR_THRESHOLD_DEFAULT,
    MAX_FLIGHT_TIME_S,
    MIN_FLIGHT_TIME_S,
    SAMPLING_RATE,
    THRESHOLD_DEFAULT,
)
from ..data import JumpWindowExport, load_dataset
from ..types import DetectionResult, HybridNames, Jump


@dataclass(slots=True)
class HybridParameters:
    takeoff_threshold: float = THRESHOLD_DEFAULT
    landing_derivative_threshold: float = 15.0
    in_air_threshold: float = IN_AIR_THRESHOLD_DEFAULT
    min_flight_time: float = MIN_FLIGHT_TIME_S
    max_flight_time: float = MAX_FLIGHT_TIME_S

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def detect_hybrid_jumps(
    data_file_path: str | Path,
    participant_name: Optional[str] = None,
    params: Optional[HybridParameters] = None,
    *,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> DetectionResult:
    parameters = params or HybridParameters()
    raw_data = load_dataset(data_file_path)

    signals, jumps, metadata = _run_hybrid_pipeline(raw_data, parameters)

    export_paths: list[Path] = []
    if save_windows and jumps:
        exporter = JumpWindowExport(
            output_dir=(output_dir or EXTRACTED_JUMPS_HYBRID_DIR),
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

    pooled = signals[HybridNames.POOLED.value]
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


def detect_hybrid_jumps_with_params(
    data_file_path: str | Path,
    takeoff_threshold: float,
    landing_derivative_threshold: float,
    in_air_threshold: float,
    participant_name: Optional[str] = None,
    *,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> DetectionResult:
    params = HybridParameters(
        takeoff_threshold=takeoff_threshold,
        landing_derivative_threshold=landing_derivative_threshold,
        in_air_threshold=in_air_threshold,
    )
    return detect_hybrid_jumps(
        data_file_path,
        participant_name,
        params,
        save_windows=save_windows,
        output_dir=output_dir,
    )


def process_all_hybrid_participants(
    data_files: Optional[Sequence[str | Path]] = None,
    *,
    params: Optional[HybridParameters] = None,
    save_windows: bool = False,
    output_dir: Path | None = None,
) -> list[DetectionResult]:
    results: list[DetectionResult] = []
    for data_file in data_files or DEFAULT_DATA_FILES:
        path = Path(data_file)
        participant = path.parent.name if path.parent != path else path.stem
        result = detect_hybrid_jumps(
            path,
            participant,
            params=params,
            save_windows=save_windows,
            output_dir=output_dir,
        )
        results.append(result)
    return results


def _run_hybrid_pipeline(
    raw_data: np.ndarray, parameters: HybridParameters
) -> tuple[dict[str, np.ndarray], list[Jump], dict[str, object]]:
    pooled = raw_data.sum(axis=1)
    derivative = np.gradient(pooled)

    # Detect takeoff events: downward threshold crossings
    # Look for transitions where pooled[i-1] >= threshold AND pooled[i] < threshold
    takeoff_mask = np.zeros_like(pooled, dtype=int)
    for i in range(1, len(pooled)):
        if pooled[i - 1] >= parameters.takeoff_threshold and pooled[i] < parameters.takeoff_threshold:
            takeoff_mask[i] = 1

    # Detect landing events: high positive derivative
    landing_mask = (derivative > parameters.landing_derivative_threshold).astype(int)

    # Pair takeoff and landing events
    takeoff_landing_pairs = _pair_takeoff_landing(
        takeoff_mask,
        landing_mask,
        min_time=parameters.min_flight_time,
        max_time=parameters.max_flight_time,
    )

    # Filter pairs by in-air validation
    in_air = (pooled < parameters.in_air_threshold).astype(int)
    valid_pairs = _filter_pairs_with_flight(takeoff_landing_pairs, in_air)

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

    pair_indicator = np.zeros_like(pooled)
    for pair in takeoff_landing_pairs:
        pair_indicator[[pair.takeoff_idx, pair.landing_idx]] = 1

    valid_pair_indicator = np.zeros_like(pooled)
    for pair in valid_pairs:
        valid_pair_indicator[[pair.takeoff_idx, pair.landing_idx]] = 1

    signals = {
        HybridNames.RAW_DATA.value: raw_data,
        HybridNames.POOLED.value: pooled,
        HybridNames.DERIVATIVE.value: derivative,
        HybridNames.TAKEOFF_MASK.value: takeoff_mask,
        HybridNames.LANDING_MASK.value: landing_mask,
        HybridNames.PAIR_INDICATOR.value: pair_indicator,
        HybridNames.IN_AIR.value: in_air,
        HybridNames.VALID_PAIR_INDICATOR.value: valid_pair_indicator,
    }

    metadata = {
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
    pairs: Iterable[_TakeoffLandingPair], in_air: np.ndarray
) -> list[_TakeoffLandingPair]:
    """Filter pairs to ensure person was in air (pooled < in_air_threshold) between takeoff and landing."""
    return [
        pair
        for pair in pairs
        if in_air[pair.takeoff_idx : pair.landing_idx + 1].any()
    ]




