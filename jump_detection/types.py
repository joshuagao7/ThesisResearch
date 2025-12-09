"""Common dataclasses and enums used across the jump detection toolkit."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

import numpy as np


class ThresholdNames(str, Enum):
    RAW_DATA = "raw_data"
    AVERAGE = "average"
    THRESHOLD_MASK = "threshold_mask"
    PHYSICS_FILTERED = "physics_filtered"
    DERIVATIVE = "derivative"
    DERIVATIVE_BINARY = "derivative_binary"


class DerivativeNames(str, Enum):
    RAW_DATA = "raw_data"
    POOLED = "pooled"
    DERIVATIVE = "derivative"
    UPPER_MASK = "derivative_upper"
    LOWER_MASK = "derivative_lower"
    PAIR_INDICATOR = "derivative_pair_indicator"
    IN_AIR = "in_air"
    VALID_PAIR_INDICATOR = "valid_pair_indicator"


class HybridNames(str, Enum):
    RAW_DATA = "raw_data"
    POOLED = "pooled"
    DERIVATIVE = "derivative"
    TAKEOFF_MASK = "takeoff_mask"
    LANDING_MASK = "landing_mask"
    PAIR_INDICATOR = "pair_indicator"
    IN_AIR = "in_air"
    VALID_PAIR_INDICATOR = "valid_pair_indicator"


class EnsembleNames(str, Enum):
    RAW_DATA = "raw_data"
    POOLED = "pooled"
    SCORE = "score"
    JUMP_MASK = "jump_mask"
    PHYSICS_FILTERED_MASK = "physics_filtered_mask"


class TemplateNames(str, Enum):
    RAW_DATA = "raw_data"
    POOLED = "pooled"
    TAKEOFF_CORRELATION = "takeoff_correlation"
    LANDING_CORRELATION = "landing_correlation"
    TAKEOFF_MASK = "takeoff_mask"
    LANDING_MASK = "landing_mask"


class LandingDerivativeNames(str, Enum):
    RAW_DATA = "raw_data"
    POOLED = "pooled"
    DERIVATIVE = "derivative"
    LANDING_MASK = "landing_mask"  # derivative > threshold (positive)
    LANDING_INDICATOR = "landing_indicator"
    IN_AIR = "in_air"


@dataclass(slots=True)
class Jump:
    """Represents a detected jump and its derived metrics."""

    start: int
    end: int
    center: int
    duration: int
    time_diff: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": int(self.start),
            "end": int(self.end),
            "center": int(self.center),
            "duration": int(self.duration),
            "time_diff": float(self.time_diff) if self.time_diff is not None else None,
        }


@dataclass(slots=True)
class PreciseJump:
    """Detailed analysis of a jump including refined boundaries."""

    jump_number: int
    original_start: int
    original_end: int
    original_center: int
    original_duration: int
    precise_start: int
    precise_end: int
    precise_center: int
    precise_duration: int
    peak_analysis: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jump_number": self.jump_number,
            "original_start": self.original_start,
            "original_end": self.original_end,
            "original_center": self.original_center,
            "original_duration": self.original_duration,
            "precise_start": self.precise_start,
            "precise_end": self.precise_end,
            "precise_center": self.precise_center,
            "precise_duration": self.precise_duration,
            "peak_analysis": dict(self.peak_analysis),
        }


@dataclass(slots=True)
class DetectionResult:
    """Container for the full output of a detector pipeline."""

    participant_name: Optional[str]
    sampling_rate: int
    raw_data: np.ndarray
    pooled_data: Optional[np.ndarray]
    signals: Dict[str, np.ndarray] = field(default_factory=dict)
    jumps: Iterable[Jump] = field(default_factory=list)
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise iterable jumps to a list for repeat usage
        if not isinstance(self.jumps, list):
            self.jumps = list(self.jumps)

    @property
    def num_jumps(self) -> int:
        return len(self.jumps)

    def iter_jumps(self) -> Iterator[Jump]:
        return iter(self.jumps)

    def signal(self, name: str) -> np.ndarray:
        try:
            return self.signals[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Signal '{name}' is not available in this result") from exc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_name": self.participant_name,
            "sampling_rate": self.sampling_rate,
            "raw_data": self.raw_data,
            "pooled_data": self.pooled_data,
            "signals": self.signals,
            "jumps": [jump.to_dict() for jump in self.jumps],
            "metadata": dict(self.metadata),
            "num_jumps": self.num_jumps,
        }


@dataclass(slots=True)
class GroundTruthAnnotations:
    """Ground truth annotations for jump detection validation.
    
    Markers are single frame indices that should be contained within
    correct jump detections.
    """

    data_file_path: str
    markers: list[int]
    created_at: str
    modified_at: str

    def __post_init__(self) -> None:
        # Ensure markers are sorted and unique
        self.markers = sorted(set(self.markers))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_file_path": self.data_file_path,
            "markers": self.markers,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GroundTruthAnnotations:
        return cls(
            data_file_path=data["data_file_path"],
            markers=data["markers"],
            created_at=data.get("created_at", datetime.now().isoformat()),
            modified_at=data.get("modified_at", datetime.now().isoformat()),
        )

    def save(self, data_file_path: Path) -> Path:
        """Save annotations to JSON file next to the data file."""
        from jump_detection.annotations import save_annotations
        return save_annotations(self, data_file_path)

    @classmethod
    def load(cls, data_file_path: Path) -> Optional[GroundTruthAnnotations]:
        """Load annotations from JSON file next to the data file."""
        from jump_detection.annotations import load_annotations
        return load_annotations(data_file_path)

