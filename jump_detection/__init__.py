"""High-level API for the jump detection toolkit."""

from .algorithms.threshold import (
    ThresholdParameters,
    detect_threshold_jumps,
    detect_threshold_jumps_with_params,
    process_all_threshold_participants,
)
from .algorithms.derivative import (
    DerivativeParameters,
    detect_derivative_jumps,
    detect_derivative_jumps_with_params,
    process_all_derivative_participants,
)
from .analysis.precise import (
    calculate_precise_jump_boundaries,
    find_peak_before_after_center,
    process_precise_jumps,
    print_summary,
)
from .data import DEFAULT_DATA_FILES, load_dataset
from .types import DetectionResult, Jump, PreciseJump, ThresholdNames, DerivativeNames

__all__ = [
    "ThresholdParameters",
    "detect_threshold_jumps",
    "detect_threshold_jumps_with_params",
    "process_all_threshold_participants",
    "DerivativeParameters",
    "detect_derivative_jumps",
    "detect_derivative_jumps_with_params",
    "process_all_derivative_participants",
    "calculate_precise_jump_boundaries",
    "find_peak_before_after_center",
    "process_precise_jumps",
    "print_summary",
    "load_dataset",
    "DEFAULT_DATA_FILES",
    "DetectionResult",
    "Jump",
    "PreciseJump",
    "ThresholdNames",
    "DerivativeNames",
]

