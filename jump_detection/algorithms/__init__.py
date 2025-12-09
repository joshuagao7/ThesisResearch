"""Aggregates the different jump detection algorithm implementations."""

from .threshold import (
    ThresholdParameters,
    detect_threshold_jumps,
    detect_threshold_jumps_with_params,
    process_all_threshold_participants,
)
from .derivative import (
    DerivativeParameters,
    detect_derivative_jumps,
    detect_derivative_jumps_with_params,
    process_all_derivative_participants,
)
from .correlation import (
    CorrelationParameters,
    detect_correlation_jumps,
    detect_correlation_jumps_with_params,
    process_all_participants_correlation,
)
from .hybrid import (
    HybridParameters,
    detect_hybrid_jumps,
    detect_hybrid_jumps_with_params,
    process_all_hybrid_participants,
)
from .ensemble import (
    EnsembleParameters,
    detect_ensemble_jumps,
    detect_ensemble_jumps_with_params,
    process_all_ensemble_participants,
)
from .template import (
    TemplateParameters,
    detect_template_jumps,
    detect_template_jumps_with_params,
    process_all_template_participants,
)
from .landing_derivative import (
    LandingDerivativeParameters,
    detect_landing_derivative_jumps,
    detect_landing_derivative_jumps_with_params,
    process_all_landing_derivative_participants,
)

__all__ = [
    "ThresholdParameters",
    "detect_threshold_jumps",
    "detect_threshold_jumps_with_params",
    "process_all_threshold_participants",
    "DerivativeParameters",
    "detect_derivative_jumps",
    "detect_derivative_jumps_with_params",
    "process_all_derivative_participants",
    "CorrelationParameters",
    "detect_correlation_jumps",
    "detect_correlation_jumps_with_params",
    "process_all_participants_correlation",
    "HybridParameters",
    "detect_hybrid_jumps",
    "detect_hybrid_jumps_with_params",
    "process_all_hybrid_participants",
    "EnsembleParameters",
    "detect_ensemble_jumps",
    "detect_ensemble_jumps_with_params",
    "process_all_ensemble_participants",
    "TemplateParameters",
    "detect_template_jumps",
    "detect_template_jumps_with_params",
    "process_all_template_participants",
    "LandingDerivativeParameters",
    "detect_landing_derivative_jumps",
    "detect_landing_derivative_jumps_with_params",
    "process_all_landing_derivative_participants",
]

