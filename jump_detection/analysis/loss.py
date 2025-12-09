"""Precision-based loss functions for jump detection evaluation."""

from __future__ import annotations

from typing import Dict

from ..types import Jump


def compute_precision_loss(
    detected_jumps: list[Jump],
    ground_truth_markers: list[int],
) -> Dict[str, int]:
    """Compute precision-based loss: false_positives + false_negatives."""
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    
    for jump in detected_jumps:
        if any(jump.start <= marker <= jump.end for marker in ground_truth_markers):
            true_positives += 1
        else:
            false_positives += 1
    
    for marker in ground_truth_markers:
        if not any(jump.start <= marker <= jump.end for jump in detected_jumps):
            false_negatives += 1
    
    return {
        "loss": false_positives + false_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
    }

