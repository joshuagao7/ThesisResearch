"""Precise boundary detection built on top of jump detection outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from ..algorithms.threshold import process_all_threshold_participants
from ..config import SAMPLING_RATE
from ..types import DetectionResult, Jump, PreciseJump


def find_peak_before_after_center(
    signal: np.ndarray, center: int, search_window: int = 70
) -> tuple[float, float, int, int]:
    left_start = max(0, center - search_window)
    right_end = min(len(signal), center + search_window)

    left_segment = signal[left_start:center]
    if left_segment.size:
        peak_before_idx = int(np.argmax(left_segment)) + left_start
        peak_before = float(signal[peak_before_idx])
    else:
        peak_before_idx = center
        peak_before = float(signal[center])

    right_segment = signal[center:right_end]
    if right_segment.size:
        peak_after_idx = int(np.argmax(right_segment)) + center
        peak_after = float(signal[peak_after_idx])
    else:
        peak_after_idx = center
        peak_after = float(signal[center])

    return peak_before, peak_after, peak_before_idx, peak_after_idx


def calculate_precise_jump_boundaries(
    signal: np.ndarray, jump_center: int, search_window: int = 70
) -> dict[str, float | int]:
    peak_before, peak_after, peak_before_idx, peak_after_idx = find_peak_before_after_center(
        signal, jump_center, search_window
    )

    # Find minimum value in the region between the peaks (resting value)
    region_start = max(0, min(peak_before_idx, peak_after_idx))
    region_end = min(len(signal), max(peak_before_idx, peak_after_idx) + 1)
    region_signal = signal[region_start:region_end]
    min_value = float(np.min(region_signal)) if region_signal.size > 0 else 0.0

    # Calculate threshold as (max - min) / 2 + min = (max + min) / 2
    # This accounts for non-zero resting values
    half_peak_before = (peak_before - min_value) / 2 + min_value
    half_peak_after = (peak_after - min_value) / 2 + min_value

    precise_start = jump_center
    search_start = max(0, jump_center - search_window)
    for idx in range(jump_center, search_start - 1, -1):
        if signal[idx] >= half_peak_before:
            precise_start = idx
            break

    precise_end = jump_center
    search_end = min(len(signal), jump_center + search_window)
    for idx in range(jump_center, search_end):
        if signal[idx] >= half_peak_after:
            precise_end = idx
            break

    # Ensure we have a valid duration (at least 1 frame)
    # If both start and end are at center, expand by at least 1 frame on each side
    if precise_start >= precise_end:
        precise_start = max(0, jump_center - 1)
        precise_end = min(len(signal) - 1, jump_center + 1)
        # Ensure end > start
        if precise_start >= precise_end:
            precise_end = precise_start + 1

    return {
        "precise_start": int(precise_start),
        "precise_end": int(precise_end),
        "precise_center": int((precise_start + precise_end) // 2),
        "precise_duration": int(precise_end - precise_start),
        "peak_before": float(peak_before),
        "peak_after": float(peak_after),
        "peak_before_idx": int(peak_before_idx),
        "peak_after_idx": int(peak_after_idx),
        "half_peak_before": float(half_peak_before),
        "half_peak_after": float(half_peak_after),
        "min_value": float(min_value),
        "original_center": int(jump_center),
    }


def process_precise_jumps(
    detection_results: Optional[Sequence[DetectionResult]] = None,
    *,
    search_window: int = 70,
) -> list[dict[str, object]]:
    if detection_results is None:
        detection_results = process_all_threshold_participants()

    precise_results: list[dict[str, object]] = []

    for result in detection_results:
        participant_name = result.participant_name or "Participant"
        signal = result.pooled_data if result.pooled_data is not None else result.raw_data.sum(axis=1)

        precise_jumps: list[PreciseJump] = []
        for index, jump in enumerate(result.jumps, start=1):
            precise_data = calculate_precise_jump_boundaries(
                signal, jump.center, search_window
            )
            precise_jumps.append(
                PreciseJump(
                    jump_number=index,
                    original_start=jump.start,
                    original_end=jump.end,
                    original_center=jump.center,
                    original_duration=jump.duration,
                    precise_start=precise_data["precise_start"],
                    precise_end=precise_data["precise_end"],
                    precise_center=precise_data["precise_center"],
                    precise_duration=precise_data["precise_duration"],
                    peak_analysis=precise_data,
                )
            )

        precise_results.append(
            {
                "participant_name": participant_name,
                "detection_result": result,
                "precise_jumps": precise_jumps,
            }
        )

    return precise_results


def print_summary(precise_results: Sequence[dict[str, object]]) -> None:
    total_original_duration = 0
    total_precise_duration = 0

    for entry in precise_results:
        participant_name = entry["participant_name"]
        precise_jumps: Sequence[PreciseJump] = entry["precise_jumps"]

        original_total = sum(jump.original_duration for jump in precise_jumps)
        precise_total = sum(jump.precise_duration for jump in precise_jumps)

        total_original_duration += original_total
        total_precise_duration += precise_total

        print(f"\n{participant_name}:")
        print(f"  Original total duration: {original_total} frames")
        print(f"  Precise total duration:  {precise_total} frames")
        print(f"  Duration change: {precise_total - original_total} frames")

        for jump in precise_jumps:
            duration_change = jump.precise_duration - jump.original_duration
            print(
                f"    Jump {jump.jump_number}: {jump.original_duration} → {jump.precise_duration} "
                f"({duration_change:+d})"
            )

    print("\nOVERALL TOTALS:")
    print(f"Total original duration: {total_original_duration} frames")
    print(f"Total precise duration:  {total_precise_duration} frames")
    print(f"Total duration change:   {total_precise_duration - total_original_duration} frames")


__all__ = [
    "calculate_precise_jump_boundaries",
    "find_peak_before_after_center",
    "process_precise_jumps",
    "print_summary",
]

