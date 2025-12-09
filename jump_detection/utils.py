"""Shared utilities for jump detection toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .config import DATASET_ROOT, DEFAULT_DATA_FILES
from .types import DetectionResult, Jump
from .analysis.precise import calculate_precise_jump_boundaries
from .annotations import get_annotation_path

# Color palette constants
PRIMARY_RED = '#C41E3A'
PRIMARY_ORANGE = '#FF6B35'
PRIMARY_YELLOW = '#FFD23F'
PRIMARY_BLUE = '#0066CC'

# Gradient colormap
GRADIENT_COLORS = [PRIMARY_RED, PRIMARY_ORANGE, PRIMARY_YELLOW, PRIMARY_BLUE]
GRADIENT_CMAP = LinearSegmentedColormap.from_list('red_orange_yellow_blue', GRADIENT_COLORS, N=100)

# Default search window for precise jump detection
SEARCH_WINDOW = 70


def find_all_data_files(dataset_root: Path = DATASET_ROOT) -> list[Path]:
    """Find all data files with corresponding annotation files from DEFAULT_DATA_FILES in config.
    
    Filters DEFAULT_DATA_FILES to only include files that have corresponding annotation files.
    """
    data_files = []
    
    for data_file_path in DEFAULT_DATA_FILES:
        # DEFAULT_DATA_FILES uses relative paths from DATASET_ROOT
        # Resolve relative to the project root (parent of dataset_root)
        if data_file_path.is_absolute():
            full_path = data_file_path
        else:
            # If path is relative, it's relative to DATASET_ROOT
            # Resolve it relative to the project root
            project_root = dataset_root.parent if dataset_root.name == "dataset" else dataset_root
            # Remove "dataset/" prefix if present in the path
            path_str = str(data_file_path)
            if path_str.startswith("dataset/"):
                path_str = path_str[8:]  # Remove "dataset/" prefix
            full_path = project_root / "dataset" / path_str
        
        # Check if file exists and has annotations
        if full_path.exists():
            annotation_path = get_annotation_path(full_path)
            if annotation_path.exists():
                data_files.append(full_path)
    
    return sorted(data_files)


def concatenate_data_and_annotations(
    data_files_with_annotations: list[tuple[Path, list[int]]],
) -> tuple[np.ndarray, list[int], list[int]]:
    """Concatenate data files and annotations into single arrays."""
    from .data import load_dataset
    
    all_data = []
    all_markers = []
    file_boundaries = [0]
    current_offset = 0
    
    for data_file, ground_truth_markers in data_files_with_annotations:
        raw_data = load_dataset(data_file)
        all_data.append(raw_data)
        all_markers.extend([m + current_offset for m in ground_truth_markers])
        current_offset += len(raw_data)
        file_boundaries.append(current_offset)
    
    return np.vstack(all_data), all_markers, file_boundaries


def compute_precise_jumps_from_result(
    data: np.ndarray,
    result: DetectionResult,
    search_window: int = SEARCH_WINDOW,
) -> list[Jump]:
    """Calculate precise jump boundaries from detection result."""
    signal = result.pooled_data if result.pooled_data is not None else data.sum(axis=1)
    precise_jumps = []
    
    for jump in result.jumps:
        precise_data = calculate_precise_jump_boundaries(signal, jump.center, search_window)
        if precise_data["precise_start"] < precise_data["precise_end"]:
            precise_jumps.append(
                Jump(
                    start=precise_data["precise_start"],
                    end=precise_data["precise_end"],
                    center=precise_data["precise_center"],
                    duration=precise_data["precise_duration"],
                )
            )
    
    return precise_jumps


class ProgressTracker:
    """Simple progress bar tracker."""
    
    def __init__(self, total: int):
        self.total = total
        self.count = 0
        self.bar_length = 40
        self.report_interval = max(1, total // self.bar_length)
        print("Progress: [", end="", flush=True)
    
    def update(self) -> None:
        self.count += 1
        if self.count % self.report_interval == 0 or self.count == self.total:
            progress = self.count / self.total
            filled = int(self.bar_length * progress)
            bar = "█" * filled + "░" * (self.bar_length - filled)
            print(f"\rProgress: [{bar}] {progress * 100:5.1f}%", end="", flush=True)
    
    def finish(self) -> None:
        print()


__all__ = [
    "PRIMARY_RED",
    "PRIMARY_ORANGE", 
    "PRIMARY_YELLOW",
    "PRIMARY_BLUE",
    "GRADIENT_COLORS",
    "GRADIENT_CMAP",
    "SEARCH_WINDOW",
    "find_all_data_files",
    "concatenate_data_and_annotations",
    "compute_precise_jumps_from_result",
    "ProgressTracker",
]

