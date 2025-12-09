"""Utilities for loading datasets and persisting extracted jump windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .config import (
    DEFAULT_DATA_FILES,
    EXTRACTION_WINDOW_DEFAULT,
    SAMPLING_RATE,
)
from .types import Jump


def load_dataset(data_file_path: str | Path) -> np.ndarray:
    """Load sensor dataset from disk."""
    path = Path(data_file_path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if " -> " in line:
                rows.append(line.split(" -> ")[1].split(","))
    return np.array([[float(x) for x in row] for row in rows])


@dataclass(slots=True)
class JumpWindowExport:
    output_dir: Path
    window_size: int = EXTRACTION_WINDOW_DEFAULT
    sampling_rate: int = SAMPLING_RATE

    def save(self, raw_data: np.ndarray, jumps: Iterable[Jump], participant_name: Optional[str] = None) -> list[Path]:
        """Persist slices of raw sensor data around each jump."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        base_timestamp = datetime.now()
        half_window = self.window_size // 2
        output_paths = []

        for index, jump in enumerate(jumps, start=1):
            start = max(0, jump.center - half_window)
            end = min(len(raw_data), jump.center + half_window)
            jump_data = raw_data[start:end, :]
            filename = f"{participant_name}_jump_{index:02d}.txt" if participant_name else f"jump_{index:02d}.txt"
            file_path = self.output_dir / filename
            
            with file_path.open("w", encoding="utf-8") as handle:
                for frame_idx, frame in enumerate(jump_data):
                    timestamp = base_timestamp + timedelta(milliseconds=(start + frame_idx) * 1000 / self.sampling_rate)
                    sensor_values = ",".join(str(int(value)) for value in frame)
                    handle.write(f"{timestamp.strftime('%H:%M:%S.%f')[:-3]} -> {sensor_values}\n")
            output_paths.append(file_path)

        return output_paths


__all__ = [
    "DEFAULT_DATA_FILES",
    "JumpWindowExport",
    "load_dataset",
]

