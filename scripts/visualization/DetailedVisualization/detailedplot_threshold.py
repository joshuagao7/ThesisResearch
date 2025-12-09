"""Detailed plotting for threshold algorithm - automatically processes all data files."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from jump_detection.algorithms.threshold import ThresholdParameters, detect_threshold_jumps
from jump_detection.annotations import load_annotations
from jump_detection.config import (
    DATASET_ROOT,
    DERIVATIVE_THRESHOLD_DEFAULT,
    SAMPLING_RATE,
    THRESHOLD_DEFAULT,
)
from jump_detection.plotting.pipeline import plot_threshold_pipeline
from jump_detection.analysis.precise import calculate_precise_jump_boundaries

import matplotlib.pyplot as plt
import numpy as np

PRIMARY_RED = '#C41E3A'
PRIMARY_ORANGE = '#FF6B35'
PRIMARY_YELLOW = '#FFD23F'
PRIMARY_BLUE = '#0066CC'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def discover_all_data_files(dataset_root: Path = DATASET_ROOT) -> list[tuple[Path, str]]:
    """Discover all data files from Test0, Test1(100Hz), Test2, and Test3 (video) folders.
    
    Returns:
        List of tuples (file_path, participant_folder_name)
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    
    data_files = []
    test_folders = ["Test0", "Test1(100Hz)", "Test2", "Test3 (video)"]
    
    for folder_name in test_folders:
        folder_path = root / folder_name
        if not folder_path.exists():
            continue
        
        if folder_name in ["Test2", "Test3 (video)", "Test1(100Hz)"]:
            # Test1, Test2 and Test3 have .txt files, exclude annotation files
            files = sorted([
                path for path in folder_path.glob("*.txt")
                if not path.name.endswith("_annotations.json")
            ])
        else:
            # Test0 has files with or without extensions, exclude annotation files
            files = sorted([
                path for path in folder_path.iterdir()
                if path.is_file() and not path.name.endswith("_annotations.json") and not path.name.endswith(".json")
            ])
        
        for file_path in files:
            data_files.append((file_path, folder_name))
    
    return data_files


def load_timestamps(data_file_path: Path) -> list:
    """Load timestamps from data file."""
    from datetime import datetime
    timestamps = []
    path = Path(data_file_path)
    
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if " -> " in line:
                timestamp_str = line.split(" -> ")[0].strip()
                try:
                    if "." in timestamp_str:
                        time_part, ms_part = timestamp_str.rsplit(".", 1)
                        if len(ms_part) < 3:
                            ms_part = ms_part.ljust(3, "0")
                        elif len(ms_part) > 3:
                            ms_part = ms_part[:3]
                        timestamp_str = f"{time_part}.{ms_part}"
                        dt = datetime.strptime(timestamp_str, "%H:%M:%S.%f")
                    else:
                        dt = datetime.strptime(timestamp_str, "%H:%M:%S")
                    timestamps.append(dt)
                except ValueError:
                    continue
    
    return timestamps


def plot_individual_jump_snapshots(
    result,
    signal: np.ndarray,
    search_window: int = 70,
    window_frames: int = 150,
    participant_name: str | None = None,
) -> None:
    """Plot individual snapshot for each jump showing both precise and initial boundaries."""
    if not result.jumps:
        return
    
    base_name = participant_name or result.participant_name or "Participant"
    
    for index, jump in enumerate(result.iter_jumps(), start=1):
        precise = calculate_precise_jump_boundaries(signal, jump.center, search_window)
        precise_start = precise["precise_start"]
        precise_end = precise["precise_end"]
        precise_center = precise["precise_center"]
        
        window_start = max(0, precise_center - window_frames)
        window_end = min(len(signal), precise_center + window_frames)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_vals = np.arange(window_start, window_end)
        y_vals = signal[window_start:window_end]
        ax.plot(x_vals, y_vals, linewidth=2, color=PRIMARY_BLUE, alpha=0.8, label="Signal")
        
        ax.axvspan(
            jump.start,
            jump.end,
            alpha=0.2,
            color=PRIMARY_BLUE,
            label="Initial Boundaries",
        )
        ax.axvline(jump.start, color=PRIMARY_BLUE, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axvline(jump.end, color=PRIMARY_BLUE, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axvline(jump.center, color=PRIMARY_BLUE, linestyle=":", linewidth=1, alpha=0.5, label="Initial Center")
        
        ax.axvspan(
            precise_start,
            precise_end,
            alpha=0.3,
            color=PRIMARY_YELLOW,
            label="Precise Boundaries",
        )
        ax.axvline(precise_start, color=PRIMARY_YELLOW, linestyle="-", linewidth=2, alpha=0.9)
        ax.axvline(precise_end, color=PRIMARY_YELLOW, linestyle="-", linewidth=2, alpha=0.9)
        ax.axvline(precise_center, color=PRIMARY_RED, linestyle="--", linewidth=2, alpha=0.9, label="Precise Center")
        
        y_max = ax.get_ylim()[1]
        y_text = y_max * 0.95
        
        initial_duration = jump.end - jump.start
        ax.text(
            jump.center,
            y_text,
            f"Initial: {initial_duration}f",
            ha="center",
            va="top",
            fontsize=9,
            color=PRIMARY_BLUE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        
        precise_duration = precise["precise_duration"]
        ax.text(
            precise_center,
            y_text * 0.85,
            f"Precise: {precise_duration}f",
            ha="center",
            va="top",
            fontsize=9,
            color=PRIMARY_RED,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        
        ax.set_xlabel("Frame Index", fontsize=12)
        ax.set_ylabel("Signal Value", fontsize=12)
        ax.set_title(
            f"Jump {index} Snapshot - {base_name}\n"
            f"Initial: frames {jump.start}-{jump.end} | Precise: frames {precise_start}-{precise_end}",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        
        plt.tight_layout()
        plt.close(fig)  # Close instead of showing


def process_all_files(
    params: ThresholdParameters | None = None,
    search_window: int = 70,
    save_plots: bool = True,
) -> None:
    """Process all data files and generate detailed plots for threshold algorithm."""
    if params is None:
        params = ThresholdParameters(
            threshold=THRESHOLD_DEFAULT,
            derivative_threshold=DERIVATIVE_THRESHOLD_DEFAULT,
        )
    
    data_files = discover_all_data_files()
    print(f"Found {len(data_files)} data files to process")
    
    project_root = Path(__file__).parent.parent.parent.parent
    save_dir = project_root / "results" / "plots" / "pipeline" / "threshold"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path, folder_name in data_files:
        participant_name = file_path.stem
        print(f"\n{'='*80}")
        print(f"Processing: {participant_name} ({folder_name})")
        print(f"{'='*80}")
        
        try:
            # Load annotations if available
            annotations = load_annotations(file_path)
            ground_truth_markers = annotations.markers if annotations else None
            
            # Detect jumps
            result = detect_threshold_jumps(
                file_path,
                participant_name,
                params=params,
                save_windows=False,
            )
            
            print(f"Threshold algorithm parameters: threshold={params.threshold:.3f}, "
                  f"derivative_threshold={params.derivative_threshold:.3f}")
            if ground_truth_markers:
                print(f"Ground truth markers: {len(ground_truth_markers)} markers loaded")
            print(f"Detected jumps: {len(result.jumps)}")
            
            # Load timestamps
            timestamps = load_timestamps(file_path)
            
            # Compute and print precise flight times
            signal = result.pooled_data if result.pooled_data is not None else result.raw_data.sum(axis=1)
            if result.jumps:
                print("\nPrecise jump flight times (threshold algorithm):")
                center_times = []
                for index, jump in enumerate(result.iter_jumps(), start=1):
                    precise = calculate_precise_jump_boundaries(signal, jump.center, search_window)
                    precise_start_idx = precise["precise_start"]
                    precise_end_idx = precise["precise_end"]
                    precise_center_idx = precise["precise_center"]
                    duration_frames = precise["precise_duration"]
                    
                    if timestamps and len(timestamps) > precise_end_idx:
                        start_timestamp = timestamps[precise_start_idx]
                        end_timestamp = timestamps[precise_end_idx]
                        center_timestamp = timestamps[precise_center_idx]
                        
                        flight_time_delta = (end_timestamp - start_timestamp).total_seconds()
                        start_time = (start_timestamp - timestamps[0]).total_seconds()
                        end_time = (end_timestamp - timestamps[0]).total_seconds()
                        center_time = (center_timestamp - timestamps[0]).total_seconds()
                    else:
                        flight_time_delta = duration_frames / SAMPLING_RATE
                        start_time = precise_start_idx / SAMPLING_RATE
                        end_time = precise_end_idx / SAMPLING_RATE
                        center_time = precise_center_idx / SAMPLING_RATE
                    
                    center_times.append(center_time)
                    print(
                        f"  Jump {index}: "
                        f"{duration_frames} frames, "
                        f"flight_time={flight_time_delta:.3f}s, "
                        f"start={start_time:.3f}s, "
                        f"end={end_time:.3f}s"
                    )
                print("  Jump center times (s):", [f"{t:.3f}" for t in center_times])
            else:
                print("\nNo jumps detected (threshold algorithm)")
            
            # Generate pipeline plot
            if save_plots:
                fig = plot_threshold_pipeline(result, search_window=search_window, show=False)
                save_path = save_dir / f"{participant_name}_pipeline.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved pipeline plot to {save_path}")
            
            # Generate individual jump snapshots (saved but not displayed)
            if result.jumps:
                plot_individual_jump_snapshots(
                    result,
                    signal,
                    search_window=search_window,
                    participant_name=participant_name,
                )
                
        except Exception as e:
            print(f"Error processing {participant_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"Completed processing {len(data_files)} files")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Use default parameters
    params = ThresholdParameters(
        threshold=THRESHOLD_DEFAULT,
        derivative_threshold=DERIVATIVE_THRESHOLD_DEFAULT,
    )
    
    process_all_files(params=params, save_plots=True)

