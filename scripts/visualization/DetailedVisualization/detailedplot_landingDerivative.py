"""Detailed plotting for landing derivative jump detection algorithm.

Visualizes the landing derivative algorithm pipeline showing:
- Raw sensor data
- Pooled signal and derivative
- Landing detection (derivative < threshold)
- Detected jumps with precise boundaries
- Ground truth markers (if available)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

PRIMARY_RED = '#C41E3A'      # Deep red
PRIMARY_ORANGE = '#FF6B35'   # Vibrant orange
PRIMARY_YELLOW = '#FFD23F'   # Golden yellow
PRIMARY_BLUE = '#0066CC'     # Electric blue

# Set consistent font styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

from jump_detection.algorithms.landing_derivative import (
    LandingDerivativeParameters,
    detect_landing_derivative_jumps,
)
from jump_detection.annotations import load_annotations
from jump_detection.config import (
    DATASET_ROOT,
    DERIVATIVE_UPPER_DEFAULT,
    LANDING_DERIVATIVE_CENTER_OFFSET_DEFAULT,
    LANDING_DERIVATIVE_SEARCH_WINDOW_DEFAULT,
    IN_AIR_THRESHOLD_DEFAULT,
    SAMPLING_RATE,
)
from jump_detection.analysis.precise import calculate_precise_jump_boundaries
from jump_detection.types import LandingDerivativeNames

SEARCH_WINDOW = 70


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
            files = sorted([
                path for path in folder_path.glob("*.txt")
                if not path.name.endswith("_annotations.json")
            ])
        else:
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


def select_dataset_path(dataset_root: Path = DATASET_ROOT) -> tuple[Path, str]:
    """Select a data file from Test0, Test1(100Hz), Test2, or Test3 (video) folders."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    # Show Test0, Test1(100Hz), Test2, and Test3 (video) folders
    test0_dir = root / "Test0"
    test1_dir = root / "Test1(100Hz)"
    test2_dir = root / "Test2"
    test3_dir = root / "Test3 (video)"
    
    participants = []
    if test0_dir.exists():
        participants.append(test0_dir)
    if test1_dir.exists():
        participants.append(test1_dir)
    if test2_dir.exists():
        participants.append(test2_dir)
    if test3_dir.exists():
        participants.append(test3_dir)
    
    if not participants:
        raise FileNotFoundError(f"No Test0, Test1(100Hz), Test2, or Test3 (video) folders found in {root}")

    participant_names = [path.name for path in participants]
    print("\nAvailable folders (Test0, Test1(100Hz), Test2, and Test3 (video)):")
    for idx, name in enumerate(participant_names, start=1):
        print(f"  {idx}. {name}")

    def _prompt_selection(options: list[str], prompt_message: str) -> int:
        """Prompt user to select from a list of options."""
        while True:
            choice = input(prompt_message).strip()
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(options):
                    return index
            print("Invalid selection. Please enter one of the listed numbers.")

    participant_index = _prompt_selection(
        participant_names, "Select a folder by number: "
    )
    participant_path = participants[participant_index]

    # Get files from the selected folder
    if participant_path.name in ["Test2", "Test3 (video)", "Test1(100Hz)"]:
        # Test1, Test2 and Test3 have .txt files, exclude annotation files
        data_files = sorted([
            path for path in participant_path.glob("*.txt")
            if not path.name.endswith("_annotations.json")
        ])
    else:
        # Test0 has files with or without extensions, exclude annotation files
        data_files = sorted([
            path for path in participant_path.iterdir()
            if path.is_file() and not path.name.endswith("_annotations.json") and not path.name.endswith(".json")
        ])
    
    if not data_files:
        raise FileNotFoundError(
            f"No data files found in '{participant_path.name}'"
        )

    file_labels = [path.name for path in data_files]
    print(f"\nAvailable files for {participant_path.name}:")
    for idx, name in enumerate(file_labels, start=1):
        print(f"  {idx}. {name}")

    file_index = _prompt_selection(
        file_labels, "Select a file by number: "
    )

    return data_files[file_index], participant_path.name


def plot_landing_derivative_pipeline(
    result,
    *,
    search_window: int = SEARCH_WINDOW,
    show: bool = True,
    ground_truth_markers: Optional[list[int]] = None,
) -> plt.Figure:
    """Plot the landing derivative detection pipeline."""
    signals = result.signals
    params = result.metadata.get("parameters", {})
    raw_data = signals[LandingDerivativeNames.RAW_DATA.value]
    pooled = signals[LandingDerivativeNames.POOLED.value]
    derivative = signals[LandingDerivativeNames.DERIVATIVE.value]
    landing_mask = signals[LandingDerivativeNames.LANDING_MASK.value]
    landing_indicator = signals[LandingDerivativeNames.LANDING_INDICATOR.value]
    in_air = signals.get(LandingDerivativeNames.IN_AIR.value, np.zeros_like(pooled))

    fig, axes = plt.subplots(6, 1, figsize=(20, 14), sharex=True)
    fig.suptitle(
        f"Jump Detection Pipeline (Landing Derivative Algorithm) - {result.participant_name or 'Participant'}",
        fontsize=18
    )

    # Plot 1: Raw sensor data
    axes[0].set_ylabel("Raw Sensors", fontsize=12)
    for channel in range(raw_data.shape[1]):
        axes[0].plot(raw_data[:, channel], linewidth=0.5, alpha=0.9)
    axes[0].set_title("Raw Sensor Data", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Pooled signal
    axes[1].set_ylabel("Signal Value", fontsize=12)
    axes[1].plot(pooled, linewidth=1.5, color=PRIMARY_BLUE, label="Pooled Data", alpha=0.8)
    
    # Show in-air threshold if available
    in_air_threshold = params.get("in_air_threshold")
    if in_air_threshold is not None:
        axes[1].axhline(
            y=in_air_threshold,
            color=PRIMARY_ORANGE,
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
            label=f"In-Air Threshold ({in_air_threshold:.1f})",
        )
    
    axes[1].set_title("Pooled Sensor Signal", fontsize=14)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Derivative
    axes[2].set_ylabel("Derivative", fontsize=12)
    axes[2].plot(derivative, linewidth=1.5, color=PRIMARY_ORANGE, label="Derivative", alpha=0.8)
    axes[2].axhline(y=0, color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
    
    # Show landing threshold (for landing detection - positive derivative)
    landing_threshold = params.get("landing_threshold")
    if landing_threshold is not None:
        axes[2].axhline(
            y=landing_threshold,
            color=PRIMARY_RED,
            linestyle="--",
            alpha=0.8,
            linewidth=2,
            label=f"Landing Threshold ({landing_threshold:.3f})",
        )
        axes[2].text(
            0.99,
            landing_threshold,
            f"Threshold: {landing_threshold:.3f}",
            color=PRIMARY_RED,
            fontsize=9,
            ha="right",
            va="bottom",
            transform=axes[2].get_yaxis_transform(),
            backgroundcolor="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
    
    axes[2].set_title("Derivative Signal", fontsize=14)
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Landing mask and indicators
    axes[3].set_ylabel("Landing Detection", fontsize=12)
    
    # Plot landing mask (where derivative > threshold)
    landing_mask_plot = axes[3].plot(
        landing_mask,
        linewidth=1.5,
        color=PRIMARY_RED,
        alpha=0.6,
        label="Landing Mask (derivative > threshold)",
    )[0]
    
    # Plot landing indicators (actual detected landing points)
    landing_indices = np.where(landing_indicator > 0)[0]
    if len(landing_indices) > 0:
        axes[3].scatter(
            landing_indices,
            landing_indicator[landing_indices],
            color=PRIMARY_RED,
            s=50,
            marker="v",
            zorder=5,
            label=f"Detected Landings ({len(landing_indices)})",
        )
    
    axes[3].set_ylim(-0.1, 1.5)
    axes[3].set_title("Landing Detection (derivative > landing_threshold)", fontsize=14)
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    # Plot 5: Center offset visualization
    center_offset = params.get("center_offset", 10)
    axes[4].set_ylabel("Center Calculation", fontsize=12)
    axes[4].plot(pooled, linewidth=1.0, color=PRIMARY_BLUE, alpha=0.5, label="Pooled Signal")
    
    # Show where centers are calculated (landing_idx - center_offset)
    for landing_idx in landing_indices:
        center = max(0, landing_idx - center_offset)
        axes[4].axvline(
            landing_idx,
            color=PRIMARY_RED,
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Landing" if landing_idx == landing_indices[0] else "",
        )
        axes[4].axvline(
            center,
            color=PRIMARY_YELLOW,
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label="Calculated Center" if landing_idx == landing_indices[0] else "",
        )
        # Draw arrow from landing to center
        axes[4].annotate(
            "",
            xy=(center, pooled[center] if center < len(pooled) else pooled[-1]),
            xytext=(landing_idx, pooled[landing_idx] if landing_idx < len(pooled) else pooled[-1]),
            arrowprops=dict(arrowstyle="->", color=PRIMARY_YELLOW, alpha=0.5, lw=1),
        )
    
    axes[4].set_title(f"Center Calculation (offset: {center_offset} frames left from landing)", fontsize=14)
    axes[4].legend(loc="upper right")
    axes[4].grid(True, alpha=0.3)

    # Plot 6: Detected jumps with precise boundaries
    axes[5].set_ylabel("Detected Jumps", fontsize=12)
    axes[5].set_xlabel("Frame Index", fontsize=12)
    
    # Plot pooled signal in background
    axes[5].plot(pooled, linewidth=1.0, color=PRIMARY_BLUE, alpha=0.4, label="Pooled Signal")
    
    # Highlight detected jumps
    text_height = axes[5].get_ylim()[1] * 0.9 if axes[5].get_ylim()[1] else 0.9
    for jump_idx, jump in enumerate(result.iter_jumps(), start=1):
        # Highlight jump region
        axes[5].axvspan(
            jump.start,
            jump.end,
            alpha=0.2,
            color=PRIMARY_YELLOW,
            label="Detected Jump" if jump_idx == 1 else "",
        )
        
        # Mark boundaries
        axes[5].axvline(jump.start, color=PRIMARY_YELLOW, linestyle="-", linewidth=2, alpha=0.8)
        axes[5].axvline(jump.end, color=PRIMARY_YELLOW, linestyle="-", linewidth=2, alpha=0.8)
        axes[5].axvline(jump.center, color=PRIMARY_RED, linestyle="--", linewidth=1.5, alpha=0.7)
        
        # Add jump label
        axes[5].text(
            jump.center,
            text_height,
            f"J{jump_idx}\n{jump.duration}f",
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color=PRIMARY_RED,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
    
    # Highlight ground truth markers if available
    if ground_truth_markers:
        for marker in ground_truth_markers:
            if 0 <= marker < len(pooled):
                axes[5].axvline(
                    marker,
                    color=PRIMARY_RED,
                    linewidth=2,
                    alpha=0.7,
                    linestyle="--",
                    label="Ground Truth Marker" if marker == ground_truth_markers[0] else "",
                )
                axes[5].plot(
                    marker,
                    pooled[marker] if marker < len(pooled) else pooled[-1],
                    "*",
                    color=PRIMARY_RED,
                    markersize=12,
                    zorder=10,
                )
        if ground_truth_markers:
            axes[5].legend(loc="upper right")
    
    axes[5].set_title(f"Detected Jumps with Precise Boundaries ({len(result.jumps)} jumps)", fontsize=14)
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def main() -> None:
    """Main function to run the landing derivative detailed plot."""
    print("=" * 80)
    print("LANDING DERIVATIVE ALGORITHM - DETAILED VISUALIZATION")
    print("=" * 80)
    
    # Select dataset
    data_file_path, participant_name = select_dataset_path()
    print(f"\nSelected: {participant_name} / {data_file_path.name}")
    
    # Get parameters
    print("\nLanding Derivative Algorithm Parameters:")
    print("  Default: landing_threshold=20, center_offset=10, search_window=70")
    
    def _read_float(prompt: str, default: float) -> float:
        while True:
            value = input(f"{prompt} (default {default:.3f}): ").strip()
            if not value:
                return default
            try:
                return float(value)
            except ValueError:
                print("Invalid number. Please try again.")
    
    def _read_int(prompt: str, default: int) -> int:
        while True:
            value = input(f"{prompt} (default {default}): ").strip()
            if not value:
                return default
            try:
                return int(value)
            except ValueError:
                print("Invalid number. Please try again.")
    
    landing_threshold = _read_float("Enter landing threshold (positive, for derivative > threshold)", 20.0)
    center_offset = _read_int("Enter center offset (frames to go left from landing)", 10)
    search_window = _read_int("Enter search window (for precise boundaries)", 70)
    
    # Optional in-air threshold
    use_in_air = input("Use in-air threshold filtering? [y/N]: ").strip().lower()
    in_air_threshold = None
    if use_in_air in {"y", "yes"}:
        in_air_threshold = _read_float("Enter in-air threshold", 190.0)
    
    # Create parameters
    params = LandingDerivativeParameters(
        landing_threshold=landing_threshold,
        center_offset=center_offset,
        search_window=search_window,
        in_air_threshold=in_air_threshold,
    )
    
    # Run detection
    print(f"\nRunning landing derivative algorithm...")
    result = detect_landing_derivative_jumps(
        data_file_path,
        participant_name,
        params=params,
        save_windows=False,
    )
    
    print(f"\nAlgorithm parameters:")
    print(f"  Landing threshold:   {params.landing_threshold:.3f} (derivative > threshold)")
    print(f"  Center offset:        {params.center_offset} frames")
    print(f"  Search window:        {params.search_window} frames")
    print(f"  In-air threshold:    {params.in_air_threshold if params.in_air_threshold else 'None'}")
    
    # Load annotations if available
    annotations = load_annotations(Path(data_file_path))
    ground_truth_markers = annotations.markers if annotations else None
    
    if ground_truth_markers:
        print(f"  Ground truth markers: {len(ground_truth_markers)} markers loaded")
    
    print(f"\nDetection results:")
    print(f"  Detected jumps: {len(result.jumps)}")
    
    if result.jumps:
        print("\nJump details:")
        for idx, jump in enumerate(result.iter_jumps(), start=1):
            print(
                f"  Jump {idx}: frames {jump.start}-{jump.end} "
                f"(center: {jump.center}, duration: {jump.duration} frames, "
                f"{jump.duration / SAMPLING_RATE:.3f}s)"
            )
    else:
        print("  No jumps detected")
    
    # Create visualization
    print("\nGenerating visualization...")
    fig = plot_landing_derivative_pipeline(
        result,
        search_window=search_window,
        show=True,
        ground_truth_markers=ground_truth_markers,
    )
    
    # Save figure
    project_root = Path(__file__).parent.parent.parent.parent
    save_path = project_root / "results" / "plots" / "pipeline" / "landing_derivative_pipeline.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved pipeline plot to {save_path}")
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)


def process_all_files(
    params: LandingDerivativeParameters | None = None,
    search_window: int = SEARCH_WINDOW,
    save_plots: bool = True,
) -> None:
    """Process all data files and generate detailed plots for landing derivative algorithm."""
    if params is None:
        params = LandingDerivativeParameters(
            landing_threshold=DERIVATIVE_UPPER_DEFAULT,
            center_offset=LANDING_DERIVATIVE_CENTER_OFFSET_DEFAULT,
            search_window=LANDING_DERIVATIVE_SEARCH_WINDOW_DEFAULT,
            in_air_threshold=IN_AIR_THRESHOLD_DEFAULT,
        )
    
    data_files = discover_all_data_files()
    print(f"Found {len(data_files)} data files to process")
    
    project_root = Path(__file__).parent.parent.parent.parent
    save_dir = project_root / "results" / "plots" / "pipeline" / "landing_derivative"
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
            result = detect_landing_derivative_jumps(
                file_path,
                participant_name,
                params=params,
                save_windows=False,
            )
            
            print(f"Landing derivative algorithm parameters: "
                  f"landing_threshold={params.landing_threshold:.3f}, "
                  f"center_offset={params.center_offset}, "
                  f"search_window={params.search_window}, "
                  f"in_air_threshold={params.in_air_threshold if params.in_air_threshold else 'None'}")
            if ground_truth_markers:
                print(f"Ground truth markers: {len(ground_truth_markers)} markers loaded")
            print(f"Detected jumps: {len(result.jumps)}")
            
            # Load timestamps
            timestamps = load_timestamps(file_path)
            
            # Compute and print precise flight times
            signal = result.pooled_data if result.pooled_data is not None else result.raw_data.sum(axis=1)
            if result.jumps:
                print("\nPrecise jump flight times (landing derivative algorithm):")
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
                print("\nNo jumps detected (landing derivative algorithm)")
            
            # Generate pipeline plot
            if save_plots:
                fig = plot_landing_derivative_pipeline(
                    result,
                    search_window=search_window,
                    ground_truth_markers=ground_truth_markers,
                    show=False,
                )
                save_path = save_dir / f"{participant_name}_pipeline.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved pipeline plot to {save_path}")
                
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
    params = LandingDerivativeParameters(
        landing_threshold=DERIVATIVE_UPPER_DEFAULT,
        center_offset=LANDING_DERIVATIVE_CENTER_OFFSET_DEFAULT,
        search_window=LANDING_DERIVATIVE_SEARCH_WINDOW_DEFAULT,
        in_air_threshold=IN_AIR_THRESHOLD_DEFAULT,
    )
    
    process_all_files(params=params, save_plots=True)

