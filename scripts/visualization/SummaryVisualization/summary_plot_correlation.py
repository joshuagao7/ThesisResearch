"""Correlation-based summary plots and interactive viewer - uses all DEFAULT_DATA_FILES."""

from dataclasses import dataclass
from typing import List, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys

# Unified color palette - car-inspired gradient scheme
PRIMARY_RED = '#C41E3A'      # Deep red
PRIMARY_ORANGE = '#FF6B35'   # Vibrant orange
PRIMARY_YELLOW = '#FFD23F'   # Golden yellow
PRIMARY_BLUE = '#0066CC'     # Electric blue

# Set consistent font styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from jump_detection.algorithms.correlation import process_all_participants_correlation
from jump_detection.analysis.precise import calculate_precise_jump_boundaries
from jump_detection.annotations import load_annotations
from jump_detection.config import DATASET_ROOT, MAX_FLIGHT_TIME_S, SAMPLING_RATE, DEFAULT_DATA_FILES
from jump_detection.types import DetectionResult, Jump


def calculate_jump_height(flight_time_seconds: float, g: float = 32.2) -> float:
    return (1 / 8) * g * (flight_time_seconds**2)


def get_all_data_files() -> List[Path]:
    """Get all data files from DEFAULT_DATA_FILES in config."""
    # Resolve relative paths to absolute paths
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_files = []
    
    for data_file_path in DEFAULT_DATA_FILES:
        if data_file_path.is_absolute():
            full_path = data_file_path
        else:
            # Resolve relative path from project root
            path_str = str(data_file_path)
            if path_str.startswith("dataset/"):
                path_str = path_str[8:]  # Remove "dataset/" prefix
            full_path = project_root / "dataset" / path_str
        
        if full_path.exists():
            data_files.append(full_path)
    
    return sorted(data_files)


def create_summary_plot() -> Sequence:
    """Create summary plot for correlation algorithm using all DEFAULT_DATA_FILES."""
    data_files = get_all_data_files()
    print(f"Processing {len(data_files)} data files for correlation algorithm...")
    results = process_all_participants_correlation(data_files=data_files, save_windows=False)

    fig, axes = plt.subplots(len(results), 1, figsize=(20, 15), sharex=True)
    fig.suptitle("Final Processed Jump Regions - All Participants (Correlation Algorithm)", fontsize=16)

    colors = [PRIMARY_BLUE, PRIMARY_ORANGE, PRIMARY_RED, PRIMARY_YELLOW, "#4A90E2"]

    for idx, result in enumerate(results):
        ax = axes[idx]
        signal = result.pooled_data if result.pooled_data is not None else result.raw_data.sum(axis=1)
        ax.plot(signal, linewidth=1, color=colors[idx % len(colors)], alpha=0.7)

        for jump in result.iter_jumps():
            precise = calculate_precise_jump_boundaries(signal, jump.center)
            precise_start = precise["precise_start"]
            precise_end = precise["precise_end"]
            precise_duration = precise["precise_duration"]

            flight_time = precise_duration / result.sampling_rate
            
            # Filter out jumps that exceed max flight time (physics constraint)
            max_flight_frames = int(MAX_FLIGHT_TIME_S * SAMPLING_RATE)
            if precise_duration > max_flight_frames:
                # Skip this jump - it exceeds physics constraints
                continue
            
            jump_height_inches = calculate_jump_height(flight_time) * 12

            ax.axvspan(precise_start, precise_end, alpha=0.3, color=PRIMARY_YELLOW)
            ymax = ax.get_ylim()[1]
            ax.text(
                precise["precise_center"],
                ymax * 0.9,
                f"{precise_duration}f\n{jump_height_inches:.1f}\"",
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold",
                color=PRIMARY_RED,
            )

            ax.axvspan(jump.start, jump.end, alpha=0.1, color=PRIMARY_BLUE)

        # Load and plot ground truth annotations
        file_path = result.metadata.get('data_file_path', '')
        annotations = None
        if file_path:
            file_path_obj = Path(file_path)
            annotations = load_annotations(file_path_obj)
            if annotations and annotations.markers:
                ymax = ax.get_ylim()[1]
                for marker in annotations.markers:
                    if 0 <= marker < len(signal):
                        ax.axvline(
                            marker,
                            color=PRIMARY_RED,
                            linestyle='--',
                            linewidth=1.5,
                            alpha=0.7,
                            label='Ground Truth' if marker == annotations.markers[0] else ''
                        )
                        # Add a small marker dot at the signal value
                        ax.plot(
                            marker,
                            signal[marker],
                            'r*',
                            markersize=8,
                            markeredgewidth=1,
                            markeredgecolor='darkred',
                            alpha=0.8
                        )

        # Get file name from metadata or construct from path
        if file_path:
            file_name = Path(file_path).name
        else:
            file_name = result.participant_name or "Participant"
        
        # Count jumps that pass physics filter (displayed jumps)
        max_flight_frames = int(MAX_FLIGHT_TIME_S * SAMPLING_RATE)
        displayed_jumps = 0
        for jump in result.iter_jumps():
            precise = calculate_precise_jump_boundaries(signal, jump.center)
            if precise["precise_duration"] <= max_flight_frames:
                displayed_jumps += 1
        
        # Show file name and jump count
        ax.set_ylabel(f"{file_name}\n({displayed_jumps} jumps)", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add legend if we have ground truth markers
        if annotations and annotations.markers:
            ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel("Frames", fontsize=12)
    plt.tight_layout()
    
    # Save figure
    project_root = Path(__file__).parent.parent.parent.parent
    save_path = project_root / "results" / "plots" / "summary_plot_correlation.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot to {save_path}")
    
    plt.show()
    
    # Create false negative and false positive diagnostic plots
    create_error_diagnostic_plots(results)
    
    return results


def find_missed_jumps(
    result: DetectionResult,
    ground_truth_markers: list[int] | None,
) -> tuple[list[int], list[Jump]]:
    """Find jumps that were missed by the algorithm."""
    if not ground_truth_markers:
        return [], []
    
    detected_jumps = list(result.jumps)
    missed_markers: list[int] = []
    false_positive_jumps: list[Jump] = []
    
    for marker in ground_truth_markers:
        marker_contained = False
        for jump in detected_jumps:
            if jump.start <= marker <= jump.end:
                marker_contained = True
                break
        if not marker_contained:
            missed_markers.append(marker)
    
    for jump in detected_jumps:
        jump_contains_marker = False
        for marker in ground_truth_markers:
            if jump.start <= marker <= jump.end:
                jump_contains_marker = True
                break
        if not jump_contains_marker:
            false_positive_jumps.append(jump)
    
    return missed_markers, false_positive_jumps


def create_error_diagnostic_plots(results: Sequence) -> None:
    """Create diagnostic plots showing false negatives and false positives."""
    all_false_negatives: list[tuple[Path, int, np.ndarray]] = []
    all_false_positives: list[tuple[Path, Jump, np.ndarray]] = []
    
    for result in results:
        file_path = result.metadata.get('data_file_path', '')
        if not file_path:
            continue
        
        file_path_obj = Path(file_path)
        annotations = load_annotations(file_path_obj)
        if not annotations or not annotations.markers:
            continue
        
        signal = result.pooled_data if result.pooled_data is not None else result.raw_data.sum(axis=1)
        missed_markers, false_positive_jumps = find_missed_jumps(result, annotations.markers)
        
        for marker in missed_markers:
            all_false_negatives.append((file_path_obj, marker, signal))
        
        for jump in false_positive_jumps:
            all_false_positives.append((file_path_obj, jump, signal))
    
    # Plot false negatives
    if all_false_negatives:
        num_false_negatives = len(all_false_negatives)
        cols = min(4, num_false_negatives)
        rows = (num_false_negatives + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True)
        fig.suptitle(
            f"False Negatives (Missed Ground Truth Markers) - Correlation Algorithm\n"
            f"Total: {num_false_negatives}",
            fontsize=14,
            fontweight="bold"
        )
        
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        window_frames = 150
        for idx, (file_path, marker, signal) in enumerate(all_false_negatives):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            
            start_frame = max(0, marker - window_frames)
            end_frame = min(len(signal), marker + window_frames)
            frames = np.arange(start_frame, end_frame)
            
            ax.plot(frames, signal[start_frame:end_frame], linewidth=2, color=PRIMARY_BLUE, alpha=0.8, label="Signal")
            ax.axvline(marker, color=PRIMARY_RED, linestyle="--", linewidth=2, alpha=0.9, label="Missed Marker")
            ax.axvspan(max(0, marker - 20), min(len(signal), marker + 20), alpha=0.2, color=PRIMARY_RED)
            
            file_name = file_path.name
            ax.set_title(f"{file_name}\nFrame {marker}", fontsize=10)
            ax.set_xlabel("Frames (relative to marker)")
            ax.set_ylabel("Signal Value")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        for idx in range(num_false_negatives, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            ax.axis("off")
        
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo false negatives found (all ground truth markers were detected)!")
    
    # Plot false positives
    if all_false_positives:
        num_false_positives = len(all_false_positives)
        cols = min(4, num_false_positives)
        rows = (num_false_positives + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True)
        fig.suptitle(
            f"False Positives (Detected Jumps Without Ground Truth) - Correlation Algorithm\n"
            f"Total: {num_false_positives}",
            fontsize=14,
            fontweight="bold"
        )
        
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        window_frames = 150
        for idx, (file_path, jump, signal) in enumerate(all_false_positives):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            
            start_frame = max(0, jump.center - window_frames)
            end_frame = min(len(signal), jump.center + window_frames)
            frames = np.arange(start_frame, end_frame)
            
            ax.plot(frames, signal[start_frame:end_frame], linewidth=2, color=PRIMARY_ORANGE, alpha=0.8, label="Signal")
            ax.axvspan(jump.start - start_frame, jump.end - start_frame, alpha=0.3, color=PRIMARY_ORANGE, label="False Positive Jump")
            ax.axvline(jump.center - start_frame, color=PRIMARY_ORANGE, linestyle="--", linewidth=2, alpha=0.9)
            
            file_name = file_path.name
            ax.set_title(f"{file_name}\nFrames {jump.start}-{jump.end}", fontsize=10)
            ax.set_xlabel("Frames (relative to jump center)")
            ax.set_ylabel("Signal Value")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        for idx in range(num_false_positives, rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            ax.axis("off")
        
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo false positives found (all detected jumps contain ground truth markers)!")
    
    # Print summary statistics
    total_markers = sum(
        len(load_annotations(Path(r.metadata.get('data_file_path', ''))).markers)
        for r in results
        if r.metadata.get('data_file_path') and load_annotations(Path(r.metadata.get('data_file_path', '')))
    )
    total_detected = sum(len(r.jumps) for r in results)
    
    print(f"\n{'='*80}")
    print(f"Error Summary - Correlation Algorithm")
    print(f"{'='*80}")
    print(f"Total ground truth markers: {total_markers}")
    print(f"Total detected jumps: {total_detected}")
    print(f"False negatives (missed markers): {len(all_false_negatives)}")
    print(f"False positives (extra detections): {len(all_false_positives)}")
    if total_markers > 0:
        recall = (total_markers - len(all_false_negatives)) / total_markers
        print(f"Recall: {recall:.3f} ({recall*100:.1f}%)")
    if total_detected > 0:
        precision = (total_detected - len(all_false_positives)) / total_detected
        print(f"Precision: {precision:.3f} ({precision*100:.1f}%)")
    print(f"{'='*80}")


@dataclass
class _JumpEntry:
    participant_name: str
    signal: np.ndarray
    jump_index: int
    jump_start: int
    jump_end: int
    jump_center: int
    precise: dict


class InteractiveJumpViewer:
    def __init__(self, results: Sequence):
        self.results = results
        self.all_jumps: List[_JumpEntry] = self._collect_jumps()
        self.current_jump = 0

        print(f"Total jumps to view: {len(self.all_jumps)}")
        self._create_interactive_plot()

    def _collect_jumps(self) -> List[_JumpEntry]:
        entries: List[_JumpEntry] = []
        max_flight_frames = int(MAX_FLIGHT_TIME_S * SAMPLING_RATE)
        for result in self.results:
            signal = result.pooled_data if result.pooled_data is not None else result.raw_data.sum(axis=1)
            participant = result.participant_name or "Participant"
            for idx, jump in enumerate(result.iter_jumps(), start=1):
                precise = calculate_precise_jump_boundaries(signal, jump.center)
                if precise["precise_duration"] > max_flight_frames:
                    continue
                entries.append(
                    _JumpEntry(
                        participant_name=participant,
                        signal=signal,
                        jump_index=idx,
                        jump_start=jump.start,
                        jump_end=jump.end,
                        jump_center=jump.center,
                        precise=precise,
                    )
                )
        return entries

    def _create_interactive_plot(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.suptitle(
            "Interactive Jump Viewer (Correlation) - Use ← → arrows to navigate, ESC to close",
            fontsize=14,
        )
        self._update_plot()
        plt.show()

    def _update_plot(self) -> None:
        self.ax.clear()
        if not self.all_jumps:
            return

        entry = self.all_jumps[self.current_jump]
        window = 150
        center = entry.precise["precise_center"]
        start_frame = max(0, center - window)
        end_frame = min(len(entry.signal), center + window)
        frames = np.arange(start_frame, end_frame)
        self.ax.plot(frames, entry.signal[start_frame:end_frame], color=PRIMARY_BLUE, linewidth=2)

        self.ax.axvspan(entry.precise["precise_start"], entry.precise["precise_end"], alpha=0.3, color=PRIMARY_YELLOW)
        self.ax.axvspan(entry.jump_start, entry.jump_end, alpha=0.1, color=PRIMARY_BLUE)
        self.ax.axvline(center, color=PRIMARY_RED, linestyle="--", alpha=0.7)
        self.ax.axhline(entry.precise["half_peak_before"], color=PRIMARY_YELLOW, linestyle=":", alpha=0.7)
        self.ax.axhline(entry.precise["half_peak_after"], color=PRIMARY_ORANGE, linestyle=":", alpha=0.7)
        self.ax.plot(entry.precise["peak_before_idx"], entry.precise["peak_before"], "o", color=PRIMARY_YELLOW, markersize=8)
        self.ax.plot(entry.precise["peak_after_idx"], entry.precise["peak_after"], "o", color=PRIMARY_ORANGE, markersize=8)

        self.ax.set_xlabel("Frame Number", fontsize=12)
        self.ax.set_ylabel("Average Sensor Value", fontsize=12)
        self.ax.set_title(
            f"{entry.participant_name} - Jump {entry.jump_index} (Overall {self.current_jump + 1} of {len(self.all_jumps)})",
            fontsize=14,
        )

        original_duration = entry.jump_end - entry.jump_start
        precise_duration = entry.precise["precise_duration"]
        original_time = original_duration / 50
        precise_time = precise_duration / 50
        original_height = calculate_jump_height(original_time) * 12
        precise_height = calculate_jump_height(precise_time) * 12

        stats_text = (
            f"Original Duration: {original_duration} frames\n"
            f"Precise Duration: {precise_duration} frames\n"
            f"Original Flight Time: {original_time:.3f} s\n"
            f"Precise Flight Time: {precise_time:.3f} s\n"
            f"Original Jump Height: {original_height:.1f}\"\n"
            f"Precise Jump Height: {precise_height:.1f}\""
        )

        self.ax.text(
            0.02,
            0.98,
            stats_text,
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        self.ax.grid(True, alpha=0.3)
        self.fig.canvas.draw()

    def _on_key_press(self, event) -> None:
        if event.key == "right" and self.current_jump < len(self.all_jumps) - 1:
            self.current_jump += 1
            self._update_plot()
        elif event.key == "left" and self.current_jump > 0:
            self.current_jump -= 1
            self._update_plot()
        elif event.key == "escape":
            plt.close(self.fig)


def create_interactive_jump_viewer() -> None:
    """Create interactive jump viewer for correlation algorithm."""
    print("Loading data for interactive jump viewer (Correlation)...")
    data_files = get_all_data_files()
    results = process_all_participants_correlation(data_files=data_files, save_windows=False)
    InteractiveJumpViewer(results)


if __name__ == "__main__":
    print("Creating correlation summary plot...")
    create_summary_plot()
    print("\nStarting correlation interactive jump viewer...")
    create_interactive_jump_viewer()

