"""Interactive GUI tool for marking ground truth jump annotations."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

# Unified color palette - car-inspired gradient scheme
PRIMARY_RED = '#C41E3A'      # Deep red
PRIMARY_ORANGE = '#FF6B35'   # Vibrant orange
PRIMARY_YELLOW = '#FFD23F'   # Golden yellow
PRIMARY_BLUE = '#0066CC'     # Electric blue

# Set consistent font styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

from jump_detection.algorithms.derivative import (
    DerivativeParameters,
    detect_derivative_jumps,
)
from jump_detection.annotations import load_annotations, save_annotations
from jump_detection.config import DATASET_ROOT, DEFAULT_DATA_FILES
from jump_detection.types import GroundTruthAnnotations


class JumpAnnotator:
    """Interactive GUI for marking jump annotations."""

    def __init__(self, data_file_path: Path):
        self.data_file_path = Path(data_file_path)
        self.fig = None
        self.ax = None
        self.pooled_data = None
        self.markers: list[int] = []
        self.detected_jumps = None
        
        # Load existing annotations if available
        annotations = load_annotations(self.data_file_path)
        if annotations:
            self.markers = annotations.markers.copy()
            print(f"Loaded {len(self.markers)} existing markers")
        
        # Run detection to show on plot
        result = detect_derivative_jumps(
            self.data_file_path,
            participant_name=None,
            params=DerivativeParameters(in_air_threshold=250.0),
            save_windows=False,
        )
        self.pooled_data = result.pooled_data
        self.detected_jumps = result.jumps

    def plot_data(self):
        """Plot the pooled signal data with markers and detected jumps."""
        self.fig, self.ax = plt.subplots(figsize=(16, 8))
        
        # Plot pooled signal
        self.ax.plot(self.pooled_data, linewidth=1, color=PRIMARY_BLUE, alpha=0.7, label='Pooled Signal')
        
        # Plot detected jumps (blue highlights)
        for jump in self.detected_jumps:
            self.ax.axvspan(
                jump.start,
                jump.end,
                alpha=0.1,
                color=PRIMARY_BLUE,
                label='Detected Jump' if jump == self.detected_jumps[0] else '',
            )
        
        # Plot markers (red vertical lines)
        for marker in self.markers:
            self.ax.axvline(
                marker,
                color=PRIMARY_RED,
                linewidth=2,
                alpha=0.7,
                linestyle='--',
                label='Ground Truth Marker' if marker == self.markers[0] else '',
            )
            self.ax.plot(
                marker,
                self.pooled_data[marker] if marker < len(self.pooled_data) else 0,
                '*',
                color=PRIMARY_RED,
                markersize=15,
            )
        
        self.ax.set_xlabel('Frame Index', fontsize=12)
        self.ax.set_ylabel('Pooled Signal Value', fontsize=12)
        self.ax.set_title(
            f'Jump Annotator - {self.data_file_path.name}\n'
            f'Markers: {len(self.markers)} | Detected: {len(self.detected_jumps)}',
            fontsize=14,
            fontweight='bold',
        )
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')
        
        # Add instructions text
        instructions = (
            'Instructions:\n'
            'Left-click: Add marker at clicked frame\n'
            'Right-click: Remove nearest marker\n'
            's: Save annotations\n'
            'q: Quit (saves automatically)\n'
            'Arrow keys: Navigate'
        )
        self.fig.text(
            0.02,
            0.02,
            instructions,
            fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )
        
        plt.tight_layout()
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_click(self, event):
        """Handle mouse clicks."""
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # Left click - add marker
            frame = int(round(event.xdata))
            if 0 <= frame < len(self.pooled_data):
                if frame not in self.markers:
                    self.markers.append(frame)
                    self.markers.sort()
                    print(f"Added marker at frame {frame}")
                    self.update_plot()
        elif event.button == 3:  # Right click - remove nearest marker
            if self.markers:
                frame = int(round(event.xdata))
                nearest = min(self.markers, key=lambda m: abs(m - frame))
                if abs(nearest - frame) < 50:  # Only remove if click is close
                    self.markers.remove(nearest)
                    print(f"Removed marker at frame {nearest}")
                    self.update_plot()

    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 's':  # Save
            self.save_annotations()
        elif event.key == 'q':  # Quit
            self.save_annotations()
            plt.close(self.fig)
        elif event.key == 'left':
            # Navigate left
            xlim = self.ax.get_xlim()
            width = xlim[1] - xlim[0]
            self.ax.set_xlim(xlim[0] - width * 0.5, xlim[1] - width * 0.5)
            self.fig.canvas.draw()
        elif event.key == 'right':
            # Navigate right
            xlim = self.ax.get_xlim()
            width = xlim[1] - xlim[0]
            self.ax.set_xlim(xlim[0] + width * 0.5, xlim[1] + width * 0.5)
            self.fig.canvas.draw()

    def update_plot(self):
        """Update the plot with current markers."""
        self.ax.clear()
        
        # Replot signal
        self.ax.plot(self.pooled_data, linewidth=1, color=PRIMARY_BLUE, alpha=0.7, label='Pooled Signal')
        
        # Replot detected jumps
        for jump in self.detected_jumps:
            self.ax.axvspan(
                jump.start,
                jump.end,
                alpha=0.1,
                color=PRIMARY_BLUE,
                label='Detected Jump' if jump == self.detected_jumps[0] else '',
            )
        
        # Replot markers
        for marker in self.markers:
            self.ax.axvline(
                marker,
                color=PRIMARY_RED,
                linewidth=2,
                alpha=0.7,
                linestyle='--',
                label='Ground Truth Marker' if marker == self.markers[0] else '',
            )
            self.ax.plot(
                marker,
                self.pooled_data[marker] if marker < len(self.pooled_data) else 0,
                'r*',
                markersize=15,
            )
        
        self.ax.set_xlabel('Frame Index', fontsize=12)
        self.ax.set_ylabel('Pooled Signal Value', fontsize=12)
        self.ax.set_title(
            f'Jump Annotator - {self.data_file_path.name}\n'
            f'Markers: {len(self.markers)} | Detected: {len(self.detected_jumps)}',
            fontsize=14,
            fontweight='bold',
        )
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')
        
        self.fig.canvas.draw()

    def save_annotations(self):
        """Save annotations to file."""
        annotations = GroundTruthAnnotations(
            data_file_path=str(self.data_file_path),
            markers=self.markers.copy(),
            created_at=datetime.now().isoformat(),
            modified_at=datetime.now().isoformat(),
        )
        
        annotation_path = save_annotations(annotations, self.data_file_path)
        print(f"Saved {len(self.markers)} markers to {annotation_path}")

    def run(self):
        """Run the annotation GUI."""
        self.plot_data()
        plt.show()


def select_dataset_path(dataset_root: Path = DATASET_ROOT) -> tuple[Path, str]:
    """Select a data file from Test0, Test1(100Hz), Test2, or Test3 (video) folders."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    # Look in Test0, Test1(100Hz), Test2, and Test3 (video) folders
    test0_dir = root / "Test0"
    test1_dir = root / "Test1(100Hz)"
    test2_dir = root / "Test2"
    test3_dir = root / "Test3 (video)"
    
    data_files = []
    
    if test0_dir.exists():
        # Get all files in Test0 (both with and without extensions)
        test0_files = sorted([f for f in test0_dir.iterdir() if f.is_file()])
        data_files.extend(test0_files)
    
    if test1_dir.exists():
        # Get all .txt files in Test1(100Hz)
        test1_files = sorted([f for f in test1_dir.glob("*.txt")])
        data_files.extend(test1_files)
    
    if test2_dir.exists():
        # Get all .txt files in Test2
        test2_files = sorted([f for f in test2_dir.glob("*.txt")])
        data_files.extend(test2_files)
    
    if test3_dir.exists():
        # Get all .txt files in Test3 (video)
        test3_files = sorted([f for f in test3_dir.glob("*.txt")])
        data_files.extend(test3_files)
    
    if not data_files:
        raise FileNotFoundError(f"No data files found in Test0, Test1(100Hz), Test2, or Test3 (video) folders")

    print("Available data files (Test0, Test1(100Hz), Test2, and Test3 (video)):")
    for idx, path in enumerate(data_files, start=1):
        relative_path = path.relative_to(root)
        print(f"  {idx}. {relative_path}")

    while True:
        try:
            choice = input("\nSelect a file by number: ").strip()
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(data_files):
                    selected = data_files[index]
                    participant = selected.parent.name if selected.parent != selected.parent.parent else "Dataset"
                    return selected, participant
            print("Invalid selection. Please enter one of the listed numbers.")
        except (KeyboardInterrupt, EOFError):
            raise


def main():
    """Main entry point for the annotation tool.
    
    Loops through all files in DEFAULT_DATA_FILES from config one by one.
    """
    try:
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        
        # Use DEFAULT_DATA_FILES from config, resolve paths
        data_files = []
        for data_file_path in DEFAULT_DATA_FILES:
            if not data_file_path.is_absolute():
                full_path = PROJECT_ROOT / data_file_path
            else:
                full_path = data_file_path
            full_path = full_path.resolve()
            
            if full_path.exists():
                data_files.append(full_path)
        
        if not data_files:
            print("No data files found in DEFAULT_DATA_FILES.")
            return

        print(f"\nFound {len(data_files)} files to annotate.")
        print("You will see each file one by one. Close each plot window to move to the next file.")
        print("Annotations are saved automatically after each file.\n")

        for idx, dataset_path in enumerate(data_files, start=1):
            relative_path = dataset_path.relative_to(PROJECT_ROOT)
            print(f"\n{'='*70}")
            print(f"File {idx}/{len(data_files)}: {relative_path}")
            print(f"{'='*70}")
            print("Instructions:")
            print("  - Left-click: Add marker at clicked frame")
            print("  - Right-click: Remove nearest marker")
            print("  - s: Save annotations manually")
            print("  - Close window: Move to next file (saves automatically)")
            print()
            
            try:
                annotator = JumpAnnotator(dataset_path)
                annotator.run()
                
                # Save annotations after each file
                annotator.save_annotations()
                
            except KeyboardInterrupt:
                print(f"\nSkipping {relative_path} and continuing to next file...")
                continue
            except Exception as e:
                print(f"Error processing {relative_path}: {e}")
                print("Continuing to next file...")
                continue

        print(f"\n{'='*70}")
        print(f"Completed annotating {len(data_files)} files.")
        print(f"{'='*70}")
        
    except (KeyboardInterrupt, EOFError):
        print("\n\nAnnotation cancelled.")
    except FileNotFoundError as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()

