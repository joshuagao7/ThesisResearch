"""Interactive 3D visualization of sensor data as a scrollable video.

This script allows you to view raw sensor data arranged in a 6x8 grid mesh,
where the height of each sensor represents its pressure value. You can scroll
through frames using keyboard controls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap

PRIMARY_RED = '#C41E3A'      # Deep red
PRIMARY_ORANGE = '#FF6B35'   # Vibrant orange
PRIMARY_YELLOW = '#FFD23F'   # Golden yellow
PRIMARY_BLUE = '#0066CC'     # Electric blue

# Create gradient colormap
GRADIENT_COLORS = [PRIMARY_RED, PRIMARY_ORANGE, PRIMARY_YELLOW, PRIMARY_BLUE]
GRADIENT_CMAP = LinearSegmentedColormap.from_list('red_orange_yellow_blue', GRADIENT_COLORS, N=100)

# Set consistent font styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

from jump_detection.config import DATASET_ROOT
from jump_detection.data import load_dataset


class SensorVideoViewer:
    """Interactive viewer for scrolling through sensor data frames."""
    
    def __init__(self, data: np.ndarray, participant_name: str, file_name: str):
        """
        Initialize the viewer.
        
        Args:
            data: Sensor data array of shape (n_frames, 48)
            participant_name: Name of the participant
            file_name: Name of the data file
        """
        self.data = data
        self.participant_name = participant_name
        self.file_name = file_name
        self.current_frame = 0
        self.n_frames = len(data)
        
        # Sensor grid dimensions: 6 rows x 8 columns = 48 sensors
        self.grid_rows = 6
        self.grid_cols = 8
        
        # Create mesh grid for 3D plotting
        self.x = np.arange(self.grid_cols)
        self.y = np.arange(self.grid_rows)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Setup figure with space for slider at bottom
        self.fig = plt.figure(figsize=(14, 11))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Fixed scale limits
        self.z_min = 0
        self.z_max = 255
        self.vmin = 0
        self.vmax = 255
        
        # Animation state
        self.is_playing = False
        self.animation_interval = 50  # milliseconds between frames (20 fps)
        self.anim = None
        self.colorbar = None
        self.updating_slider = False  # Flag to prevent recursion
        
        # Create slider at the bottom
        self.fig.subplots_adjust(bottom=0.15)
        ax_slider = plt.axes([0.1, 0.02, 0.8, 0.03])
        self.slider = Slider(
            ax_slider, 'Frame', 0, self.n_frames - 1,
            valinit=0, valstep=1, valfmt='%d'
        )
        self.slider.on_changed(self.on_slider_change)
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initial plot
        self.update_frame()
        
    def reshape_to_grid(self, frame_data: np.ndarray) -> np.ndarray:
        """
        Reshape 48 sensor values to 6x8 grid.
        
        Args:
            frame_data: Array of 48 sensor values
            
        Returns:
            6x8 array representing sensor grid
        """
        if len(frame_data) != 48:
            raise ValueError(f"Expected 48 sensor values, got {len(frame_data)}")
        
        # Reshape to 6x8 grid (row-major order)
        grid = frame_data.reshape(self.grid_rows, self.grid_cols)
        return grid
    
    def update_frame(self):
        """Update the 3D plot with current frame data."""
        # Remove old colorbar if it exists
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        
        self.ax.clear()
        
        # Get current frame data
        frame_data = self.data[self.current_frame]
        
        # Reshape to grid
        Z = self.reshape_to_grid(frame_data)
        
        # Create 3D surface plot with fixed color range
        surf = self.ax.plot_surface(
            self.X, self.Y, Z,
            cmap=GRADIENT_CMAP,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.9,
            antialiased=True,
            vmin=self.vmin,
            vmax=self.vmax
        )
        
        # Set labels and title
        self.ax.set_xlabel('Column (8 sensors)', fontsize=12, labelpad=10)
        self.ax.set_ylabel('Row (6 sensors)', fontsize=12, labelpad=10)
        self.ax.set_zlabel('Pressure Value', fontsize=12, labelpad=10)
        
        # Set title with frame information
        title = (
            f'Sensor Grid Visualization - {self.participant_name} / {self.file_name}\n'
            f'Frame {self.current_frame + 1} / {self.n_frames} '
            f'(Time: {self.current_frame / 50:.2f}s)'
        )
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Set axis limits
        self.ax.set_xlim(-0.5, self.grid_cols - 0.5)
        self.ax.set_ylim(-0.5, self.grid_rows - 0.5)
        
        # Set fixed z-axis limits from 0 to 255
        self.ax.set_zlim(self.z_min, self.z_max)
        
        # Set tick marks
        self.ax.set_xticks(np.arange(self.grid_cols))
        self.ax.set_yticks(np.arange(self.grid_rows))
        
        # Add colorbar with fixed range (range is set via vmin/vmax in plot_surface)
        self.colorbar = self.fig.colorbar(
            surf, ax=self.ax, shrink=0.5, aspect=20, pad=0.1
        )
        self.colorbar.set_label('Pressure Value', rotation=270, labelpad=20)
        
        # Add instructions
        play_status = "Playing" if self.is_playing else "Paused"
        instructions = (
            f"Controls:\n"
            f"Slider : Drag to navigate frames\n"
            f"← → : Previous/Next frame\n"
            f"Home/End : First/Last frame\n"
            f"Space : Play/Pause ({play_status})\n"
            f"Q/Esc : Quit"
        )
        self.fig.text(0.02, 0.02, instructions, fontsize=9, 
                     verticalalignment='bottom', family='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.draw()
    
    def animate(self, frame):
        """Animation function called by matplotlib animation."""
        if self.is_playing:
            self.current_frame = (self.current_frame + 1) % self.n_frames
            self.update_slider(self.current_frame)
            self.update_frame()
        return []
    
    def toggle_play(self):
        """Toggle play/pause animation."""
        self.is_playing = not self.is_playing
        if self.is_playing and self.anim is None:
            # Start animation
            self.anim = animation.FuncAnimation(
                self.fig, self.animate, interval=self.animation_interval,
                blit=False, repeat=True
            )
        self.update_frame()
    
    def on_slider_change(self, val):
        """Handle slider value change."""
        if self.updating_slider:
            return
        frame = int(self.slider.val)
        if frame != self.current_frame:
            self.current_frame = frame
            self.update_frame()
    
    def update_slider(self, frame):
        """Update slider position without triggering callback."""
        self.updating_slider = True
        self.slider.set_val(frame)
        self.updating_slider = False
    
    def on_key_press(self, event):
        """Handle keyboard events for frame navigation."""
        if event.key == 'right' or event.key == 'd':
            if self.is_playing:
                self.toggle_play()  # Pause first
            self.current_frame = min(self.current_frame + 1, self.n_frames - 1)
            self.update_slider(self.current_frame)
            self.update_frame()
        elif event.key == 'left' or event.key == 'a':
            if self.is_playing:
                self.toggle_play()  # Pause first
            self.current_frame = max(self.current_frame - 1, 0)
            self.update_slider(self.current_frame)
            self.update_frame()
        elif event.key == 'home':
            if self.is_playing:
                self.toggle_play()  # Pause first
            self.current_frame = 0
            self.update_slider(self.current_frame)
            self.update_frame()
        elif event.key == 'end':
            if self.is_playing:
                self.toggle_play()  # Pause first
            self.current_frame = self.n_frames - 1
            self.update_slider(self.current_frame)
            self.update_frame()
        elif event.key == 'q' or event.key == 'escape':
            plt.close(self.fig)
        elif event.key == ' ':
            self.toggle_play()


def _prompt_selection(options: list[str], prompt_message: str) -> int:
    """Prompt user to select from a list of options."""
    while True:
        choice = input(prompt_message).strip()
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(options):
                return index
        print("Invalid selection. Please enter one of the listed numbers.")


def select_dataset_path(dataset_root: Path = DATASET_ROOT) -> Tuple[Path, str]:
    """
    Interactive selection of dataset folder and file.
    
    Args:
        dataset_root: Root directory containing participant folders
        
    Returns:
        Tuple of (selected_file_path, participant_name)
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    participants = sorted([p for p in root.iterdir() if p.is_dir()])
    if not participants:
        raise FileNotFoundError(f"No participant folders found in {root}")

    participant_names = [path.name for path in participants]
    print("\nAvailable participants:")
    for idx, name in enumerate(participant_names, start=1):
        print(f"  {idx}. {name}")

    participant_index = _prompt_selection(
        participant_names, "\nSelect a participant by number: "
    )
    participant_path = participants[participant_index]

    data_files = sorted(
        [path for path in participant_path.iterdir() if path.is_file()]
    )
    if not data_files:
        raise FileNotFoundError(
            f"No data files found for participant '{participant_path.name}'"
        )

    file_labels = [path.name for path in data_files]
    print(f"\nAvailable files for {participant_path.name}:")
    for idx, name in enumerate(file_labels, start=1):
        print(f"  {idx}. {name}")

    file_index = _prompt_selection(
        file_labels, "\nSelect a file by number: "
    )

    return data_files[file_index], participant_path.name


def main():
    """Main function to run the interactive video viewer."""
    try:
        print("=" * 60)
        print("Sensor Data Video Viewer")
        print("=" * 60)
        
        dataset_path, participant = select_dataset_path()
        
        print(f"\nLoading data from: {dataset_path}")
        data = load_dataset(dataset_path)
        
        print(f"Loaded {len(data)} frames with {data.shape[1]} sensors")
        print(f"Data range: {np.min(data):.2f} to {np.max(data):.2f}")
        
        print("\n" + "=" * 60)
        print("Starting visualization...")
        print("Use arrow keys to navigate frames")
        print("=" * 60 + "\n")
        
        viewer = SensorVideoViewer(data, participant, dataset_path.name)
        plt.show()
        
    except (KeyboardInterrupt, EOFError):
        print("\n\nSelection cancelled.")
    except FileNotFoundError as error:
        print(f"\nError: {error}")
    except Exception as error:
        print(f"\nUnexpected error: {error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

