"""Script to generate all figures needed for the paper."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jump_detection.algorithms.derivative import (
    DerivativeParameters,
    detect_derivative_jumps,
)
from jump_detection.config import DATASET_ROOT, DEFAULT_DATA_FILES
from jump_detection.plotting.snapshots import plot_jump_snapshots


def generate_jump_snapshots():
    """Generate jump snapshots figure using derivative algorithm."""
    print("Generating jump snapshots figure...")
    
    # Use first available dataset file
    data_file = Path(DEFAULT_DATA_FILES[0]) if DEFAULT_DATA_FILES else None
    if not data_file or not data_file.exists():
        # Try to find any data file
        dataset_root = Path(DATASET_ROOT)
        test0_dir = dataset_root / "Test0"
        if test0_dir.exists():
            data_files = list(test0_dir.glob("*10CMJ"))
            if data_files:
                data_file = data_files[0]
            else:
                data_files = [f for f in test0_dir.iterdir() if f.is_file() and not f.name.endswith('.json')]
                if data_files:
                    data_file = data_files[0]
    
    if not data_file or not data_file.exists():
        print(f"Error: Could not find data file. Tried: {data_file}")
        return
    
    print(f"Using data file: {data_file}")
    
    # Use optimal parameters
    params = DerivativeParameters(
        upper_threshold=0.75,
        lower_threshold=-0.25,
        in_air_threshold=130.0,
    )
    
    result = detect_derivative_jumps(
        data_file,
        participant_name=data_file.stem,
        params=params,
        save_windows=False,
    )
    
    # Save path
    project_root = Path(__file__).parent.parent
    save_path = project_root / "results" / "plots" / "jump_snapshots.png"
    
    # Generate and save snapshots
    plot_jump_snapshots(
        result,
        signal=result.pooled_data,
        search_window=70,
        title="Jump Snapshots - Precise Boundary Detection",
        show=False,
        save_path=str(save_path),
    )
    
    print(f"✓ Saved jump snapshots to {save_path}")


if __name__ == "__main__":
    generate_jump_snapshots()

