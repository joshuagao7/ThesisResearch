"""Non-interactive script to generate pipeline figures for the paper."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jump_detection.algorithms.correlation import (
    CorrelationParameters,
    detect_correlation_jumps,
)
from jump_detection.algorithms.derivative import (
    DerivativeParameters,
    detect_derivative_jumps,
)
from jump_detection.algorithms.threshold import (
    ThresholdParameters,
    detect_threshold_jumps,
)
from jump_detection.config import DATASET_ROOT
from scripts.visualization.detailedPlot import create_detailed_plot


def find_representative_data_file():
    """Find a representative data file for generating pipeline figures."""
    dataset_root = Path(DATASET_ROOT)
    test0_dir = dataset_root / "Test0"
    
    if test0_dir.exists():
        # Prefer files with 10CMJ in the name
        data_files = list(test0_dir.glob("*10CMJ"))
        if data_files:
            return data_files[0]
        
        # Otherwise, get any non-json file
        data_files = [f for f in test0_dir.iterdir() 
                     if f.is_file() and not f.name.endswith('.json')]
        if data_files:
            return data_files[0]
    
    raise FileNotFoundError(f"Could not find data file in {test0_dir}")


def generate_pipeline_figures():
    """Generate all three pipeline figures."""
    print("="*80)
    print("GENERATING PIPELINE FIGURES")
    print("="*80)
    
    data_file = find_representative_data_file()
    participant_name = data_file.stem
    print(f"Using data file: {data_file}")
    print(f"Participant: {participant_name}\n")
    
    # Generate threshold pipeline
    print("Generating threshold pipeline figure...")
    create_detailed_plot(
        "threshold",
        data_file,
        participant_name,
        params=ThresholdParameters(threshold=111.0, derivative_threshold=1.0),
        show=False,
    )
    
    # Generate derivative pipeline
    print("\nGenerating derivative pipeline figure...")
    create_detailed_plot(
        "derivative",
        data_file,
        participant_name,
        params=DerivativeParameters(
            upper_threshold=0.75,
            lower_threshold=-0.25,
            in_air_threshold=130.0,
        ),
        show=False,
    )
    
    # Generate correlation pipeline
    print("\nGenerating correlation pipeline figure...")
    create_detailed_plot(
        "correlation",
        data_file,
        participant_name,
        params=CorrelationParameters(),
        show=False,
    )
    
    print("\n" + "="*80)
    print("All pipeline figures generated successfully!")
    print("="*80)


if __name__ == "__main__":
    try:
        generate_pipeline_figures()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

