"""Centralised configuration values for the jump detection toolkit."""

from __future__ import annotations

from pathlib import Path

# Sampling characteristics
SAMPLING_RATE: int = 50

# Threshold algorithm defaults (optimized from combined_loss_comparison.py)
THRESHOLD_DEFAULT: float = 149.23
DERIVATIVE_THRESHOLD_DEFAULT: float = 38.46
DERIVATIVE_WINDOW_DEFAULT: int = 10
EXTRACTION_WINDOW_DEFAULT: int = 300

# Derivative algorithm defaults (optimized from combined_loss_comparison.py)
DERIVATIVE_UPPER_DEFAULT: float = 18
DERIVATIVE_LOWER_DEFAULT: float = -15
IN_AIR_THRESHOLD_DEFAULT: float = 190  # Used in loss_derivative.py and detailedPlot.py

# Correlation algorithm defaults (optimized from combined_loss_comparison.py)
CORRELATION_BUFFER_SIZE_DEFAULT: int = 32
CORRELATION_NEGATIVE_FRAMES_DEFAULT: int = 19
CORRELATION_ZERO_FRAMES_DEFAULT: int = 8
CORRELATION_POSITIVE_FRAMES_DEFAULT: int = 5
CORRELATION_THRESHOLD_DEFAULT: float = 253.06

# Hybrid algorithm defaults
HYBRID_TAKEOFF_THRESHOLD_DEFAULT: float = THRESHOLD_DEFAULT
HYBRID_LANDING_DERIVATIVE_THRESHOLD_DEFAULT: float = 15.0
HYBRID_IN_AIR_THRESHOLD_DEFAULT: float = IN_AIR_THRESHOLD_DEFAULT

# Landing derivative algorithm defaults
LANDING_DERIVATIVE_CENTER_OFFSET_DEFAULT: int = 10
LANDING_DERIVATIVE_SEARCH_WINDOW_DEFAULT: int = 70

# Physics constraints (in seconds)
GRAVITY_FTPS2: float = 32.2
MIN_FLIGHT_TIME_S: float = 0.2  # minimum realistic flight time
MAX_FLIGHT_TIME_S: float = 1.2  # maximum realistic flight time

# Dataset configuration
DATASET_ROOT: Path = Path("dataset")
DEFAULT_DATA_FILES = [
    # Test0 files
    DATASET_ROOT / "Test0" / "DanBraun10sequential",
    DATASET_ROOT / "Test0" / "Darwin10CMJ",
    DATASET_ROOT / "Test0" / "Joey10CMJ",
    DATASET_ROOT / "Test0" / "JoshuaGao10CMJ",
    DATASET_ROOT / "Test0" / "JoshuaGaonatural_interaction_hopper.txt",
    DATASET_ROOT / "Test0" / "JoshuaKerner10CMJ",
    DATASET_ROOT / "Test0" / "MatthewRiley10CMJ",
    # Test2 files
    DATASET_ROOT / "Test2" / "PWG Subject 1.txt",
    DATASET_ROOT / "Test2" / "pwg subject 2.txt",
    DATASET_ROOT / "Test2" / "PWG Subject 3.txt",
    DATASET_ROOT / "Test2" / "PWG subject 5 190lbs-taller.txt",
    DATASET_ROOT / "Test2" / "PWG Subject 6 160 lbs.txt",
    DATASET_ROOT / "Test2" / "PWG subject 7 220 lbs.txt",
    DATASET_ROOT / "Test2" / "PWG subject 8 175.txt",
    DATASET_ROOT / "Test2" / "PWG Subject 9 170.txt",
    # Test3 (video) files
    DATASET_ROOT / "Test3 (video)" / "2scaleup-Matthew.txt",
    DATASET_ROOT / "Test3 (video)" / "Dante-205-5’7-pwgcourt.txt",  # Note: uses curly apostrophe
    DATASET_ROOT / "Test3 (video)" / "DormRoom-Tommy-200lbs-6ft.txt",
    DATASET_ROOT / "Test3 (video)" / "izayah-149-5'7-pwgcourt.txt",  # Note: uses curly apostrophe
    DATASET_ROOT / "Test3 (video)" / "JoshuaScaleup.txt",
    DATASET_ROOT / "Test3 (video)" / "lofty-70kg-5’7-pwgcourt.txt",  # Note: uses curly apostrophe
    DATASET_ROOT / "Test3 (video)" / "Matthew-scaleup-1-10.txt",
    DATASET_ROOT / "Test3 (video)" / "Mouse-subject1- 150lb.txt",
    DATASET_ROOT / "Test3 (video)" / "Mouse-Subject2-145lbs.txt",
    DATASET_ROOT / "Test3 (video)" / "Mouse-subject3-145.txt",
    DATASET_ROOT / "Test3 (video)" / "Mouse-subject4-5’5-135.txt",  # Note: uses curly apostrophe
    DATASET_ROOT / "Test3 (video)" / "Mouse-subject5-5’6~160",  # Note: uses curly apostrophe
    DATASET_ROOT / "Test3 (video)" / "mouse-subject6.txt",
    DATASET_ROOT / "Test3 (video)" / "Noah-130-dorm.txt",
]

# Output directories
RESULTS_ROOT: Path = Path("results")
EXTRACTED_JUMPS_DIR: Path = RESULTS_ROOT / "extracted_jumps" / "threshold"
EXTRACTED_JUMPS_DERIVATIVE_DIR: Path = RESULTS_ROOT / "extracted_jumps" / "derivative"
EXTRACTED_JUMPS_CORRELATION_DIR: Path = RESULTS_ROOT / "extracted_jumps" / "correlation"
EXTRACTED_JUMPS_HYBRID_DIR: Path = RESULTS_ROOT / "extracted_jumps" / "hybrid"
EXTRACTED_JUMPS_ENSEMBLE_DIR: Path = RESULTS_ROOT / "extracted_jumps" / "ensemble"
EXTRACTED_JUMPS_TEMPLATE_DIR: Path = RESULTS_ROOT / "extracted_jumps" / "template"
EXTRACTED_JUMPS_LANDING_DERIVATIVE_DIR: Path = RESULTS_ROOT / "extracted_jumps" / "landing_derivative"

