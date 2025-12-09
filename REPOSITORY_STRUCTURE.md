# Repository Structure and Functionality Summary

## Overview

This repository implements a jump detection system using sensor data analysis. The codebase is organized into a core library (`jump_detection/`) and analysis scripts (`scripts/`). The system provides three distinct jump detection algorithms and universal analysis functions that work across all algorithms.

---

## Core Architecture

### Package Structure

```
jump_detection/
├── algorithms/          # Three jump detection algorithms
│   ├── threshold.py    # Threshold-based detection
│   ├── derivative.py   # Derivative-based detection
│   └── correlation.py  # Correlation-based detection
├── analysis/           # Universal analysis functions
│   ├── loss.py         # Precision loss calculation
│   └── precise.py      # Precise boundary detection
├── data.py             # Data loading and export utilities
├── config.py           # Centralized configuration
├── types.py            # Type definitions (Jump, DetectionResult, etc.)
├── annotations.py      # Ground truth annotation handling
└── plotting/           # Visualization utilities
```

---

## Three Jump Detection Algorithms

All three algorithms follow a **consistent structural pattern**:

1. **Parameters Dataclass** - Encapsulates algorithm-specific parameters
2. **Main Detection Function** - `detect_*_jumps()` - Primary entry point
3. **Convenience Function** - `detect_*_jumps_with_params()` - Direct parameter passing
4. **Batch Processing** - `process_all_*_participants()` - Process multiple files
5. **Internal Pipeline** - `_run_*_pipeline()` - Core algorithm logic

All algorithms return a `DetectionResult` object containing:
- `raw_data`: Original sensor data
- `pooled_data`: Aggregated signal (sum/average/range)
- `signals`: Dictionary of intermediate processing signals
- `jumps`: List of detected `Jump` objects
- `metadata`: Algorithm-specific metadata

### 1. Threshold Algorithm (`jump_detection/algorithms/threshold.py`)

**Purpose**: Detects jumps based on pressure threshold crossing and derivative analysis.

**Parameters** (`ThresholdParameters`):
- `threshold`: Pressure threshold (default: 111)
- `derivative_threshold`: Derivative magnitude threshold (default: 1.0)
- `derivative_window`: Window size for motion detection (default: 10)
- `min_flight_time`: Minimum realistic flight time in seconds (default: 0.2)
- `max_flight_time`: Maximum realistic flight time in seconds (default: 1.2)
- `extraction_window`: Window size for jump extraction (default: 300)

**Pipeline** (`_run_threshold_pipeline`):
1. **Pool Signal**: Sum all sensor channels → `average`
2. **Threshold Mask**: Create binary mask where `average < threshold`
3. **Derivative Calculation**: Compute gradient of average signal
4. **Derivative Binary**: Create mask for high derivative magnitude
5. **Physics Filtering**: Remove segments outside realistic flight time bounds
6. **Jump Extraction**: Convert remaining segments to `Jump` objects

**Signals Generated**:
- `raw_data`: Original multi-channel sensor data
- `average`: Sum of all channels
- `threshold_mask`: Binary mask of below-threshold regions
- `physics_filtered`: After physics constraints applied
- `derivative`: Gradient of average signal
- `derivative_binary`: High-magnitude derivative regions

**Key Functions**:
- `detect_threshold_jumps()`: Main detection function
- `detect_threshold_jumps_with_params()`: Convenience wrapper
- `process_all_threshold_participants()`: Batch processing

---

### 2. Derivative Algorithm (`jump_detection/algorithms/derivative.py`)

**Purpose**: Detects jumps by pairing negative and positive derivative threshold crossings.

**Parameters** (`DerivativeParameters`):
- `upper_threshold`: Positive derivative threshold (default: 0.75)
- `lower_threshold`: Negative derivative threshold (default: -0.25)
- `in_air_threshold`: Pressure threshold for in-air detection (default: 130)
- `min_flight_time`: Minimum flight time (default: 0.2s)
- `max_flight_time`: Maximum flight time (default: 1.2s)

**Pipeline** (`_run_derivative_pipeline`):
1. **Pool Signal**: Sum all sensor channels → `pooled`
2. **Derivative Calculation**: Compute gradient of pooled signal
3. **Upper/Lower Masks**: Create masks for positive/negative threshold crossings
4. **Pair Threshold Crossings**: Match lower crossings (takeoff) with upper crossings (landing)
5. **Physics Filtering**: Filter pairs by flight time constraints
6. **In-Air Validation**: Require pooled signal below `in_air_threshold` between takeoff and landing
7. **Jump Extraction**: Convert valid pairs to `Jump` objects

**Signals Generated**:
- `raw_data`: Original multi-channel sensor data
- `pooled`: Sum of all channels
- `derivative`: Gradient of pooled signal
- `derivative_upper`: Positive threshold crossings
- `derivative_lower`: Negative threshold crossings
- `derivative_pair_indicator`: Markers for paired crossings
- `in_air`: Binary mask of below-threshold regions
- `valid_pair_indicator`: Markers for validated jump pairs

**Key Functions**:
- `detect_derivative_jumps()`: Main detection function
- `detect_derivative_jumps_with_params()`: Convenience wrapper
- `process_all_derivative_participants()`: Batch processing

**Internal Helpers**:
- `_pair_threshold_crossings()`: Matches lower and upper crossings with time constraints
- `_filter_pairs_with_flight()`: Validates pairs have in-air region

---

### 3. Correlation Algorithm (`jump_detection/algorithms/correlation.py`)

**Purpose**: Detects jumps based on template correlation matching.

**Parameters** (`CorrelationParameters`):
- Template-based correlation matching parameters
- Window size configurations for correlation analysis

**Pipeline** (`_run_correlation_pipeline`):
1. **Template Matching**: Correlate signal with jump templates
2. **Correlation Thresholding**: Identify regions with high correlation
3. **Jump Extraction**: Convert correlation peaks to `Jump` objects

**Key Functions**:
- `detect_correlation_jumps()`: Main detection function
- `detect_correlation_jumps_with_params()`: Convenience wrapper
- `process_all_participants_correlation()`: Batch processing

---

## Universal Functions

### 1. Precision Loss Calculation (`jump_detection/analysis/loss.py`)

**Function**: `compute_precision_loss(detected_jumps, ground_truth_markers)`

**Purpose**: Evaluates detection accuracy by comparing detected jumps against ground truth annotations.

**Input**:
- `detected_jumps`: List of `Jump` objects from any algorithm
- `ground_truth_markers`: List of frame indices marking true jump locations

**Logic**:
- **True Positive**: A detected jump contains at least one ground truth marker within `[start, end]`
- **False Positive**: A detected jump contains no ground truth markers
- **False Negative**: A ground truth marker is not contained by any detected jump

**Output Dictionary**:
```python
{
    "loss": int,              # Total loss = false_positives + false_negatives
    "false_positives": int,    # Detected jumps without markers
    "false_negatives": int,    # Markers without detected jumps
    "true_positives": int      # Detected jumps with at least one marker
}
```

**Usage Pattern**:
```python
from jump_detection.analysis.loss import compute_precision_loss
from jump_detection.annotations import load_annotations

# Load annotations
annotations = load_annotations(data_file_path)
ground_truth_markers = annotations.markers

# Run detection
result = detect_threshold_jumps(data_file_path, params=params)

# Compute loss
metrics = compute_precision_loss(result.jumps, ground_truth_markers)
loss = metrics["loss"]
```

**Note**: This function is **algorithm-agnostic** - it works with jumps from any of the three algorithms.

---

### 2. Precise Boundary Detection (`jump_detection/analysis/precise.py`)

**Function**: `calculate_precise_jump_boundaries(signal, jump_center, search_window=70)`

**Purpose**: Refines jump boundaries by finding peak values before and after the jump center, then locating half-peak crossings.

**Input**:
- `signal`: 1D numpy array (typically pooled/averaged sensor data)
- `jump_center`: Center frame index of a detected jump
- `search_window`: Number of frames to search on each side (default: 70)

**Algorithm**:
1. **Find Peaks**: Locate maximum value in `[center - window, center]` (peak_before) and `[center, center + window]` (peak_after)
2. **Half-Peak Thresholds**: Compute `half_peak_before = peak_before / 2` and `half_peak_after = peak_after / 2`
3. **Precise Start**: Search backward from center to find first frame where `signal >= half_peak_before`
4. **Precise End**: Search forward from center to find first frame where `signal >= half_peak_after`

**Output Dictionary**:
```python
{
    "precise_start": int,        # Refined start frame
    "precise_end": int,           # Refined end frame
    "precise_center": int,        # Center of refined boundaries
    "precise_duration": int,      # Duration of refined jump
    "peak_before": float,         # Peak value before center
    "peak_after": float,          # Peak value after center
    "peak_before_idx": int,       # Frame index of peak_before
    "peak_after_idx": int,         # Frame index of peak_after
    "half_peak_before": float,    # Half-peak threshold for start
    "half_peak_after": float,     # Half-peak threshold for end
    "original_center": int        # Original jump center
}
```

**Batch Processing**: `process_precise_jumps(detection_results, search_window=70)`

Processes multiple `DetectionResult` objects and returns a list of dictionaries containing:
- `participant_name`: Participant identifier
- `detection_result`: Original `DetectionResult`
- `precise_jumps`: List of `PreciseJump` objects with refined boundaries

**Usage Pattern**:
```python
from jump_detection.analysis.precise import calculate_precise_jump_boundaries

# Get pooled signal from detection result
signal = result.pooled_data  # or result.raw_data.sum(axis=1)

# Refine boundaries for a jump
precise_data = calculate_precise_jump_boundaries(signal, jump.center)
precise_start = precise_data["precise_start"]
precise_end = precise_data["precise_end"]
```

**Note**: This function is **algorithm-agnostic** - it works with any jump center and signal, regardless of which algorithm detected the jump.

---

## Shared Infrastructure

### Data Loading (`jump_detection/data.py`)

**Function**: `load_dataset(data_file_path) -> np.ndarray`

**Format**: Text files with lines like:
```
HH:MM:SS.mmm -> value1,value2,value3,...
```

**Returns**: 2D numpy array `(num_frames, num_sensors)`

**Class**: `JumpWindowExport`
- Exports detected jump segments to text files
- Configurable window size and output directory
- Preserves timestamp format

---

### Configuration (`jump_detection/config.py`)

Centralized constants:
- `SAMPLING_RATE`: 50 Hz
- Algorithm default parameters
- Physics constraints (min/max flight times)
- Default data file paths
- Output directory paths

---

### Type Definitions (`jump_detection/types.py`)

**Core Types**:
- `Jump`: Detected jump with `start`, `end`, `center`, `duration`, optional `time_diff`
- `PreciseJump`: Extended jump with original and precise boundaries
- `DetectionResult`: Complete output from detection pipeline
- `GroundTruthAnnotations`: Ground truth markers with metadata

**Signal Name Enums**:
- `ThresholdNames`: Signal names for threshold algorithm
- `DerivativeNames`: Signal names for derivative algorithm

---

### Annotations (`jump_detection/annotations.py`)

**Functions**:
- `load_annotations(data_file_path)`: Load ground truth from JSON
- `save_annotations(annotations, data_file_path)`: Save ground truth to JSON

**File Pattern**: `{data_file_stem}_annotations.json` stored next to data file

**Format**: JSON with `data_file_path`, `markers` (list of frame indices), `created_at`, `modified_at`

---

## Algorithm Comparison

| Feature | Threshold | Derivative | Correlation |
|---------|-----------|------------|-------------|
| **Primary Signal** | Average (sum) | Derivative of pooled | Template correlation |
| **Detection Method** | Threshold crossing + motion validation | Pairing takeoff/landing crossings | Template matching |
| **Parameters** | 2 main (threshold, derivative_threshold) | 2 main (upper, lower) + in_air | Template-based |
| **Physics Filtering** | Yes (time constraints) | Yes (time constraints) | Yes (time constraints) |
| **Additional Validation** | -- | In-air requirement | Correlation threshold |
| **Complexity** | Medium | High | Medium |

---

## Usage Examples

### Running a Single Algorithm

```python
from jump_detection.algorithms.threshold import (
    ThresholdParameters,
    detect_threshold_jumps
)

# Configure parameters
params = ThresholdParameters(
    threshold=105.0,
    derivative_threshold=1.2
)

# Detect jumps
result = detect_threshold_jumps(
    "dataset/Test0/JoshuaGao10CMJ",
    participant_name="JoshuaGao",
    params=params,
    save_windows=True
)

# Access results
print(f"Detected {result.num_jumps} jumps")
for jump in result.jumps:
    print(f"Jump: frames {jump.start}-{jump.end}, duration={jump.duration}")
```

### Computing Loss

```python
from jump_detection.analysis.loss import compute_precision_loss
from jump_detection.annotations import load_annotations

# Load ground truth
annotations = load_annotations(Path("dataset/Test0/JoshuaGao10CMJ"))
markers = annotations.markers

# Run detection
result = detect_threshold_jumps("dataset/Test0/JoshuaGao10CMJ")

# Compute precision metrics
metrics = compute_precision_loss(result.jumps, markers)
print(f"Loss: {metrics['loss']}")
print(f"True Positives: {metrics['true_positives']}")
print(f"False Positives: {metrics['false_positives']}")
print(f"False Negatives: {metrics['false_negatives']}")
```

### Refining Jump Boundaries

```python
from jump_detection.analysis.precise import calculate_precise_jump_boundaries

# Get signal
signal = result.pooled_data

# Refine each jump
for jump in result.jumps:
    precise = calculate_precise_jump_boundaries(signal, jump.center)
    print(f"Original: {jump.start}-{jump.end}")
    print(f"Precise: {precise['precise_start']}-{precise['precise_end']}")
```

---

## Structural Consistency Verification

### ✅ Algorithm Structure

All three algorithms follow the same pattern:
1. Parameters dataclass with `as_dict()` method
2. Main `detect_*_jumps()` function returning `DetectionResult`
3. Convenience `detect_*_jumps_with_params()` function
4. Batch `process_all_*_participants()` function
5. Internal `_run_*_pipeline()` function
6. Consistent signal dictionary structure
7. Same physics filtering approach (shared helper in threshold/derivative)

### ✅ Universal Functions

**Loss Calculation**:
- ✅ Works with any `list[Jump]` from any algorithm
- ✅ Uses ground truth markers (frame indices)
- ✅ Returns standardized metrics dictionary
- ✅ Algorithm-agnostic matching logic

**Precise Boundaries**:
- ✅ Works with any signal array and jump center
- ✅ Independent of detection algorithm
- ✅ Returns standardized dictionary
- ✅ Can be applied post-hoc to any detection result

### ✅ Data Flow

```
Raw Data File
    ↓
load_dataset() → np.ndarray
    ↓
Algorithm Pipeline → DetectionResult
    ↓
Universal Functions:
    - compute_precision_loss() → metrics dict
    - calculate_precise_jump_boundaries() → precise dict
```

---

## Key Design Principles

1. **Separation of Concerns**: Algorithms, analysis, and visualization are separate modules
2. **Consistent Interfaces**: All algorithms return `DetectionResult` with standardized structure
3. **Universal Analysis**: Loss and precise calculations work with any algorithm output
4. **Type Safety**: Extensive use of dataclasses and type hints
5. **Configuration Centralization**: All constants in `config.py`
6. **Extensibility**: Easy to add new algorithms following the established pattern

---

## Potential Issues and Recommendations

### ⚠️ Missing Export

**Issue**: `compute_precision_loss` is not exported in `jump_detection/analysis/__init__.py`

**Impact**: Scripts must import directly from `jump_detection.analysis.loss` instead of `jump_detection.analysis`

**Recommendation**: Add to `__init__.py`:
```python
from .loss import compute_precision_loss
```

### ✅ Structure Validation

The repository structure is **well-organized and consistent**:
- All algorithms follow identical patterns
- Universal functions are truly algorithm-agnostic
- Type system is comprehensive
- Configuration is centralized
- Data loading is unified

The architecture supports easy extension and maintenance.

