# Jump Detection Research

This repository contains code and data for jump detection research using sensor data analysis.

## Project Structure

```
ThesisResearch/
├── jump_detection/          # Core library package
│   ├── algorithms/          # Jump detection algorithms (threshold, derivative, correlation)
│   ├── analysis/            # Analysis utilities (loss computation, precise calculations)
│   ├── plotting/            # Plotting utilities
│   ├── data.py              # Data loading utilities
│   ├── config.py            # Configuration constants
│   └── types.py             # Type definitions
│
├── scripts/                 # Executable analysis scripts
│   ├── detection/           # Legacy wrapper scripts for backwards compatibility
│   ├── visualization/       # Plotting and visualization scripts
│   ├── analysis/            # Data analysis and machine learning scripts
│   └── tools/               # Utility tools (annotation interface, etc.)
│
├── dataset/                 # Input data files
│   ├── Test0/               # Test dataset 0
│   ├── Test1(100Hz)/        # Test dataset 1 (100Hz sampling)
│   ├── Test2/               # Test dataset 2
│   └── ...
│
├── results/                 # Generated outputs (gitignored)
│   ├── plots/               # All generated plots
│   │   ├── grid_search/     # Grid search results
│   │   ├── loss/            # Loss landscape plots
│   │   ├── detailed/        # Detailed plots
│   │   └── summary/         # Summary plots
│   ├── extracted_jumps/     # Extracted jump segments
│   └── analysis/            # Analysis outputs (PCA, SVM, etc.)
│
├── experimentation/         # Experimental scripts
└── writeup/                 # LaTeX thesis document
```

## Installation

Install required Python packages:

```bash
pip install numpy matplotlib scikit-learn pillow
```

## Usage

### Running Scripts

All scripts are organized in the `scripts/` directory. Run them from the project root:

```bash
# Visualization scripts
python scripts/visualization/summary_plot.py
python scripts/analysis/loss_threshold.py    # Threshold algorithm grid search
python scripts/analysis/loss_derivative.py    # Derivative algorithm grid search
python scripts/analysis/loss_correlation.py    # Correlation algorithm grid search

# Analysis scripts
python scripts/advanced_analysis/PCA_jumps.py
python scripts/advanced_analysis/SVMclassification.py

# Tools
python scripts/tools/annotate_jumps.py
```

### Detection Algorithms

The core library provides three jump detection algorithms:

1. **Threshold Algorithm** - Based on pressure threshold
2. **Derivative Algorithm** - Based on derivative of pressure signal
3. **Correlation Algorithm** - Based on template correlation matching

See `jump_detection/algorithms/` for implementation details.

### Data Format

Input data files should be formatted as:
```
HH:MM:SS.mmm -> value1,value2,value3,...
```

Where each line contains a timestamp and comma-separated sensor values.

## Outputs

All generated outputs (plots, extracted jumps, analysis results) are saved to the `results/` directory, which is excluded from version control.

## Development

The project uses Python type hints and follows PEP 8 style guidelines. Core functionality is in the `jump_detection/` package, while analysis scripts are in `scripts/`.

