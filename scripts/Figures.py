"""Generate presentation figures for jump detection visualization."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json

import matplotlib.pyplot as plt
import numpy as np

from jump_detection.algorithms.derivative import (
    DerivativeParameters,
    detect_derivative_jumps,
)
from jump_detection.config import DATASET_ROOT
from jump_detection.data import load_dataset
from jump_detection.utils import PRIMARY_BLUE, PRIMARY_RED, PRIMARY_ORANGE

# Set up matplotlib style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def load_annotations(data_path: Path) -> list[int]:
    """Load ground truth jump markers from annotations JSON file."""
    annotations_path = data_path.parent / f"{data_path.name}_annotations.json"
    
    if not annotations_path.exists():
        print(f"Warning: Annotations file not found: {annotations_path}")
        return []
    
    try:
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        markers = data.get('markers', [])
        print(f"Loaded {len(markers)} ground truth markers")
        return markers
    except Exception as e:
        print(f"Warning: Failed to load annotations: {e}")
        return []


def check_file_accessible(file_path: Path) -> bool:
    """Check if a file is accessible and can be read.
    
    Returns True if file is accessible, False otherwise.
    """
    if not file_path.exists():
        return False
    
    try:
        # Try to read a small portion to verify accessibility
        with open(file_path, 'rb') as f:
            f.read(1024)  # Read first 1KB
        return True
    except (OSError, IOError, TimeoutError):
        return False


def generate_presentation_figures():
    """Generate all 4 presentation figures from JoshuaGao10CMJ dataset."""
    # Load data
    data_path = DATASET_ROOT / "Test0" / "JoshuaGao10CMJ"
    data_path_str = str(data_path.resolve())
    
    # Check if file exists
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path_str}\n"
            "Please ensure the file exists and is accessible."
        )
    
    # Check file size
    file_size = data_path.stat().st_size
    print(f"File size: {file_size} bytes")
    if file_size == 0:
        raise ValueError(
            f"ERROR: Data file is empty (0 bytes): {data_path_str}\n"
            "The file exists but contains no data. Please check if the file needs to be downloaded from cloud storage."
        )
    
    # Check if file is accessible (may be on cloud storage that needs to download)
    print("Checking file accessibility...")
    if not check_file_accessible(data_path):
        error_msg = (
            f"\n{'='*70}\n"
            f"ERROR: Cannot access data file: {data_path_str}\n"
            f"{'='*70}\n\n"
            "This file appears to be on cloud storage (iCloud Drive, Dropbox, etc.)\n"
            "and may not be fully downloaded.\n\n"
            "SOLUTIONS:\n"
            "  1. Open Finder and navigate to: dataset/Test0/\n"
            "  2. Right-click 'JoshuaGao10CMJ' and select 'Download Now' (iCloud)\n"
            "     OR double-click the file to trigger download\n"
            "  3. Wait for the download to complete (check iCloud status)\n"
            "  4. Run this script again\n\n"
            "ALTERNATIVE:\n"
            "  Copy the file to a local directory (not in cloud storage)\n"
            "  and update the path in this script.\n"
            f"{'='*70}\n"
        )
        raise FileNotFoundError(error_msg)
    
    print(f"Loading data from: {data_path_str}")
    try:
        raw_data = load_dataset(data_path)
    except TimeoutError as e:
        error_msg = (
            f"\nERROR: Timeout while loading data file: {data_path_str}\n"
            f"Error: {e}\n\n"
            "This may indicate:\n"
            "  1. The file is on a network mount that is slow or unavailable\n"
            "  2. The file is on cloud storage and needs to be downloaded\n"
            "  3. File system permissions issue\n"
            "  4. The file is locked by another process\n\n"
            "Solutions:\n"
            "  - Ensure the file is fully downloaded from cloud storage\n"
            "  - Copy the file to a local directory\n"
            "  - Check file permissions\n"
        )
        raise TimeoutError(error_msg) from e
    except Exception as e:
        error_msg = (
            f"\nERROR: Failed to load data file: {data_path_str}\n"
            f"Error: {type(e).__name__}: {e}\n"
        )
        raise RuntimeError(error_msg) from e
    
    print(f"Successfully loaded data: shape {raw_data.shape}")
    
    # Validate data
    if raw_data.size == 0:
        raise ValueError(
            f"ERROR: Loaded data is empty!\n"
            f"The file '{data_path_str}' appears to be empty or could not be read properly.\n"
            f"Please check:\n"
            f"  1. The file is not empty\n"
            f"  2. The file format is correct\n"
            f"  3. The file is fully downloaded from cloud storage"
        )
    
    if len(raw_data.shape) != 2:
        raise ValueError(
            f"ERROR: Expected 2D array (n_samples, n_sensors), but got shape {raw_data.shape}\n"
            f"The data should have shape (n_samples, 48) for 48 sensors."
        )
    
    if raw_data.shape[1] != 48:
        print(f"WARNING: Expected 48 sensors, but got {raw_data.shape[1]} sensors")
    
    print(f"Data validation passed: {raw_data.shape[0]} samples, {raw_data.shape[1]} sensors")
    
    # Load annotations
    ground_truth_markers = load_annotations(data_path)
    
    # Compute pooled signal and derivative
    pooled = raw_data.sum(axis=1)
    derivative = np.gradient(pooled)
    
    # Run actual derivative detection algorithm
    params = DerivativeParameters(
        upper_threshold=20,
        lower_threshold=-15,
        in_air_threshold=190,
        min_flight_time=0.2,
        max_flight_time=1.2,
    )
    detection_result = detect_derivative_jumps(data_path, params=params)
    detected_jumps = detection_result.jumps
    
    # Create output directory
    output_dir = Path("results/plots/presentation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    plot_raw_sensors(raw_data, output_dir)
    plot_raw_with_pooled(raw_data, pooled, output_dir)
    plot_raw_with_annotations(raw_data, pooled, ground_truth_markers, output_dir)
    plot_threshold_naive(raw_data, pooled, output_dir)
    plot_derivative_overlay(raw_data, derivative, detected_jumps, output_dir)
    plot_derivative_thresholds(derivative, detected_jumps, ground_truth_markers, output_dir)
    
    print(f"All figures saved to {output_dir}")


def plot_raw_sensors(raw_data: np.ndarray, output_dir: Path):
    """Plot 1: Raw 48 sensor data."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Use a colormap to assign different colors to each sensor
    # Using tab20 colormap which has 20 distinct colors, cycling through for 48 sensors
    cmap = plt.cm.get_cmap('tab20')
    num_sensors = raw_data.shape[1]
    
    # Plot all 48 sensor channels with different colors
    for channel in range(num_sensors):
        # Cycle through colors: use modulo to repeat colors if needed
        color = cmap(channel % 20)
        ax.plot(raw_data[:, channel], linewidth=0.5, alpha=0.7, color=color)
    
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Sensor Value", fontsize=12)
    ax.set_title("Raw 48-Sensor Data", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "raw_sensors.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'raw_sensors.png'}")


def plot_raw_with_pooled(raw_data: np.ndarray, pooled: np.ndarray, output_dir: Path):
    """Plot: Raw data with pooled/average signal (no ground truth markers)."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot all 48 sensor channels
    for channel in range(raw_data.shape[1]):
        ax.plot(raw_data[:, channel], linewidth=0.5, alpha=0.5, color='gray')
    
    # Plot pooled/average signal
    ax.plot(pooled, linewidth=2, color=PRIMARY_BLUE, alpha=0.9, 
            label="Average Signal (Pooled)")
    
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Sensor Value", fontsize=12)
    ax.set_title("Raw Data with Average Signal", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "raw_with_pooled.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'raw_with_pooled.png'}")


def plot_raw_with_annotations(raw_data: np.ndarray, pooled: np.ndarray, 
                               ground_truth_markers: list[int], output_dir: Path):
    """Plot 2: Raw data with ground truth jump markers and average signal."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot all 48 sensor channels
    for channel in range(raw_data.shape[1]):
        ax.plot(raw_data[:, channel], linewidth=0.5, alpha=0.5, color='gray')
    
    # Plot pooled/average signal
    ax.plot(pooled, linewidth=2, color=PRIMARY_BLUE, alpha=0.9, 
            label="Average Signal (Pooled)")
    
    # Plot ground truth markers
    if ground_truth_markers:
        for i, marker in enumerate(ground_truth_markers):
            if 0 <= marker < len(pooled):
                ax.axvline(marker, color=PRIMARY_RED, linewidth=2, 
                          alpha=0.7, linestyle="--",
                          label="Ground Truth Jump" if i == 0 else "")
                # Add a marker point on the pooled signal
                ax.plot(marker, pooled[marker], "*", color=PRIMARY_RED, 
                       markersize=15, markeredgewidth=1.5, markeredgecolor='white',
                       zorder=10)
    
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Sensor Value", fontsize=12)
    ax.set_title("Raw Data with Ground Truth Jump Markers", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "raw_with_annotations.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'raw_with_annotations.png'}")


def plot_threshold_naive(raw_data: np.ndarray, pooled: np.ndarray, output_dir: Path):
    """Plot 2: Raw data with threshold at 90, highlighting naive detection regions."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot all 48 sensor channels
    for channel in range(raw_data.shape[1]):
        ax.plot(raw_data[:, channel], linewidth=0.5, alpha=0.7, color='gray')
    
    # Plot pooled/average signal
    ax.plot(pooled, linewidth=2, color=PRIMARY_BLUE, alpha=0.9, 
            label="Average Signal (Pooled)")
    
    # Add threshold line at 90
    threshold = 90
    ax.axhline(y=threshold, color=PRIMARY_RED, linestyle="--", linewidth=2, 
               alpha=0.8, label=f"Threshold ({threshold})")
    
    # Highlight regions where pooled signal < 90 (naive threshold detection)
    below_threshold = pooled < threshold
    # Find contiguous regions
    regions = []
    in_region = False
    start_idx = None
    
    for i in range(len(below_threshold)):
        if below_threshold[i] and not in_region:
            start_idx = i
            in_region = True
        elif not below_threshold[i] and in_region:
            regions.append((start_idx, i - 1))
            in_region = False
    
    # Close any open region at the end
    if in_region:
        regions.append((start_idx, len(below_threshold) - 1))
    
    # Highlight regions
    for start, end in regions:
        ax.axvspan(start, end, alpha=0.2, color=PRIMARY_RED, 
                   label="Naive Threshold Detection" if start == regions[0][0] else "")
    
    # Count number of detected jumps
    num_jumps = len(regions)
    
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Sensor Value", fontsize=12)
    ax.set_title("Raw Data with Naive Threshold (90)", fontsize=14, fontweight='bold')
    # Add number of detected jumps in top right
    ax.text(0.98, 0.98, f"Detected Jumps: {num_jumps}", 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_naive.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'threshold_naive.png'}")


def plot_derivative_overlay(raw_data: np.ndarray, derivative: np.ndarray, 
                            detected_jumps: list, output_dir: Path):
    """Plot 3: Raw data with derivative values overlayed."""
    fig, ax1 = plt.subplots(figsize=(16, 6))
    
    # Plot raw sensors on primary axis
    for channel in range(raw_data.shape[1]):
        ax1.plot(raw_data[:, channel], linewidth=0.5, alpha=0.5, color='gray')
    
    ax1.set_xlabel("Frame", fontsize=12)
    ax1.set_ylabel("Sensor Value", fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Plot derivative on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(derivative, linewidth=2, color=PRIMARY_ORANGE, alpha=0.8, label="Derivative")
    ax2.set_ylabel("Derivative Value", fontsize=12, color=PRIMARY_ORANGE)
    ax2.tick_params(axis='y', labelcolor=PRIMARY_ORANGE)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10)
    
    # Count number of detected jumps
    num_jumps = len(detected_jumps)
    
    ax1.set_title("Raw Data with Derivative Overlay", fontsize=14, fontweight='bold')
    # Add number of detected jumps in top right
    ax1.text(0.98, 0.98, f"Detected Jumps: {num_jumps}", 
            transform=ax1.transAxes, fontsize=12, fontweight='bold',
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "derivative_overlay.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'derivative_overlay.png'}")


def plot_derivative_thresholds(derivative: np.ndarray, detected_jumps: list, 
                                ground_truth_markers: list[int], output_dir: Path):
    """Plot 4: Derivative with thresholds at +20 and -15, highlighting detected jumps."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot derivative signal
    ax.plot(derivative, linewidth=1.5, color=PRIMARY_ORANGE, label="Derivative")
    
    # Add threshold lines
    upper_threshold = 20
    lower_threshold = -15
    ax.axhline(y=upper_threshold, color=PRIMARY_RED, linestyle="--", 
               linewidth=2, alpha=0.8, label=f"Upper Threshold (+{upper_threshold})")
    ax.axhline(y=lower_threshold, color=PRIMARY_RED, linestyle="--", 
               linewidth=2, alpha=0.8, label=f"Lower Threshold ({lower_threshold})")
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Highlight detected jumps
    for jump in detected_jumps:
        if jump.start < jump.end:
            ax.axvspan(jump.start, jump.end, alpha=0.2, color=PRIMARY_BLUE,
                      label="Detected Jump" if jump == detected_jumps[0] else "")
            # Mark takeoff (lower threshold crossing) and landing (upper threshold crossing)
            ax.axvline(jump.start, color=PRIMARY_BLUE, linewidth=2, 
                      linestyle='--', alpha=0.7)
            ax.axvline(jump.end, color=PRIMARY_BLUE, linewidth=2, 
                      linestyle='--', alpha=0.7)
    
    # Plot ground truth markers
    if ground_truth_markers:
        for i, marker in enumerate(ground_truth_markers):
            if 0 <= marker < len(derivative):
                ax.axvline(marker, color=PRIMARY_RED, linewidth=2, 
                          alpha=0.7, linestyle=":",
                          label="Ground Truth Jump" if i == 0 else "")
                # Add a marker point on the derivative signal
                ax.plot(marker, derivative[marker], "*", color=PRIMARY_RED, 
                       markersize=15, markeredgewidth=1.5, markeredgecolor='white',
                       zorder=10)
    
    # Count number of detected jumps
    num_jumps = len(detected_jumps)
    
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Derivative Value", fontsize=12)
    ax.set_title("Derivative with Thresholds (+20 / -15)", fontsize=14, fontweight='bold')
    # Add number of detected jumps in top right
    ax.text(0.98, 0.98, f"Detected Jumps: {num_jumps}", 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-100, 100)  # Match the style from pipeline plots
    
    plt.tight_layout()
    plt.savefig(output_dir / "derivative_thresholds.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'derivative_thresholds.png'}")


if __name__ == "__main__":
    generate_presentation_figures()

