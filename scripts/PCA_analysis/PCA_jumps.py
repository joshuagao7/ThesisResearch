"""PCA analysis of jump segments extracted from all participants."""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import List, Tuple
from mpl_toolkits.mplot3d import Axes3D

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jump_detection.algorithms.derivative import process_all_derivative_participants
from jump_detection.analysis.precise import calculate_precise_jump_boundaries
from jump_detection.config import DEFAULT_DATA_FILES


def extract_jump_segments(
    window_size: int = 150,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract jump segments from all participants.
    
    Each data file is treated as a unique person. All jumps extracted from
    a given file are labeled with that file's identifier (file name).
    This enables person identification: classifying which person (file) a jump belongs to.
    
    Args:
        window_size: Number of frames to extract per jump (default 150)
        
    Returns:
        Tuple of (jump_matrix, participant_names) where:
        - jump_matrix: numpy array of shape (num_jumps, window_size)
        - participant_names: list of participant names (file identifiers), one per jump
    """
    # Filter data files to exclude Dan Braun (who uses a different jump technique)
    # DEFAULT_DATA_FILES already excludes Dan Braun (he uses 10sequential, not 10CMJ),
    # but we'll be explicit and filter out any files from "Dan Braun" directory
    data_files = [
        f for f in DEFAULT_DATA_FILES
        if Path(f).parent.name != "Dan Braun"
    ]
    
    print(f"Loading data from {len(data_files)} participants (excluding Dan Braun)")
    
    # Process all participants using derivative algorithm
    # Explicitly pass filtered data_files to avoid loading Dan Braun's data
    results = process_all_derivative_participants(
        data_files=data_files,
        save_windows=False
    )
    
    jump_segments: List[np.ndarray] = []
    participant_names: List[str] = []
    
    half_window = window_size // 2  # 75 frames before and after
    
    for result in results:
        # Get the signal (pooled_data if available, else sum of raw_data)
        signal = result.pooled_data if result.pooled_data is not None else result.raw_data.sum(axis=1)
        # Use file path to create unique participant identifier (each file = one person)
        # Extract from metadata if available, otherwise use participant_name
        data_file_path = result.metadata.get("data_file_path", "")
        if data_file_path:
            path = Path(data_file_path)
            # Use file name as unique identifier - each file = one person
            # Handle both files (with extension) and directories (without extension)
            if path.suffix:
                participant = path.stem  # Remove extension for files
            else:
                participant = path.name  # Use full name for directories
        else:
            participant = result.participant_name or "Participant"
        
        # Process each jump
        for jump in result.iter_jumps():
            # Calculate precise jump boundaries
            precise = calculate_precise_jump_boundaries(signal, jump.center)
            precise_center = precise["precise_center"]
            
            # Extract window around precise center
            ideal_start = precise_center - half_window
            ideal_end = precise_center + half_window
            
            start_frame = max(0, ideal_start)
            end_frame = min(len(signal), ideal_end)
            
            # Extract the segment
            segment = signal[start_frame:end_frame]
            
            # Handle edge cases: pad if necessary
            # We want precise_center to be at position half_window in the final segment
            # In the extracted segment, precise_center is at index (precise_center - start_frame)
            # So we need to pad: half_window - (precise_center - start_frame) frames before
            pad_before = max(0, half_window - (precise_center - start_frame))
            # Similarly for after: we want (end_frame - precise_center) frames after precise_center
            # We need: half_window - (end_frame - precise_center) frames after
            pad_after = max(0, half_window - (end_frame - precise_center))
            
            # Pad with edge values
            if pad_before > 0:
                segment = np.concatenate([np.full(pad_before, segment[0]), segment])
            if pad_after > 0:
                segment = np.concatenate([segment, np.full(pad_after, segment[-1])])
            
            # Ensure exactly window_size frames (should be after padding, but double-check)
            if len(segment) != window_size:
                # This shouldn't happen, but if it does, trim or pad to exact size
                if len(segment) > window_size:
                    # Trim from center
                    center_idx = len(segment) // 2
                    segment = segment[center_idx - half_window:center_idx + half_window]
                else:
                    # Pad more if needed (shouldn't happen)
                    while len(segment) < window_size:
                        if pad_before > 0:
                            segment = np.concatenate([[segment[0]], segment])
                        else:
                            segment = np.concatenate([segment, [segment[-1]]])
            
            jump_segments.append(segment)
            participant_names.append(participant)
    
    # Convert to numpy array
    jump_matrix = np.array(jump_segments)
    
    print(f"\n=== Jump Extraction Summary ===")
    print(f"Total jumps extracted: {len(jump_segments)}")
    print(f"Number of participants: {len(results)}")
    print(f"Final matrix shape: {jump_matrix.shape}")
    print(f"  - Rows (jumps): {jump_matrix.shape[0]}")
    print(f"  - Columns (time frames per jump): {jump_matrix.shape[1]}")
    
    # Print per-participant statistics
    participant_counts = Counter(participant_names)
    print(f"\nJumps per participant:")
    for participant, count in sorted(participant_counts.items()):
        print(f"  - {participant}: {count} jumps")
    print(f"================================\n")
    
    return jump_matrix, participant_names


def perform_pca(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Principal Component Analysis on the data.
    
    Args:
        data: Input data matrix of shape (n_samples, n_features)
        
    Returns:
        Tuple of (principal_components, eigenvalues, explained_variance)
        - principal_components: eigenvectors (columns are PCs)
        - eigenvalues: eigenvalues
        - explained_variance: percentage of variance explained by each PC
    """
    print(f"\n=== PCA Diagnostic Information ===")
    print(f"Input data matrix shape: {data.shape}")
    print(f"  - Number of samples (jumps): {data.shape[0]}")
    print(f"  - Number of features (time frames): {data.shape[1]}")
    
    # Mean-center the data
    mean = np.mean(data, axis=0)
    data_centered = data - mean
    print(f"\nMean-centered data matrix shape: {data_centered.shape}")
    print(f"  - Mean vector shape: {mean.shape}")
    
    # Compute covariance matrix
    cov_matrix = np.cov(data_centered.T)
    print(f"\nCovariance matrix shape: {cov_matrix.shape}")
    print(f"  - Should be ({data.shape[1]}, {data.shape[1]}) for features x features")
    
    # Diagonalize (eigenvalue decomposition)
    # Use eigh for symmetric matrices (guarantees real eigenvalues)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    print(f"\nEigenvalue decomposition:")
    print(f"  - Eigenvalues shape: {eigenvalues.shape}")
    print(f"  - Eigenvectors shape: {eigenvectors.shape}")
    print(f"  - Number of principal components: {len(eigenvalues)}")
    
    # Ensure eigenvalues and eigenvectors are real (should be, but be safe)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Calculate explained variance
    total_variance = np.sum(eigenvalues)
    explained_variance = (eigenvalues / total_variance) * 100
    explained_variance = np.real(explained_variance)  # Ensure real
    print(f"\nExplained variance shape: {explained_variance.shape}")
    
    # Project data onto principal components
    principal_components = data_centered @ eigenvectors
    principal_components = np.real(principal_components)  # Ensure real
    print(f"\nPrincipal components (projected data) shape: {principal_components.shape}")
    print(f"  - Should be ({data.shape[0]}, {data.shape[1]}) for samples x PCs")
    print(f"  - Each row is a jump, each column is a PC")
    print(f"===================================\n")
    
    return principal_components, eigenvalues, explained_variance


def plot_pca_results(
    principal_components: np.ndarray,
    explained_variance: np.ndarray,
    participant_names: List[str],
) -> None:
    """
    Plot PCA results for PC1-PC4.
    
    Args:
        principal_components: Projected data onto PCs
        explained_variance: Percentage of variance explained by each PC
        participant_names: List of participant names for coloring
    """
    # Get unique participants for color mapping
    # Use colors matching grid search plots: blue, orange, red, purple, yellow
    unique_participants = sorted(set(participant_names))
    num_participants = len(unique_participants)
    
    # Color scheme matching grid search plots aesthetic
    # Blue (#2166ac), Orange (#fdae61), Red (#d73027), Purple, Yellow
    color_palette = [
        "#2166ac",  # Blue (from grid search)
        "#fdae61",  # Orange (from grid search)
        "#d73027",  # Red (from grid search highlights)
        "#762a83",  # Purple
        "#f1c40f",  # Yellow
    ]
    
    # Extend palette if more than 5 participants
    if num_participants > len(color_palette):
        # Add more colors if needed
        color_palette.extend([
            "#2ecc71",  # Green
            "#e74c3c",  # Red variant
            "#3498db",  # Light blue
            "#9b59b6",  # Purple variant
            "#f39c12",  # Orange variant
        ])
    
    participant_to_color = {
        p: color_palette[i % len(color_palette)] 
        for i, p in enumerate(unique_participants)
    }
    
    # Extract PC1-PC4 and ensure they're real
    pc1 = np.real(principal_components[:, 0])
    pc2 = np.real(principal_components[:, 1])
    pc3 = np.real(principal_components[:, 2])
    pc4 = np.real(principal_components[:, 3])
    
    # Create figure with subplots for all PC pairs
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Principal Component Analysis of Jump Segments", fontsize=16)
    
    # PC1 vs PC2
    ax = axes[0, 0]
    for participant in unique_participants:
        mask = np.array(participant_names) == participant
        ax.scatter(
            pc1[mask],
            pc2[mask],
            c=[participant_to_color[participant]],
            label=participant,
            alpha=0.7,
            s=50,
        )
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.2f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.2f}% variance)", fontsize=12)
    ax.set_title("PC1 vs PC2", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # PC1 vs PC3
    ax = axes[0, 1]
    for participant in unique_participants:
        mask = np.array(participant_names) == participant
        ax.scatter(
            pc1[mask],
            pc3[mask],
            c=[participant_to_color[participant]],
            label=participant,
            alpha=0.7,
            s=50,
        )
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.2f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC3 ({explained_variance[2]:.2f}% variance)", fontsize=12)
    ax.set_title("PC1 vs PC3", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # PC1 vs PC4
    ax = axes[0, 2]
    for participant in unique_participants:
        mask = np.array(participant_names) == participant
        ax.scatter(
            pc1[mask],
            pc4[mask],
            c=[participant_to_color[participant]],
            label=participant,
            alpha=0.7,
            s=50,
        )
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.2f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC4 ({explained_variance[3]:.2f}% variance)", fontsize=12)
    ax.set_title("PC1 vs PC4", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # PC2 vs PC3
    ax = axes[1, 0]
    for participant in unique_participants:
        mask = np.array(participant_names) == participant
        ax.scatter(
            pc2[mask],
            pc3[mask],
            c=[participant_to_color[participant]],
            label=participant,
            alpha=0.7,
            s=50,
        )
    ax.set_xlabel(f"PC2 ({explained_variance[1]:.2f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC3 ({explained_variance[2]:.2f}% variance)", fontsize=12)
    ax.set_title("PC2 vs PC3", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # PC2 vs PC4
    ax = axes[1, 1]
    for participant in unique_participants:
        mask = np.array(participant_names) == participant
        ax.scatter(
            pc2[mask],
            pc4[mask],
            c=[participant_to_color[participant]],
            label=participant,
            alpha=0.7,
            s=50,
        )
    ax.set_xlabel(f"PC2 ({explained_variance[1]:.2f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC4 ({explained_variance[3]:.2f}% variance)", fontsize=12)
    ax.set_title("PC2 vs PC4", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # PC3 vs PC4
    ax = axes[1, 2]
    for participant in unique_participants:
        mask = np.array(participant_names) == participant
        ax.scatter(
            pc3[mask],
            pc4[mask],
            c=[participant_to_color[participant]],
            label=participant,
            alpha=0.7,
            s=50,
        )
    ax.set_xlabel(f"PC3 ({explained_variance[2]:.2f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC4 ({explained_variance[3]:.2f}% variance)", fontsize=12)
    ax.set_title("PC3 vs PC4", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    project_root = Path(__file__).parent.parent.parent
    save_path = project_root / "results" / "plots" / "pca_results.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved PCA results plot to {save_path}")
    
    plt.show()
    
    # Print explained variance summary
    print("\nExplained Variance by Principal Component:")
    for i in range(min(10, len(explained_variance))):
        print(f"PC{i+1}: {np.real(explained_variance[i]):.2f}%")


def plot_average_jump_profile(
    jump_matrix: np.ndarray,
    participant_names: List[str],
    window_size: int = 150,
) -> None:
    """
    Plot the average jump profile across all people, along with individual profiles.
    
    Args:
        jump_matrix: Matrix of jump segments (num_jumps, window_size)
        participant_names: List of participant names (file identifiers), one per jump
        window_size: Number of frames per jump
    """
    print("\n=== Computing Average Jump Profile ===")
    
    # Compute average across all jumps
    mean_profile = np.mean(jump_matrix, axis=0)
    std_profile = np.std(jump_matrix, axis=0)
    
    # Compute per-person averages
    unique_participants = sorted(set(participant_names))
    participant_profiles = {}
    for participant in unique_participants:
        mask = np.array(participant_names) == participant
        participant_profiles[participant] = np.mean(jump_matrix[mask], axis=0)
    
    print(f"Total jumps: {jump_matrix.shape[0]}")
    print(f"Number of people: {len(unique_participants)}")
    print(f"Average jumps per person: {jump_matrix.shape[0] / len(unique_participants):.1f}")
    
    # Create time axis (frames, centered at 0)
    time_frames = np.arange(window_size) - window_size // 2
    
    # Color scheme matching other plots
    color_palette = [
        "#2166ac",  # Blue
        "#fdae61",  # Orange
        "#d73027",  # Red
        "#762a83",  # Purple
        "#f1c40f",  # Yellow
        "#2ecc71",  # Green
        "#e74c3c",  # Red variant
        "#3498db",  # Light blue
        "#9b59b6",  # Purple variant
        "#f39c12",  # Orange variant
    ]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Average Jump Profile Across All People", fontsize=16)
    
    # Plot 1: Overall average with standard deviation
    ax = axes[0]
    ax.plot(time_frames, mean_profile, linewidth=3, color="#2166ac", label="Average (All People)", zorder=3)
    ax.fill_between(
        time_frames,
        mean_profile - std_profile,
        mean_profile + std_profile,
        alpha=0.3,
        color="#2166ac",
        label="±1 Std Dev",
    )
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Jump Center")
    ax.set_xlabel("Frames (relative to jump center)", fontsize=12)
    ax.set_ylabel("Signal Value", fontsize=12)
    ax.set_title("Average Jump Profile Across All People (with Standard Deviation)", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Average per person + overall average
    ax = axes[1]
    # Plot individual person averages
    for i, participant in enumerate(unique_participants):
        color = color_palette[i % len(color_palette)]
        ax.plot(
            time_frames,
            participant_profiles[participant],
            linewidth=1.5,
            alpha=0.6,
            color=color,
            label=participant,
        )
    
    # Plot overall average on top
    ax.plot(time_frames, mean_profile, linewidth=3, color="black", linestyle="--", label="Overall Average", zorder=10)
    ax.axvline(x=0, color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Frames (relative to jump center)", fontsize=12)
    ax.set_ylabel("Signal Value", fontsize=12)
    ax.set_title("Average Jump Profile Per Person (with Overall Average)", fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nAverage Profile Statistics:")
    print(f"  Mean value: {np.mean(mean_profile):.2f}")
    print(f"  Min value: {np.min(mean_profile):.2f} (at frame {np.argmin(mean_profile) - window_size // 2})")
    print(f"  Max value: {np.max(mean_profile):.2f} (at frame {np.argmax(mean_profile) - window_size // 2})")
    print(f"  Std dev: {np.mean(std_profile):.2f} (average across frames)")
    print("=" * 50)


def plot_pca_3d(
    principal_components: np.ndarray,
    explained_variance: np.ndarray,
    participant_names: List[str],
) -> None:
    """
    Plot PCA results in 3D to better visualize separation between participants.
    
    Args:
        principal_components: Projected data onto PCs
        explained_variance: Percentage of variance explained by each PC
        participant_names: List of participant names for coloring
    """
    # Get unique participants for color mapping
    # Use colors matching grid search plots: blue, orange, red, purple, yellow
    unique_participants = sorted(set(participant_names))
    num_participants = len(unique_participants)
    
    # Color scheme matching grid search plots aesthetic
    # Blue (#2166ac), Orange (#fdae61), Red (#d73027), Purple, Yellow
    color_palette = [
        "#2166ac",  # Blue (from grid search)
        "#fdae61",  # Orange (from grid search)
        "#d73027",  # Red (from grid search highlights)
        "#762a83",  # Purple
        "#f1c40f",  # Yellow
    ]
    
    # Extend palette if more than 5 participants
    if num_participants > len(color_palette):
        # Add more colors if needed
        color_palette.extend([
            "#2ecc71",  # Green
            "#e74c3c",  # Red variant
            "#3498db",  # Light blue
            "#9b59b6",  # Purple variant
            "#f39c12",  # Orange variant
        ])
    
    participant_to_color = {
        p: color_palette[i % len(color_palette)] 
        for i, p in enumerate(unique_participants)
    }
    
    # Extract PC1-PC4 and ensure they're real
    pc1 = np.real(principal_components[:, 0])
    pc2 = np.real(principal_components[:, 1])
    pc3 = np.real(principal_components[:, 2])
    pc4 = np.real(principal_components[:, 3])
    
    # Create figure with multiple 3D subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("3D Principal Component Analysis - Interactive (Rotate to View)", fontsize=16)
    
    # Plot 1: PC1, PC2, PC3 (most variance)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    for participant in unique_participants:
        mask = np.array(participant_names) == participant
        ax1.scatter(
            pc1[mask],
            pc2[mask],
            pc3[mask],
            c=[participant_to_color[participant]],
            label=participant,
            alpha=0.7,
            s=50,
        )
    ax1.set_xlabel(f"PC1 ({explained_variance[0]:.2f}%)", fontsize=10)
    ax1.set_ylabel(f"PC2 ({explained_variance[1]:.2f}%)", fontsize=10)
    ax1.set_zlabel(f"PC3 ({explained_variance[2]:.2f}%)", fontsize=10)
    ax1.set_title("PC1 vs PC2 vs PC3", fontsize=12)
    ax1.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0, 1))
    
    # Plot 2: PC1, PC2, PC4
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    for participant in unique_participants:
        mask = np.array(participant_names) == participant
        ax2.scatter(
            pc1[mask],
            pc2[mask],
            pc4[mask],
            c=[participant_to_color[participant]],
            label=participant,
            alpha=0.7,
            s=50,
        )
    ax2.set_xlabel(f"PC1 ({explained_variance[0]:.2f}%)", fontsize=10)
    ax2.set_ylabel(f"PC2 ({explained_variance[1]:.2f}%)", fontsize=10)
    ax2.set_zlabel(f"PC4 ({explained_variance[3]:.2f}%)", fontsize=10)
    ax2.set_title("PC1 vs PC2 vs PC4", fontsize=12)
    ax2.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0, 1))
    
    # Plot 3: PC1, PC3, PC4
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    for participant in unique_participants:
        mask = np.array(participant_names) == participant
        ax3.scatter(
            pc1[mask],
            pc3[mask],
            pc4[mask],
            c=[participant_to_color[participant]],
            label=participant,
            alpha=0.7,
            s=50,
        )
    ax3.set_xlabel(f"PC1 ({explained_variance[0]:.2f}%)", fontsize=10)
    ax3.set_ylabel(f"PC3 ({explained_variance[2]:.2f}%)", fontsize=10)
    ax3.set_zlabel(f"PC4 ({explained_variance[3]:.2f}%)", fontsize=10)
    ax3.set_title("PC1 vs PC3 vs PC4", fontsize=12)
    ax3.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0, 1))
    
    # Plot 4: PC2, PC3, PC4
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    for participant in unique_participants:
        mask = np.array(participant_names) == participant
        ax4.scatter(
            pc2[mask],
            pc3[mask],
            pc4[mask],
            c=[participant_to_color[participant]],
            label=participant,
            alpha=0.7,
            s=50,
        )
    ax4.set_xlabel(f"PC2 ({explained_variance[1]:.2f}%)", fontsize=10)
    ax4.set_ylabel(f"PC3 ({explained_variance[2]:.2f}%)", fontsize=10)
    ax4.set_zlabel(f"PC4 ({explained_variance[3]:.2f}%)", fontsize=10)
    ax4.set_title("PC2 vs PC3 vs PC4", fontsize=12)
    ax4.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0, 1))
    
    plt.tight_layout()
    plt.show()
    
    print("\n3D plots displayed. Rotate the plots interactively to see separation between participants.")


def main() -> None:
    """Main function to run PCA analysis on jump segments."""
    print("Extracting jump segments from all participants...")
    jump_matrix, participant_names = extract_jump_segments(window_size=150)
    
    print("\nPlotting average jump profile across all people...")
    plot_average_jump_profile(jump_matrix, participant_names, window_size=150)
    
    print("\nPerforming PCA...")
    principal_components, eigenvalues, explained_variance = perform_pca(jump_matrix)
    
    print("\nPlotting 2D PCA results...")
    plot_pca_results(principal_components, explained_variance, participant_names)
    
    print("\nPlotting 3D PCA results...")
    plot_pca_3d(principal_components, explained_variance, participant_names)
    
    print("\nPCA analysis complete!")
    print("\nTo run classification analysis, use: python SVMclassification.py")


if __name__ == "__main__":
    main()

