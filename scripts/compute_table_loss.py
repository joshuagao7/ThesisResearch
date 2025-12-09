"""Compute loss values for each participant to update Table 1 in the writeup.

This script computes precision loss (FP + FN) for each participant using
the derivative algorithm with optimal parameters and precise jump boundaries.
"""

from pathlib import Path
import sys

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import numpy as np
from jump_detection.utils import find_all_data_files
from jump_detection.annotations import load_annotations
from jump_detection.algorithms.derivative import detect_derivative_jumps, DerivativeParameters
from jump_detection.analysis.loss import compute_precision_loss
from jump_detection.analysis.precise import calculate_precise_jump_boundaries
from jump_detection.config import (
    SAMPLING_RATE,
    DERIVATIVE_UPPER_DEFAULT,
    DERIVATIVE_LOWER_DEFAULT,
    IN_AIR_THRESHOLD_DEFAULT,
    DEFAULT_DATA_FILES,
)
from jump_detection.types import Jump

SEARCH_WINDOW = 70


def compute_precise_jumps_from_result(result):
    """Calculate precise jump boundaries from detection result."""
    signal = result.pooled_data if result.pooled_data is not None else result.raw_data.sum(axis=1)
    precise_jumps = []
    
    for jump in result.jumps:
        # Calculate precise boundaries
        precise_data = calculate_precise_jump_boundaries(
            signal, jump.center, SEARCH_WINDOW
        )
        # Only include if precise boundaries are valid
        if precise_data["precise_start"] < precise_data["precise_end"]:
            precise_jumps.append(
                Jump(
                    start=precise_data["precise_start"],
                    end=precise_data["precise_end"],
                    center=precise_data["precise_center"],
                    duration=precise_data["precise_duration"],
                )
            )
    
    return precise_jumps


def get_participant_name_from_path(data_file: Path) -> str:
    """Extract participant name from data file path."""
    name = data_file.stem
    # Map file names to table names
    name_mapping = {
        # Test0 files
        "DanBraun10sequential": "Dan Braun",
        "Darwin10CMJ": "Darwin",
        "Joey10CMJ": "Joey",
        "JoshuaGao10CMJ": "Joshua Gao",
        "JoshuaGaonatural_interaction_hopper": "Joshua Gao (natural)",
        "JoshuaKerner10CMJ": "Joshua Kerner",
        "MatthewRiley10CMJ": "Matthew Riley",
        # Test2 files
        "PWG Subject 1": "PWG Subject 1",
        "pwg subject 2": "PWG Subject 2",
        "PWG Subject 3": "PWG Subject 3",
        "PWG subject 5 190lbs-taller": "PWG Subject 5 (190 lbs)",
        "PWG Subject 6 160 lbs": "PWG Subject 6 (160 lbs)",
        "PWG subject 7 220 lbs": "PWG Subject 7 (220 lbs)",
        "PWG subject 8 175": "PWG Subject 8 (175 lbs)",
        "PWG Subject 9 170": "PWG Subject 9 (170 lbs)",
        # Test3 (video) files - note: some use curly apostrophe (\u2019)
        "2scaleup-Matthew": "Matthew (2scaleup)",
        "Dante-205-5\u20197-pwgcourt": "Dante (205 lbs, 5'7'')",  # curly apostrophe
        "DormRoom-Tommy-200lbs-6ft": "Tommy (200 lbs, 6ft)",
        "izayah-149-5\u20197-pwgcourt": "Izayah (149 lbs, 5'7'')",  # curly apostrophe
        "lofty-70kg-5\u20197-pwgcourt": "Lofty (70 kg, 5'7'')",  # curly apostrophe
        "Matthew-scaleup-1-10": "Matthew (scaleup)",
        "Mouse-subject1- 150lb": "Mouse Subject 1 (150 lbs)",
        "Mouse-Subject2-145lbs": "Mouse Subject 2 (145 lbs)",
        "Mouse-subject3-145": "Mouse Subject 3 (145 lbs)",
        "Mouse-subject4-5\u20195-135": "Mouse Subject 4 (5'5'', 135 lbs)",  # curly apostrophe
        "Mouse-subject5-5\u20196~160": "Mouse Subject 5 (5'6'', 160 lbs)",  # curly apostrophe
        "mouse-subject6": "Mouse Subject 6",
        "Noah-130-dorm": "Noah (130 lbs)",
    }
    return name_mapping.get(name, name)


def main():
    print("Computing loss values for each participant...")
    print(f"Using optimal parameters: upper={DERIVATIVE_UPPER_DEFAULT}, lower={DERIVATIVE_LOWER_DEFAULT}")
    
    # Use optimal parameters
    params = DerivativeParameters(
        upper_threshold=DERIVATIVE_UPPER_DEFAULT,
        lower_threshold=DERIVATIVE_LOWER_DEFAULT,
        in_air_threshold=IN_AIR_THRESHOLD_DEFAULT,
    )
    
    results = []
    
    # Process each data file
    for data_file in DEFAULT_DATA_FILES:
        data_path = Path(data_file)
        if not data_path.exists():
            print(f"Warning: File not found: {data_path}")
            continue
            
        participant_name = get_participant_name_from_path(data_path)
        print(f"\nProcessing: {participant_name}")
        
        # Load annotations
        annotations = load_annotations(data_path)
        if annotations is None or not annotations.markers:
            print(f"  Warning: No annotations found for {participant_name}")
            continue
        
        ground_truth_markers = annotations.markers
        expected_jumps = len(ground_truth_markers)
        
        # Run derivative algorithm
        result = detect_derivative_jumps(
            data_path,
            participant_name=participant_name,
            params=params,
        )
        
        detected_jumps = len(result.jumps)
        
        # Calculate precise jumps
        precise_jumps = compute_precise_jumps_from_result(result)
        
        # Compute loss (FP + FN)
        metrics = compute_precision_loss(precise_jumps, ground_truth_markers)
        loss = metrics["loss"]
        fp = metrics["false_positives"]
        fn = metrics["false_negatives"]
        tp = metrics["true_positives"]
        
        results.append({
            "participant": participant_name,
            "expected": expected_jumps,
            "detected": detected_jumps,
            "loss": loss,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })
        
        print(f"  Expected: {expected_jumps}, Detected: {detected_jumps}, Loss: {loss} (FP={fp}, FN={fn}, TP={tp})")
    
    # Sort results to match table order
    table_order = [
        # Test0 participants
        "Dan Braun",
        "Darwin",
        "Joey",
        "Joshua Gao",
        "Joshua Gao (natural)",
        "Joshua Kerner",
        "Matthew Riley",
        # Test2 participants
        "PWG Subject 1",
        "PWG Subject 2",
        "PWG Subject 3",
        "PWG Subject 5 (190 lbs)",
        "PWG Subject 6 (160 lbs)",
        "PWG Subject 7 (220 lbs)",
        "PWG Subject 8 (175 lbs)",
        "PWG Subject 9 (170 lbs)",
        # Test3 (video) participants
        "Matthew (2scaleup)",
        "Dante (205 lbs, 5'7'')",
        "Tommy (200 lbs, 6ft)",
        "Izayah (149 lbs, 5'7'')",
        "Lofty (70 kg, 5'7'')",
        "Matthew (scaleup)",
        "Mouse Subject 1 (150 lbs)",
        "Mouse Subject 2 (145 lbs)",
        "Mouse Subject 3 (145 lbs)",
        "Mouse Subject 4 (5'5'', 135 lbs)",
        "Mouse Subject 5 (5'6'', 160 lbs)",
        "Mouse Subject 6",
        "Noah (130 lbs)",
    ]
    
    # Sort by table order, then add any remaining participants not in table_order
    sorted_results = []
    found_names = set()
    for name in table_order:
        for r in results:
            if r["participant"] == name:
                sorted_results.append(r)
                found_names.add(name)
                break
    
    # Add any participants that weren't in table_order
    for r in results:
        if r["participant"] not in found_names:
            sorted_results.append(r)
    
    # Print table
    print("\n" + "="*80)
    print("TABLE DATA FOR LATEX")
    print("="*80)
    print("\nParticipant & Expected Jumps & Detected Jumps & Loss (FP+FN) \\\\")
    print("\\midrule")
    
    total_expected = 0
    total_detected = 0
    total_loss = 0
    
    for r in sorted_results:
        total_expected += r["expected"]
        total_detected += r["detected"]
        total_loss += r["loss"]
        # Escape special characters in participant names
        name = r["participant"].replace("&", "\\&")
        print(f"{name} & {r['expected']} & {r['detected']} & {r['loss']} \\\\")
    
    print("\\midrule")
    print(f"\\textbf{{Total}} & \\textbf{{{total_expected}}} & \\textbf{{{total_detected}}} & \\textbf{{{total_loss}}} \\\\")
    
    # Calculate accuracy based on loss
    # Accuracy = (Total Expected - Total Loss) / Total Expected
    # This accounts for both FPs and FNs
    total_correct = total_expected - total_loss
    accuracy = (total_correct / total_expected) * 100 if total_expected > 0 else 0
    
    print("\n" + "="*80)
    print("ACCURACY CALCULATION")
    print("="*80)
    print(f"Total Expected: {total_expected}")
    print(f"Total Loss (FP+FN): {total_loss}")
    print(f"Total Correct: {total_expected - total_loss}")
    print(f"Accuracy: {accuracy:.1f}%")
    print("="*80)


if __name__ == "__main__":
    main()

