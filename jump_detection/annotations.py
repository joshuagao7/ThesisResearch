"""Utilities for loading and saving ground truth annotations."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .types import GroundTruthAnnotations


def get_annotation_path(data_file_path: Path) -> Path:
    """Get annotation file path: {data_file_stem}_annotations.json."""
    return data_file_path.parent / f"{data_file_path.stem}_annotations.json"


def load_annotations(data_file_path: Path) -> Optional[GroundTruthAnnotations]:
    """Load ground truth annotations for a data file."""
    annotation_path = get_annotation_path(data_file_path)
    
    if not annotation_path.exists():
        return None
    
    try:
        with annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Ensure data_file_path in annotation matches (allow for relative paths)
        annotations = GroundTruthAnnotations.from_dict(data)
        return annotations
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Warning: Failed to load annotations from {annotation_path}: {e}")
        return None


def save_annotations(
    annotations: GroundTruthAnnotations,
    data_file_path: Path,
) -> Path:
    """Save ground truth annotations to JSON file."""
    annotation_path = get_annotation_path(data_file_path)
    
    # Update timestamps
    if not annotation_path.exists():
        annotations.created_at = datetime.now().isoformat()
    annotations.modified_at = datetime.now().isoformat()
    
    # Ensure data_file_path is relative or absolute as needed
    annotations.data_file_path = str(data_file_path)
    
    # Save to JSON
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    with annotation_path.open("w", encoding="utf-8") as f:
        json.dump(annotations.to_dict(), f, indent=2)
    
    return annotation_path

