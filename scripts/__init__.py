"""Scripts package for jump detection analysis."""

from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory (where jump_detection package is located)."""
    # This file is in scripts/, so go up one level to get project root
    # Project root is where jump_detection/ package exists
    current = Path(__file__).resolve().parent
    project_root = current.parent
    
    # Verify jump_detection exists
    if not (project_root / "jump_detection").exists():
        # If not found, try looking for it by traversing up
        for parent in current.parents:
            if (parent / "jump_detection").exists():
                return parent
        
        # Fallback: assume we're in project root if jump_detection is importable
        return current
    
    return project_root


PROJECT_ROOT = get_project_root()
