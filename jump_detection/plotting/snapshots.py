"""Reusable jump snapshot plotting helpers."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..analysis.precise import calculate_precise_jump_boundaries
from ..types import DetectionResult
from ..utils import PRIMARY_BLUE, PRIMARY_ORANGE, PRIMARY_YELLOW, SEARCH_WINDOW

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def plot_jump_snapshots(
    result: DetectionResult,
    *,
    signal: Optional[np.ndarray] = None,
    window_frames: int = 150,
    search_window: int = SEARCH_WINDOW,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    if signal is None:
        signal = result.pooled_data if result.pooled_data is not None else result.raw_data.sum(axis=1)
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.suptitle(title or f"Jump Snapshots - {result.participant_name or 'Participant'} (window={window_frames * 2} frames)", fontsize=16)
    
    colors = (PRIMARY_BLUE, PRIMARY_ORANGE)
    for idx, jump in enumerate(result.iter_jumps()):
        precise_center = calculate_precise_jump_boundaries(signal, jump.center, search_window)["precise_center"]
        start = max(0, precise_center - window_frames)
        end = min(len(signal), precise_center + window_frames)
        color = colors[idx % len(colors)] if idx < 5 else PRIMARY_YELLOW
        ax.plot(signal[start:end], linewidth=2, alpha=0.7, color=color)
    
    ax.set_xlabel("Frames (relative to jump center)")
    ax.set_ylabel("Signal Value")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path_obj, dpi=300, bbox_inches='tight')
        print(f"Saved jump snapshots to {save_path_obj}")
    
    if show:
        plt.show()
    return fig


__all__ = ["plot_jump_snapshots"]

