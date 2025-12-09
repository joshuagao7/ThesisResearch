from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image


def _load_images(paths: Iterable[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        if path.exists():
            images.append(Image.open(path).convert("RGB"))
    return images


def _grid_dimensions(count: int) -> tuple[int, int]:
    if count == 0:
        return 0, 0
    rows = int(count**0.5)
    cols = rows
    # Expand grid until it fits all images
    while rows * cols < count:
        # Try adding a column first
        if (rows * (cols + 1)) >= count:
            cols += 1
            break
        # If that's not enough, add both a row and column
        cols += 1
        rows += 1
    return rows, cols


def _stitch(images: list[Image.Image], output_path: Path) -> None:
    rows, cols = _grid_dimensions(len(images))
    if rows == 0:
        print("No images to stitch.")
        return

    width, height = images[0].size
    canvas = Image.new("RGB", (cols * width, rows * height), color="white")

    for index, img in enumerate(images):
        row = index // cols
        col = index % cols
        canvas.paste(img, (col * width, row * height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved stitched image to {output_path}")


def stitch_directory(source_dir: Path, output_path: Path) -> None:
    # Get all PNG files but exclude the output file itself
    all_pngs = sorted(source_dir.glob("*.png"))
    # Exclude the output file by comparing resolved paths
    output_name = output_path.name
    image_paths = [p for p in all_pngs if p.name != output_name]
    
    print(f"Found {len(image_paths)} PNG files in {source_dir}")
    if len(all_pngs) != len(image_paths):
        print(f"Excluding output file: {output_name}")
    
    images = _load_images(image_paths)
    if not images:
        print(f"No PNG files found in {source_dir} (excluding output file)")
        return
    
    print(f"Stitching {len(images)} images...")
    _stitch(images, output_path)


def main() -> None:
    # Get project root (where this script is located: scripts/visualization/)
    project_root = Path(__file__).resolve().parent.parent.parent
    stitch_directory(
        project_root / "results/plots/grid_search/threshold",
        project_root / "results/plots/grid_search/threshold/threshold_combined.png",
    )
    stitch_directory(
        project_root / "results/plots/grid_search/derivative",
        project_root / "results/plots/grid_search/derivative/derivative_combined.png",
    )


if __name__ == "__main__":
    main()

