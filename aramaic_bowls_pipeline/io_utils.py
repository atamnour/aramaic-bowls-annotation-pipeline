from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def list_images(folder: Path) -> List[Path]:
    """Return image files in a folder, sorted by filename."""
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def find_matching_mask(
    image_path: Path,
    masks_dir: Optional[Path],
    suffixes: Iterable[str] = ("_mask", "_maks", ""),
) -> Optional[Path]:
    """Find a mask file matching an image stem."""
    if masks_dir is None or not masks_dir.exists():
        return None

    stem = image_path.stem
    for suffix in suffixes:
        for ext in IMAGE_EXTENSIONS:
            candidate = masks_dir / f"{stem}{suffix}{ext}"
            if candidate.exists():
                return candidate
    return None


def read_image(path: Path, mode: str = "color") -> np.ndarray:
    """Read an image with OpenCV and raise a clear error on failure."""
    flag = cv2.IMREAD_COLOR if mode == "color" else cv2.IMREAD_GRAYSCALE
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: object, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(rows: List[Dict], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
