from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Detection:
    """Axis-aligned detection box in full-image pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    conf: float = 1.0
    cls: int = 0
    patch_idx: Optional[int] = None

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageSummary:
    """Summary of one processed bowl image."""

    image_id: str
    image_path: str
    mask_path: str
    output_dir: str
    num_patches: int
    num_raw_detections: int
    num_refined_detections: int
    num_letter_instances: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetLayout:
    """Expected dataset layout."""

    root: Path
    images_dirname: str = "raw"
    masks_dirname: str = "masks"

    @property
    def images_dir(self) -> Path:
        return self.root / self.images_dirname

    @property
    def masks_dir(self) -> Path:
        return self.root / self.masks_dirname
