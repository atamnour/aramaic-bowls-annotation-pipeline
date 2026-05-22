from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PatchingConfig:
    """Sliding-window patch extraction configuration."""

    patch_size: int = 512
    stride: int = 256
    min_mask_ratio: float = 0.01
    resize_mask_to_image: bool = True
    save_patches: bool = True
    save_patch_map: bool = True
    patch_prefix: str = "patch"


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _prepare_mask(image_shape: Tuple[int, int], mask: Optional[np.ndarray], resize: bool) -> np.ndarray:
    h, w = image_shape
    if mask is None:
        return np.ones((h, w), dtype=np.uint8)

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if resize and mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return (mask > 0).astype(np.uint8)


def generate_patches_from_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    config: PatchingConfig,
    patches_dir: Optional[Path] = None,
) -> Tuple[List[np.ndarray], List[Dict], Optional[np.ndarray]]:
    """Generate square image patches and metadata from an image/mask pair."""
    img = _as_bgr(image)
    h, w = img.shape[:2]
    mask_bin = _prepare_mask((h, w), mask, resize=config.resize_mask_to_image)

    if config.patch_size <= 0 or config.stride <= 0:
        raise ValueError("patch_size and stride must be positive integers.")
    if not (0.0 <= config.min_mask_ratio <= 1.0):
        raise ValueError("min_mask_ratio must be in [0, 1].")

    patches: List[np.ndarray] = []
    metadata: List[Dict] = []
    patch_map: Optional[np.ndarray] = img.copy() if config.save_patch_map else None

    if patches_dir is not None and config.save_patches:
        patches_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    ps = config.patch_size
    for y in range(0, max(0, h - ps + 1), config.stride):
        for x in range(0, max(0, w - ps + 1), config.stride):
            mask_patch = mask_bin[y : y + ps, x : x + ps]
            mask_ratio = float(mask_patch.mean())
            if mask_ratio < config.min_mask_ratio:
                continue

            patch = img[y : y + ps, x : x + ps]
            patches.append(patch)
            metadata.append(
                {
                    "idx": idx,
                    "x": int(x),
                    "y": int(y),
                    "w": int(ps),
                    "h": int(ps),
                    "mask_ratio": mask_ratio,
                }
            )

            if patches_dir is not None and config.save_patches:
                cv2.imwrite(str(patches_dir / f"{config.patch_prefix}_{idx:04d}.png"), patch)

            if patch_map is not None:
                cv2.rectangle(patch_map, (x, y), (x + ps, y + ps), (255, 0, 0), 2)
                cv2.putText(
                    patch_map,
                    str(idx),
                    (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            idx += 1

    if patches_dir is not None and patch_map is not None:
        patches_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(patches_dir / "patch_map.png"), patch_map)

    return patches, metadata, patch_map
