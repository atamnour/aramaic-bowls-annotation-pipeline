from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class SamPostprocessConfig:
    """Configuration for splitting and cleaning SAM masks into letter instances."""

    min_component_area: int = 20
    max_component_area: int = 120000
    max_hole_area: int = 200
    morph_kernel_size: int = 3
    keep_largest_per_box: bool = False
    use_grayscale_refinement: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)


def ensure_uint8_binary(mask: np.ndarray) -> np.ndarray:
    """Convert a SAM mask to uint8 binary values {0, 255}."""
    if mask.dtype == bool:
        return mask.astype(np.uint8) * 255
    m = mask.astype(np.float32)
    if m.max(initial=0) <= 1.0:
        return (m > 0.5).astype(np.uint8) * 255
    return (m > 127).astype(np.uint8) * 255


def fill_small_holes(binary_mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    """Fill small holes in a binary foreground mask."""
    binary = (binary_mask > 0).astype(np.uint8)
    inverted = 1 - binary
    contours, _ = cv2.findContours(inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) <= max_hole_area:
            cv2.drawContours(binary, [contour], 0, 1, -1)
    return (binary * 255).astype(np.uint8)


def refine_with_grayscale(gray_crop: np.ndarray, mask_crop: np.ndarray, config: SamPostprocessConfig) -> np.ndarray:
    """Optionally sharpen SAM masks using local grayscale thresholding."""
    clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=config.clahe_tile_grid)
    enhanced = clahe.apply(gray_crop)
    _, thresholded = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_and(thresholded, mask_crop)


def split_sam_masks_into_instances(
    image: np.ndarray,
    detections: List[Dict],
    sam_masks: List[np.ndarray],
    config: SamPostprocessConfig,
    image_id: str,
) -> List[Dict]:
    """Convert SAM masks for detection boxes into tight letter instances."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    instances: List[Dict] = []
    instance_id = 0

    for det, masks_for_box in zip(detections, sam_masks):
        mask = masks_for_box[0] if getattr(masks_for_box, "ndim", 0) == 3 else masks_for_box
        mask = ensure_uint8_binary(mask)

        x1 = max(0, min(width - 1, int(det["x1"])))
        y1 = max(0, min(height - 1, int(det["y1"])))
        x2 = max(0, min(width - 1, int(det["x2"])))
        y2 = max(0, min(height - 1, int(det["y2"])))
        if x2 <= x1 or y2 <= y1:
            continue

        mask_crop = mask[y1:y2, x1:x2]
        if mask_crop.size == 0:
            continue

        if config.use_grayscale_refinement:
            mask_crop = refine_with_grayscale(gray[y1:y2, x1:x2], mask_crop, config)

        mask_crop = fill_small_holes(mask_crop, config.max_hole_area)
        if config.morph_kernel_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_kernel_size, config.morph_kernel_size))
            mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_crop > 0).astype(np.uint8), connectivity=8)
        component_ids = []
        for cid in range(1, num_labels):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if config.min_component_area <= area <= config.max_component_area:
                component_ids.append(cid)

        if not component_ids and num_labels > 1:
            component_ids = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]

        if config.keep_largest_per_box and len(component_ids) > 1:
            component_ids = [max(component_ids, key=lambda cid: int(stats[cid, cv2.CC_STAT_AREA]))]

        for cid in component_ids:
            local_x = int(stats[cid, cv2.CC_STAT_LEFT])
            local_y = int(stats[cid, cv2.CC_STAT_TOP])
            local_w = int(stats[cid, cv2.CC_STAT_WIDTH])
            local_h = int(stats[cid, cv2.CC_STAT_HEIGHT])
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if local_w <= 0 or local_h <= 0:
                continue

            component_mask = ((labels == cid).astype(np.uint8) * 255)[local_y : local_y + local_h, local_x : local_x + local_w]
            gx1, gy1 = x1 + local_x, y1 + local_y
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            instances.append(
                {
                    "image_id": image_id,
                    "instance_id": instance_id,
                    "bbox": [int(gx1), int(gy1), int(gx2), int(gy2)],
                    "area": area,
                    "cls": int(det.get("cls", 0)),
                    "conf": float(det.get("conf", 1.0)),
                    "mask_crop": component_mask,
                }
            )
            instance_id += 1

    return instances


def export_letter_instances(instances: List[Dict], image: np.ndarray, output_dir: Path, image_id: str) -> List[Dict]:
    """Save letter crops, masks, and JSON metadata."""
    images_dir = output_dir / "letters" / "images"
    masks_dir = output_dir / "letters" / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    height, width = image.shape[:2]
    metadata: List[Dict] = []

    for inst in instances:
        x1, y1, x2, y2 = inst["bbox"]
        x1, x2 = max(0, x1), min(width - 1, x2)
        y1, y2 = max(0, y1), min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        instance_id = int(inst["instance_id"])
        image_name = f"{image_id}_inst{instance_id:04d}.png"
        mask_name = f"{image_id}_inst{instance_id:04d}_mask.png"

        cv2.imwrite(str(images_dir / image_name), image[y1:y2, x1:x2])
        cv2.imwrite(str(masks_dir / mask_name), inst["mask_crop"])

        meta = {k: v for k, v in inst.items() if k != "mask_crop"}
        meta["image_file"] = str(Path("letters") / "images" / image_name)
        meta["mask_file"] = str(Path("letters") / "masks" / mask_name)
        metadata.append(meta)

    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)
    with (output_dir / "annotations" / f"{image_id}_instances.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def make_instance_overlay(image: np.ndarray, instances: List[Dict], color=(0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    """Create a full-image visualization of extracted SAM letter instances."""
    overlay = image.copy()
    height, width = image.shape[:2]
    full_mask = np.zeros((height, width), dtype=np.uint8)
    for inst in instances:
        x1, y1, x2, y2 = inst["bbox"]
        mask_crop = inst["mask_crop"]
        target_h, target_w = y2 - y1, x2 - x1
        if target_h <= 0 or target_w <= 0:
            continue
        if mask_crop.shape[:2] != (target_h, target_w):
            mask_crop = cv2.resize(mask_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], mask_crop)
    idx = full_mask > 0
    overlay[idx] = ((1 - alpha) * overlay[idx] + alpha * np.array(color)).astype(np.uint8)
    return overlay
