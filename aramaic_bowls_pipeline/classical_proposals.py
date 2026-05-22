from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ClassicalProposalConfig:
    """Configurable classical baseline for letter proposal generation."""

    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    gaussian_ksize: int = 5
    median_ksize: int = 5
    adaptive_block_size: int = 51
    adaptive_c: int = 5
    invert_output: bool = True
    morph_close_ksize: int = 3
    min_area: int = 300
    max_area: int = 50000
    max_aspect_ratio: float = 10.0
    min_solidity: float = 0.20
    min_mask_overlap_ratio: float = 0.0
    class_id: int = 0


def _odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _preprocess(gray: np.ndarray, config: ClassicalProposalConfig) -> np.ndarray:
    image = gray.copy()
    if config.use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_tile_grid_size, config.clahe_tile_grid_size),
        )
        image = clahe.apply(image)
    image = cv2.GaussianBlur(image, (_odd(config.gaussian_ksize), _odd(config.gaussian_ksize)), 0)
    image = cv2.medianBlur(image, _odd(config.median_ksize))
    return image


def _binarize(gray: np.ndarray, config: ClassicalProposalConfig) -> np.ndarray:
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        _odd(config.adaptive_block_size),
        config.adaptive_c,
    )
    binary = cv2.bitwise_or(otsu, adaptive)
    if config.invert_output:
        binary = 255 - binary
    if config.morph_close_ksize > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_close_ksize, config.morph_close_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def _mask_overlap(contour: np.ndarray, mask: Optional[np.ndarray], bbox: Tuple[int, int, int, int]) -> float:
    if mask is None:
        return 1.0
    x, y, w, h = bbox
    mask_crop = (mask[y : y + h, x : x + w] > 0).astype(np.uint8)
    contour_mask = np.zeros_like(mask_crop, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour - [x, y]], -1, 1, thickness=-1)
    denom = int(contour_mask.sum())
    if denom == 0:
        return 0.0
    return float((mask_crop & contour_mask).sum()) / denom


def extract_classical_proposals(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    config: ClassicalProposalConfig = ClassicalProposalConfig(),
) -> Tuple[List[Dict], np.ndarray]:
    """Return classical letter-like bounding boxes and the binary map used to extract them."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if mask is not None and mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    binary = _binarize(_preprocess(gray, config), config)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proposals: List[Dict] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < config.min_area or area > config.max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > config.max_aspect_ratio:
            continue
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < config.min_solidity:
            continue
        overlap = _mask_overlap(contour, mask, (x, y, w, h))
        if overlap < config.min_mask_overlap_ratio:
            continue
        proposals.append(
            {
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + w),
                "y2": int(y + h),
                "area": area,
                "aspect_ratio": aspect,
                "solidity": solidity,
                "mask_overlap": overlap,
                "cls": config.class_id,
                "conf": 1.0,
            }
        )
    return proposals, binary


def draw_proposals(image: np.ndarray, proposals: List[Dict]) -> np.ndarray:
    vis = image.copy()
    for prop in proposals:
        cv2.rectangle(vis, (prop["x1"], prop["y1"]), (prop["x2"], prop["y2"]), (0, 0, 255), 2)
    return vis


def save_yolo_proposals(proposals: List[Dict], out_txt: Path, image_width: int, image_height: int) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        for prop in proposals:
            w, h = prop["x2"] - prop["x1"], prop["y2"] - prop["y1"]
            cx = (prop["x1"] + w / 2.0) / image_width
            cy = (prop["y1"] + h / 2.0) / image_height
            f.write(f"{int(prop.get('cls', 0))} {cx:.6f} {cy:.6f} {w / image_width:.6f} {h / image_height:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical letter proposal baseline for Aramaic bowls.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, default=None)
    parser.add_argument("--out_vis", type=Path, default=None)
    parser.add_argument("--out_yolo", type=Path, default=None)
    parser.add_argument("--min_area", type=int, default=300)
    parser.add_argument("--max_area", type=int, default=50000)
    parser.add_argument("--min_mask_overlap", type=float, default=0.0)
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE) if args.mask else None
    config = ClassicalProposalConfig(
        min_area=args.min_area,
        max_area=args.max_area,
        min_mask_overlap_ratio=args.min_mask_overlap,
    )
    proposals, _ = extract_classical_proposals(image, mask, config)
    print(f"Extracted {len(proposals)} proposals.")
    if args.out_vis:
        args.out_vis.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out_vis), draw_proposals(image, proposals))
    if args.out_yolo:
        save_yolo_proposals(proposals, args.out_yolo, image_width=image.shape[1], image_height=image.shape[0])


if __name__ == "__main__":
    main()
