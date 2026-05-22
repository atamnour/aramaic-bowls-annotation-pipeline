from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def overlay_mask(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a binary mask on an image."""
    output = image.copy()
    if mask is None:
        return output
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    idx = mask > 0
    output[idx] = ((1 - alpha) * output[idx] + alpha * np.array(color)).astype(np.uint8)
    return output


def draw_boxes(
    image: np.ndarray,
    detections: List[Dict],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    draw_confidence: bool = True,
) -> np.ndarray:
    """Draw detections on a copy of the image."""
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        if draw_confidence:
            label = f"{int(det.get('cls', 0))}:{float(det.get('conf', 1.0)):.2f}"
            cv2.putText(vis, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return vis


def stack_horizontal(images: List[np.ndarray], target_height: Optional[int] = None) -> np.ndarray:
    """Resize images to a common height and concatenate them horizontally."""
    if not images:
        raise ValueError("No images to stack.")
    if target_height is None:
        target_height = images[0].shape[0]

    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_height / float(h)
        resized.append(cv2.resize(img, (int(w * scale), target_height), interpolation=cv2.INTER_AREA))
    return np.concatenate(resized, axis=1)


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    """Add a small black title banner to an image."""
    out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
    cv2.rectangle(out, (8, 8), (18 + text_w, 20 + text_h), (0, 0, 0), cv2.FILLED)
    cv2.putText(out, title, (13, 16 + text_h), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out
