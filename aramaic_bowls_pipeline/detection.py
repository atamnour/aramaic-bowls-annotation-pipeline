from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class YoloConfig:
    """YOLO inference and post-processing configuration."""

    imgsz: int = 512
    conf: float = 0.25
    device: Optional[str] = None
    default_cls: int = 0
    min_box_area: float = 200.0
    max_box_area: float = 60000.0
    min_mask_overlap: float = 0.1
    nms_iou: float = 0.4


def run_yolo_on_patches(model, patches: List[np.ndarray], config: YoloConfig):
    """Run an Ultralytics YOLO model on a list of image patches."""
    if not patches:
        return []
    return model.predict(
        patches,
        imgsz=config.imgsz,
        conf=config.conf,
        device=config.device,
        verbose=False,
    )


def collect_detections_from_patch_results(
    patch_metadata: List[Dict],
    yolo_results,
    image_width: int,
    image_height: int,
    conf_threshold: float,
) -> List[Dict]:
    """Map YOLO detections from patch coordinates back to full-image coordinates."""
    detections: List[Dict] = []

    for meta, result in zip(patch_metadata, yolo_results):
        if result is None or result.boxes is None or len(result.boxes) == 0:
            continue

        x0, y0, pw, ph = meta["x"], meta["y"], meta["w"], meta["h"]
        xywhn = result.boxes.xywhn.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy()

        for (xc_n, yc_n, bw_n, bh_n), conf, cls_id in zip(xywhn, confs, clses):
            if float(conf) < conf_threshold:
                continue

            xc, yc = xc_n * pw, yc_n * ph
            bw, bh = bw_n * pw, bh_n * ph

            x1 = int(x0 + xc - bw / 2)
            y1 = int(y0 + yc - bh / 2)
            x2 = int(x0 + xc + bw / 2)
            y2 = int(y0 + yc + bh / 2)

            x1 = max(0, min(image_width - 1, x1))
            y1 = max(0, min(image_height - 1, y1))
            x2 = max(0, min(image_width - 1, x2))
            y2 = max(0, min(image_height - 1, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": float(conf),
                    "cls": int(cls_id),
                    "patch_idx": int(meta.get("idx", -1)),
                }
            )

    return detections


def box_area(det: Dict) -> float:
    return float(max(0, det["x2"] - det["x1"]) * max(0, det["y2"] - det["y1"]))


def box_iou(a: Dict, b: Dict) -> float:
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def greedy_nms(detections: List[Dict], iou_threshold: float) -> List[Dict]:
    """Simple confidence-sorted non-maximum suppression."""
    remaining = sorted(detections, key=lambda d: d.get("conf", 0.0), reverse=True)
    kept: List[Dict] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [d for d in remaining if box_iou(best, d) < iou_threshold]
    return kept


def refine_detections(
    detections: List[Dict],
    baseline_mask: Optional[np.ndarray],
    image_width: int,
    image_height: int,
    config: YoloConfig,
) -> List[Dict]:
    """Filter detections by area, baseline-mask overlap, and IoU suppression."""
    if baseline_mask is not None:
        if baseline_mask.ndim == 3:
            baseline_mask = cv2.cvtColor(baseline_mask, cv2.COLOR_BGR2GRAY)
        mask_bin = (baseline_mask > 0).astype(np.uint8)
    else:
        mask_bin = None

    filtered: List[Dict] = []
    for det in detections:
        area = box_area(det)
        if area < config.min_box_area or area > config.max_box_area:
            continue

        if mask_bin is not None:
            x1 = max(0, min(image_width - 1, int(det["x1"])))
            y1 = max(0, min(image_height - 1, int(det["y1"])))
            x2 = max(0, min(image_width - 1, int(det["x2"])))
            y2 = max(0, min(image_height - 1, int(det["y2"])))
            if x2 <= x1 or y2 <= y1:
                continue
            overlap = float(mask_bin[y1:y2, x1:x2].mean())
            if overlap < config.min_mask_overlap:
                continue
            det = dict(det)
            det["mask_overlap"] = overlap

        filtered.append(det)

    return greedy_nms(filtered, config.nms_iou)


def save_yolo_txt(
    detections: List[Dict],
    txt_path: Path,
    image_width: int,
    image_height: int,
    default_cls: int = 0,
    include_confidence: bool = True,
) -> None:
    """Save detections in YOLO normalized format."""
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as f:
        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            cx = (x1 + w / 2.0) / image_width
            cy = (y1 + h / 2.0) / image_height
            wn = w / image_width
            hn = h / image_height
            cls_id = int(det.get("cls", default_cls))
            if include_confidence:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f} {float(det.get('conf', 1.0)):.4f}\n")
            else:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")
