from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


class SAMWrapper:
    """Small wrapper around Meta's Segment Anything predictor."""

    def __init__(self, model_type: str, checkpoint_path: str, device: str = "cuda") -> None:
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "segment-anything is required for SAM inference. Install it from the official repository."
            ) from exc

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device)
        self.predictor = SamPredictor(sam)
        self.current_image: np.ndarray | None = None

    def set_image(self, image_bgr: np.ndarray) -> None:
        """Set the current image for SAM. Input is expected as BGR OpenCV image."""
        if image_bgr is None:
            raise ValueError("image_bgr cannot be None.")
        if image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_bgr
        self.current_image = image_rgb
        self.predictor.set_image(image_rgb)

    def segment_from_box(self, box: Sequence[float], multimask_output: bool = False):
        """Segment a single box prompt."""
        if self.current_image is None:
            raise RuntimeError("Call set_image() before segmenting.")
        box_np = np.asarray(box, dtype=np.float32)
        masks, scores, logits = self.predictor.predict(box=box_np, multimask_output=multimask_output)
        return masks, scores, logits

    def segment_many_boxes(self, boxes: Iterable[Sequence[float]], multimask_output: bool = False):
        """Segment a sequence of box prompts."""
        all_masks: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        for box in boxes:
            masks, scores, _ = self.segment_from_box(box, multimask_output=multimask_output)
            all_masks.append(masks)
            all_scores.append(scores)
        return all_masks, all_scores
