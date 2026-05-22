from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from .detection import (
    YoloConfig,
    collect_detections_from_patch_results,
    refine_detections,
    run_yolo_on_patches,
    save_yolo_txt,
)
from .io_utils import ensure_dir, find_matching_mask, list_images, read_image, save_csv, save_json
from .patching import PatchingConfig, generate_patches_from_mask
from .sam_postprocess import (
    SamPostprocessConfig,
    export_letter_instances,
    make_instance_overlay,
    split_sam_masks_into_instances,
)
from .sam_wrapper import SAMWrapper
from .types import ImageSummary
from .visualization import add_title, draw_boxes, overlay_mask, stack_horizontal


@dataclass(frozen=True)
class PipelineConfig:
    """Full dataset-construction pipeline configuration."""

    patching: PatchingConfig = PatchingConfig()
    yolo: YoloConfig = YoloConfig()
    sam_postprocess: SamPostprocessConfig = SamPostprocessConfig()
    sam_model_type: str = "vit_h"
    sam_device: str = "cuda"
    limit_images: int = -1


def _normalize_device(device: Optional[str]) -> Optional[str]:
    if device in {"0", "1", "2", "3", "4", "5", "6", "7"}:
        return f"cuda:{device}"
    return device


def process_one_image(
    image_path: Path,
    mask_path: Optional[Path],
    output_root: Path,
    yolo_model,
    sam: SAMWrapper,
    config: PipelineConfig,
) -> ImageSummary:
    """Process one bowl image and save all intermediate/final outputs."""
    image_id = image_path.stem
    image_out = ensure_dir(output_root / image_id)
    patches_dir = ensure_dir(image_out / "patches")
    vis_dir = ensure_dir(image_out / "visualizations")
    ann_dir = ensure_dir(image_out / "annotations")

    image = read_image(image_path, mode="color")
    height, width = image.shape[:2]
    mask = read_image(mask_path, mode="gray") if mask_path is not None else None

    cv2.imwrite(str(image_out / image_path.name), image)
    if mask is not None:
        cv2.imwrite(str(image_out / f"{image_id}_mask.png"), mask)

    patches, patch_meta, _ = generate_patches_from_mask(
        image=image,
        mask=mask,
        config=config.patching,
        patches_dir=patches_dir,
    )

    yolo_results = run_yolo_on_patches(yolo_model, patches, config.yolo)
    raw_detections = collect_detections_from_patch_results(
        patch_meta,
        yolo_results,
        image_width=width,
        image_height=height,
        conf_threshold=config.yolo.conf,
    )
    refined_detections = refine_detections(
        raw_detections,
        baseline_mask=mask,
        image_width=width,
        image_height=height,
        config=config.yolo,
    )

    save_yolo_txt(
        refined_detections,
        ann_dir / f"{image_id}_yolo_refined.txt",
        image_width=width,
        image_height=height,
        default_cls=config.yolo.default_cls,
    )
    save_json(raw_detections, ann_dir / f"{image_id}_yolo_raw.json")
    save_json(refined_detections, ann_dir / f"{image_id}_yolo_refined.json")

    baseline_vis = overlay_mask(image, mask)
    raw_yolo_vis = draw_boxes(image, raw_detections, color=(0, 0, 255))
    refined_yolo_vis = draw_boxes(image, refined_detections, color=(0, 255, 255))

    cv2.imwrite(str(vis_dir / "baseline_mask_overlay.png"), baseline_vis)
    cv2.imwrite(str(vis_dir / "yolo_raw.png"), raw_yolo_vis)
    cv2.imwrite(str(vis_dir / "yolo_refined.png"), refined_yolo_vis)

    instances: List[Dict] = []
    if refined_detections:
        sam.set_image(image)
        boxes = [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in refined_detections]
        sam_masks, _ = sam.segment_many_boxes(boxes, multimask_output=False)
        instances = split_sam_masks_into_instances(
            image=image,
            detections=refined_detections,
            sam_masks=sam_masks,
            config=config.sam_postprocess,
            image_id=image_id,
        )
        export_letter_instances(instances, image, image_out, image_id)

    sam_overlay = make_instance_overlay(image, instances)
    cv2.imwrite(str(vis_dir / "sam_letter_instances.png"), sam_overlay)

    stacked = stack_horizontal(
        [
            add_title(image, "Original"),
            add_title(baseline_vis, "Baseline Mask"),
            add_title(refined_yolo_vis, "YOLO Refined"),
            add_title(sam_overlay, "SAM Instances"),
        ],
        target_height=500,
    )
    cv2.imwrite(str(vis_dir / "pipeline_overview.png"), stacked)

    summary = ImageSummary(
        image_id=image_id,
        image_path=str(image_path),
        mask_path=str(mask_path) if mask_path is not None else "",
        output_dir=str(image_out),
        num_patches=len(patches),
        num_raw_detections=len(raw_detections),
        num_refined_detections=len(refined_detections),
        num_letter_instances=len(instances),
    )
    save_json(summary.to_dict(), ann_dir / f"{image_id}_summary.json")
    return summary


def run_dataset_pipeline(
    dataset_root: Path,
    output_root: Path,
    yolo_weights: Path,
    sam_checkpoint: Path,
    images_dirname: str = "raw",
    masks_dirname: str = "masks",
    config: PipelineConfig = PipelineConfig(),
) -> List[ImageSummary]:
    """Run the complete YOLO+SAM pipeline for all images in a dataset root."""
    from ultralytics import YOLO

    images_dir = dataset_root / images_dirname
    masks_dir = dataset_root / masks_dirname
    image_files = list_images(images_dir)
    if config.limit_images > 0:
        image_files = image_files[: config.limit_images]

    output_root.mkdir(parents=True, exist_ok=True)
    yolo_model = YOLO(str(yolo_weights))
    sam = SAMWrapper(
        model_type=config.sam_model_type,
        checkpoint_path=str(sam_checkpoint),
        device=_normalize_device(config.sam_device) or "cuda",
    )

    summaries: List[ImageSummary] = []
    for image_path in image_files:
        mask_path = find_matching_mask(image_path, masks_dir)
        summary = process_one_image(image_path, mask_path, output_root, yolo_model, sam, config)
        summaries.append(summary)

    rows = [s.to_dict() for s in summaries]
    save_csv(rows, output_root / "summary.csv")
    save_json(rows, output_root / "summary.json")
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Aramaic bowls YOLO+SAM dataset-construction pipeline.")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Dataset root containing raw/ and masks/ folders.")
    parser.add_argument("--output_root", type=Path, required=True, help="Output folder for processed results.")
    parser.add_argument("--images_dirname", type=str, default="raw")
    parser.add_argument("--masks_dirname", type=str, default="masks")
    parser.add_argument("--yolo_weights", type=Path, required=True, help="Path to YOLO .pt weights.")
    parser.add_argument("--sam_checkpoint", type=Path, required=True, help="Path to SAM checkpoint .pth.")
    parser.add_argument("--sam_model_type", type=str, default="vit_h")
    parser.add_argument("--sam_device", type=str, default="cuda")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min_mask_ratio", type=float, default=0.01)
    parser.add_argument("--yolo_conf", type=float, default=0.25)
    parser.add_argument("--yolo_imgsz", type=int, default=512)
    parser.add_argument("--yolo_device", type=str, default=None)
    parser.add_argument("--min_box_area", type=float, default=200.0)
    parser.add_argument("--max_box_area", type=float, default=60000.0)
    parser.add_argument("--min_mask_overlap", type=float, default=0.1)
    parser.add_argument("--nms_iou", type=float, default=0.4)
    parser.add_argument("--min_component_area", type=int, default=20)
    parser.add_argument("--limit_images", type=int, default=-1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = PipelineConfig(
        patching=PatchingConfig(
            patch_size=args.patch_size,
            stride=args.stride,
            min_mask_ratio=args.min_mask_ratio,
        ),
        yolo=YoloConfig(
            imgsz=args.yolo_imgsz,
            conf=args.yolo_conf,
            device=_normalize_device(args.yolo_device),
            min_box_area=args.min_box_area,
            max_box_area=args.max_box_area,
            min_mask_overlap=args.min_mask_overlap,
            nms_iou=args.nms_iou,
        ),
        sam_postprocess=SamPostprocessConfig(min_component_area=args.min_component_area),
        sam_model_type=args.sam_model_type,
        sam_device=args.sam_device,
        limit_images=args.limit_images,
    )
    run_dataset_pipeline(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        yolo_weights=args.yolo_weights,
        sam_checkpoint=args.sam_checkpoint,
        images_dirname=args.images_dirname,
        masks_dirname=args.masks_dirname,
        config=config,
    )


if __name__ == "__main__":
    main()
