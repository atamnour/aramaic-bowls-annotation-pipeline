#!/usr/bin/env bash
set -euo pipefail

python -m aramaic_bowls_pipeline.pipeline \
  --dataset_root data/example_dataset \
  --images_dirname raw \
  --masks_dirname masks \
  --output_root outputs/processed_dataset \
  --yolo_weights checkpoints/yolo_letters.pt \
  --sam_checkpoint checkpoints/sam_vit_h.pth \
  --sam_model_type vit_h \
  --sam_device cuda \
  --patch_size 512 \
  --stride 256 \
  --min_mask_ratio 0.01 \
  --yolo_conf 0.25 \
  --min_box_area 200 \
  --max_box_area 60000 \
  --min_mask_overlap 0.1 \
  --nms_iou 0.4 \
  --min_component_area 20
