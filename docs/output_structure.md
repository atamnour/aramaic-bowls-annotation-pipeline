# Output Structure

For each input image, the pipeline creates:

```text
outputs/processed_dataset/<image_id>/
├── <image_id>.jpg                         # copy of the input image
├── <image_id>_mask.png                    # copy of the baseline mask, if provided
├── patches/
│   ├── patch_0000.png
│   └── patch_map.png
├── annotations/
│   ├── <image_id>_yolo_raw.json
│   ├── <image_id>_yolo_refined.json
│   ├── <image_id>_yolo_refined.txt
│   ├── <image_id>_instances.json
│   └── <image_id>_summary.json
├── letters/
│   ├── images/
│   │   └── <image_id>_inst0000.png
│   └── masks/
│       └── <image_id>_inst0000_mask.png
└── visualizations/
    ├── baseline_mask_overlay.png
    ├── yolo_raw.png
    ├── yolo_refined.png
    ├── sam_letter_instances.png
    └── pipeline_overview.png
```

At the dataset level, the pipeline also creates:

```text
outputs/processed_dataset/
├── summary.csv
└── summary.json
```
