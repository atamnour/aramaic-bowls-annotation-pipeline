# Input Structure

The expected input layout is:

```text
data/example_dataset/
├── raw/
│   ├── bowl_001.jpg
│   └── bowl_002.jpg
└── masks/
    ├── bowl_001_mask.png
    └── bowl_002_mask.png
```

Mask matching is based on the image stem. For `bowl_001.jpg`, the pipeline searches for:

- `bowl_001_mask.*`
- `bowl_001_maks.*` for typo tolerance
- `bowl_001.*`

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`.
