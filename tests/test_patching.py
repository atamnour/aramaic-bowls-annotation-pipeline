import numpy as np

from aramaic_bowls_pipeline.patching import PatchingConfig, generate_patches_from_mask


def test_generate_patches_from_mask_basic():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=np.uint8) * 255
    config = PatchingConfig(patch_size=32, stride=32, min_mask_ratio=0.1, save_patches=False, save_patch_map=False)
    patches, meta, patch_map = generate_patches_from_mask(image, mask, config)
    assert len(patches) == 4
    assert len(meta) == 4
    assert patch_map is None
