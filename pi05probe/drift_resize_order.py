"""Color-preserving composition controls using the production PIL resize.

P is defined at 224. Lifting it to 256 requires interpolation; pattern224
measures that interpolation separately. No photometric transform is applied.
"""
import numpy as np
from PIL import Image
from openpi_client import image_tools


def resize_rgb(image, size):
    return image_tools.resize_with_pad(image, size, size)


def resize_mask(mask, size, nearest=False):
    method = Image.NEAREST if nearest else Image.BILINEAR
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (size, size), resample=method)).astype(np.float32) / 255


def masks_from_pair(clean256, probe256, threshold=12):
    diff256 = np.abs(probe256.astype(np.int16) - clean256.astype(np.int16)).max(-1)
    clean224, probe224 = resize_rgb(clean256, 224), resize_rgb(probe256, 224)
    diff224 = np.abs(probe224.astype(np.int16) - clean224.astype(np.int16)).max(-1)
    return diff256 > threshold, diff224 > threshold


def compose_variants(clean256, mask256, mask224, patch224):
    clean224 = resize_rgb(clean256, 224)
    patch256 = resize_rgb(patch224, 256)
    pattern224 = resize_rgb(patch256, 224)
    matched256 = resize_mask(mask224, 256, nearest=True) > 0.5
    native_pre = resize_rgb(np.where(mask256[..., None], patch256, clean256), 224)
    matched_pre = resize_rgb(np.where(matched256[..., None], patch256, clean256), 224)
    alpha224 = resize_mask(mask256, 224)[..., None]
    boundary224 = np.rint(clean224 * (1 - alpha224) + patch224 * alpha224).clip(0, 255).astype(np.uint8)
    return {
        'post224': np.where(mask224[..., None], patch224, clean224),
        'pattern224': np.where(mask224[..., None], pattern224, clean224),
        'boundary224': boundary224,
        'pre256_matched': matched_pre,
        'pre256_native': native_pre,
    }
