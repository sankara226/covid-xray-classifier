"""Image-region masking and source-confounding summaries."""

from __future__ import annotations

import numpy as np


def mask_region(image: np.ndarray, box: tuple[int, int, int, int], *, fill: float = 0) -> np.ndarray:
    r0, r1, c0, c1 = box
    if not (0 <= r0 < r1 <= image.shape[0] and 0 <= c0 < c1 <= image.shape[1]):
        raise ValueError("Mask box is outside image bounds")
    result = np.array(image, copy=True)
    result[r0:r1, c0:c1, ...] = fill
    return result


def mask_variants(
    image: np.ndarray, box: tuple[int, int, int, int], corner: tuple[int, int, int, int]
) -> dict[str, np.ndarray]:
    center_only = np.zeros_like(image)
    r0, r1, c0, c1 = box
    center_only[r0:r1, c0:c1, ...] = image[r0:r1, c0:c1, ...]
    return {
        "full": np.array(image, copy=True),
        "lungs_hidden": mask_region(image, box),
        "lungs_only": center_only,
        "corner_hidden": mask_region(image, corner),
    }
