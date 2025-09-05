from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix


def second_nearest_neighbour_glcm_test() -> None:
    """Codes a gray level cooccurrence matrix, which show the number of times that gray
    level j occurs at a distance d and angle theta from gray level i.
    """  # noqa: D205
    image_data = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)
    glcm: np.ndarray = graycomatrix(
        image=image_data,
        distances=[2],
        angles=[0],
        levels=4,
        symmetric=True,
        normed=True,
    )

    assert glcm[0, 0, 0, 0] == 0.5
    assert glcm[1, 1, 0, 0] == 0.5
