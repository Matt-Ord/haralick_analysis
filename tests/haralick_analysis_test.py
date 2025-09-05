from __future__ import annotations

import numpy as np

from haralick_analysis import (
    CoOccurrenceMatrix,
    CoOccurrenceParams,
)


def test_analysis_distance_2() -> None:
    image_data = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float64)
    params = CoOccurrenceParams(n_levels=2, angle=0, distance=2)
    glcm = CoOccurrenceMatrix.from_image(image_data, params=params)

    assert glcm.data[0, 0] == 0.5  # noqa: PLR2004
    assert glcm.data[1, 1] == 0.5  # noqa: PLR2004
