from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.feature import graycomatrix, graycoprops  # type: ignore[import]

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def read_image_to_grayscale(
    path: Path,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Read image and convert to grayscale."""
    return io.imread(path, as_gray=True)  # type: ignore[return-value]


@dataclass
class CoOccurrenceParams:
    """Parameters for the co-occurrence matrix calculation."""

    n_levels: int = 256
    angle: int = 0
    distance: int = 1


class CoOccurrenceMatrix:
    """Represents a grey level co-occurrence matrix and provides methods to compute Haralick metrics."""

    def __init__(self, data: np.ndarray[tuple[int, int], np.dtype[np.float64]]) -> None:
        self._data = data

    @property
    def data(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Obtain the gray level co-occurrence matrix data."""
        return np.copy(self._data)

    @property
    def haralick_asm(self) -> float:
        """Obtain value for the Haralick metric of Angular Second Moment (ASM)."""
        return float(
            graycoprops(self._data.reshape(*self._data.shape, 1, 1), prop="ASM")
        )

    @property
    def haralick_energy(self) -> float:
        """Obtain value for the Haralick metric of Energy."""
        return float(
            graycoprops(self._data.reshape(*self._data.shape, 1, 1), prop="energy")
        )

    @property
    def haralick_contrast(self) -> float:
        """Obtain value for the Haralick metric of Contrast."""
        return float(
            graycoprops(self._data.reshape(*self._data.shape, 1, 1), prop="contrast")
        )

    @property
    def haralick_correlation(self) -> float:
        """Obtain value for the Haralick metric of Correlation."""
        return float(
            graycoprops(self._data.reshape(*self._data.shape, 1, 1), prop="correlation")
        )

    @property
    def haralick_homogeneity(self) -> float:
        """Obtain value for the Haralick metric of Homogeneity."""
        return float(
            graycoprops(self._data.reshape(*self._data.shape, 1, 1), prop="homogeneity")
        )

    @property
    def haralick_mean(self) -> float:
        """Obtain value for the Haralick metric of Mean."""
        return float(
            graycoprops(self._data.reshape(*self._data.shape, 1, 1), prop="mean")
        )

    @property
    def haralick_std(self) -> float:
        """Obtain value for the Haralick metric of Standard Deviation."""
        return float(
            graycoprops(self._data.reshape(*self._data.shape, 1, 1), prop="std")
        )

    @classmethod
    def from_image(
        cls,
        image: np.ndarray[tuple[int, int], np.dtype[np.float64]],
        *,
        params: CoOccurrenceParams | None = None,
    ) -> CoOccurrenceMatrix:
        """Create a Co-occurrenceMatrix from a grayscale image."""
        params = params or CoOccurrenceParams()
        grayscale_image = (image * (params.n_levels - 1)).astype("uint8")
        data = graycomatrix(
            image=grayscale_image,
            distances=[params.distance],
            angles=[params.angle],
            levels=params.n_levels,
            symmetric=True,
            normed=True,
        )
        return CoOccurrenceMatrix(data[:, :, 0, 0])  # type: ignore[index]

    @classmethod
    def from_file(
        cls, path: Path, *, params: CoOccurrenceParams | None = None
    ) -> CoOccurrenceMatrix:
        """Create a Co-occurrenceMatrix from a grayscale image file."""
        image = read_image_to_grayscale(path)
        return cls.from_image(image, params=params)


def plot_co_occurrence_matrix(
    matrix: CoOccurrenceMatrix, *, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    """Produce 2D colour map of gray level co-occurrence matrix for angle of 0 rad."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = cast("Figure", ax.figure)

    im = ax.imshow(matrix.data)
    ax.set_title("Gray Level Co-occurrence Matrix")
    ax.set_xlabel("Gray level in j")
    ax.set_ylabel("Gray level in i")
    fig.colorbar(im, ax=ax, label="Normalized frequency")

    mean = matrix.haralick_mean
    std = matrix.haralick_std
    contrast = matrix.haralick_contrast

    ax.text(
        0.95,
        0.95,
        f"Mean = {mean:.3g}\n Standard Deviation = {std:.3g}\n Contrast = {contrast:.3g}",
        fontsize=10,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.75},
    )
    ax.plot(  # Plots line connecting points one std away from the mean along the diagonal.
        [mean - std, mean + std],
        [mean - std, mean + std],
        color="black",
        linewidth=1,
    )

    ax.plot(  # Plots line connecting points one contrast away from the mean perpendicular to the diagonal.
        [mean - np.sqrt(contrast), mean + np.sqrt(contrast)],
        [mean + np.sqrt(contrast), mean - np.sqrt(contrast)],
        color="black",
        linewidth=1,
    )
    return fig, ax
