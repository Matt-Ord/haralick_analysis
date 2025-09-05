from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.feature import graycomatrix, graycoprops

if TYPE_CHECKING:
    from pathlib import Path


def read_image_to_grayscale(
    path: Path,
) -> np.ndarray[Any, Any]:
    """Read image and convert to grayscale."""
    return io.imread(
        path, as_gray=True
    )  # To read images, use io.imread; as_gray is parameter within io.imread which converts colour images into grayscale.


def get_gray_level_cooccurrence_matrix(
    grayscale_image_array: np.ndarray, *, n_levels: int = 256
) -> np.ndarray[Any, Any]:
    """Codes a gray level cooccurrence matrix, which show the number of times that gray
    level j occurs at a distance d and angle theta from gray level i.
    """  # noqa: D205
    grayscale_image_array = (grayscale_image_array * (n_levels - 1)).astype("uint8")
    glcm: np.ndarray = graycomatrix(
        image=grayscale_image_array,
        distances=[
            1
        ],  # Compares gray levels of pixels at a distance of 1 from a reference pixel.
        angles=[
            0
        ],  # Compares gray levels of pixels at angles of 0, pi/4, pi/2 and 3*pi/4 radians from a reference pixel.
        levels=n_levels,  # Indicates number of gray levels counted (256 in this case)
        symmetric=True,  # Matrix is symmetric e.g. if a gray value of 2 is at angle = pi/4 and distance = 1 from a reference pixel with gray value of 1, an extra count is recorded for both gray levels (2, 1) and (1,2)
        normed=True,
    )  # Matrix is normalized so all values represent probabilities which add up to 1.
    return glcm


def get_haralick_asm(glcm_matrix) -> np.float64:
    """Obtain value for the Haralick metric of Angular Second Moment (ASM)."""
    return float(graycoprops(glcm_matrix, prop="ASM"))


def get_haralick_energy(glcm_matrix) -> np.float64:
    """Obtain value for the Haralick metric of Energy."""
    return float(graycoprops(glcm_matrix, prop="energy"))


def get_haralick_contrast(glcm_matrix) -> np.float64:
    """Obtain value for the Haralick metric of Contrast."""
    return float(graycoprops(glcm_matrix, prop="contrast"))


def get_haralick_correlation(glcm_matrix) -> np.float64:
    """Obtain value for the Haralick metric of Correlation."""
    return float(graycoprops(glcm_matrix, prop="correlation"))


def get_haralick_homogeneity(glcm_matrix) -> np.float64:
    """Obtain value for the Haralick metric of Homogeneity."""
    return float(graycoprops(glcm_matrix, prop="homogeneity"))


def get_haralick_mean(glcm_matrix) -> float:
    """Obtain value for the Haralick metric of Mean."""
    return float(graycoprops(glcm_matrix, prop="mean"))


def get_haralick_std(glcm_matrix) -> float:
    """Obtain value for the Haralick metric of Standard Deviation."""
    return float(graycoprops(glcm_matrix, prop="std"))


def get_one_std_away_points(mean, std) -> float:
    """Obtain coordinates of points which are one Standard Deviation away from the mean."""
    return mean - std, mean + std


def plot_gray_level_cooccurrence_matrix_zero_rad(
    glcm_matrix, mean, std, contrast
) -> None:
    """Produce 2D colour map of gray level cooccurrence matrix for angle of 0 rad."""
    _fig, ax = plt.subplots()
    glcm = glcm_matrix[:, :, 0, 0]
    plt.imshow(glcm, cmap="viridis")  # Use viridis colour mapping when showing matrix
    plt.title("Gray Level Cooccurrence Matrix (distance = 1, angle = 0 rad)")
    plt.colorbar(label="Normalised frequency")
    plt.xlabel("Gray level in j")
    plt.ylabel("Gray level in i")
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
    plt.show()
