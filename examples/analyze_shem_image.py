from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt

from haralick_analysis import (
    CoOccurrenceMatrix,
    CoOccurrenceParams,
    plot_co_occurrence_matrix,
)

FILENAME = Path("./data/example_shem_image.jpg")


if __name__ == "__main__":
    params = CoOccurrenceParams(n_levels=100, angle=0, distance=2)
    co_occurrence_matrix = CoOccurrenceMatrix.from_file(FILENAME, params=params)

    print(f"ASM = {co_occurrence_matrix.haralick_asm:.3g}")
    print(f"Energy = {co_occurrence_matrix.haralick_energy:.3g}")
    print(f"Contrast = {co_occurrence_matrix.haralick_contrast:.3g}")
    print(f"Correlation = {co_occurrence_matrix.haralick_correlation:.3g}")
    print(f"Homogeneity = {co_occurrence_matrix.haralick_homogeneity:.3g}")
    print(f"Mean = {co_occurrence_matrix.haralick_mean:.3g}")
    print(f"Standard Deviation = {co_occurrence_matrix.haralick_std:.3g}")

    fig, ax = plt.subplots()
    fig, _ax = plot_co_occurrence_matrix(co_occurrence_matrix, ax=ax)

    plt.show()
