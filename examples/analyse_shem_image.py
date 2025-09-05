from __future__ import annotations

from pathlib import Path

from haralick_analysis.haralick_analysis_scikit import (
    get_gray_level_cooccurrence_matrix,
    get_haralick_asm,
    get_haralick_contrast,
    get_haralick_correlation,
    get_haralick_energy,
    get_haralick_homogeneity,
    get_haralick_mean,
    get_haralick_std,
    plot_gray_level_cooccurrence_matrix,
    read_image_to_grayscale,
)

FILENAME = Path("./data/example_shem_image.jpg")


def main() -> None:
    """Call all functions in turn. Carry out Haralick Analysis on inputted image."""
    get_grayscale_image = read_image_to_grayscale(FILENAME)

    get_glcm = get_gray_level_cooccurrence_matrix(get_grayscale_image, n_levels=100)

    haralick_asm = get_haralick_asm(get_glcm)
    print(f"ASM = {haralick_asm:.3g}")

    haralick_energy = get_haralick_energy(get_glcm)
    print(f"Energy = {haralick_energy:.3g}")

    haralick_contrast = get_haralick_contrast(get_glcm)
    print(f"Contrast = {haralick_contrast:.3g}")

    haralick_correlation = get_haralick_correlation(get_glcm)
    print(f"Correlation = {haralick_correlation:.3g}")

    haralick_homogeneity = get_haralick_homogeneity(get_glcm)
    print(f"Homogeneity = {haralick_homogeneity:.3g}")

    haralick_mean = get_haralick_mean(get_glcm)
    print(f"Mean = {haralick_mean:.3g}")

    haralick_std = get_haralick_std(get_glcm)
    print(f"Standard Deviation = {haralick_std:.3g}")

    plot_gray_level_cooccurrence_matrix(
        get_glcm, haralick_mean, haralick_std, haralick_contrast
    )


if __name__ == "__main__":
    main()
