from __future__ import annotations


def test_import() -> None:
    try:
        from haralick_analysis import haralick_analysis_scikit  # noqa: PLC0415
    except ImportError:
        haralick_analysis_scikit = None

    assert haralick_analysis_scikit is not None, "my_project module should not be None"
