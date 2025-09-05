from __future__ import annotations


def test_import() -> None:
    try:
        import haralick_analysis  # noqa: PLC0415
    except ImportError:
        haralick_analysis = None

    assert haralick_analysis is not None, "haralick_analysis module should not be None"
