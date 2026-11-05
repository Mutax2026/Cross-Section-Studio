"""Tests for Word figure pack export."""

from __future__ import annotations

from io import BytesIO

import pytest

docx = pytest.importorskip("docx")

from docx_export import build_figure_docx_bytes


def tiny_png() -> bytes:
    """A real (4 x 4 px) PNG for the figure slot."""
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    from matplotlib import pyplot as plt

    buffer = BytesIO()
    plt.imsave(buffer, np.zeros((4, 4, 3)), format="png")
    return buffer.getvalue()


def test_build_figure_docx_bytes_minimal() -> None:
    payload = build_figure_docx_bytes(
        png_bytes=tiny_png(),
        caption="Figure 1 — Test section",
        title="Test Section",
        metadata={"project": "Demo"},
    )
    assert payload.startswith(b"PK")
    assert len(payload) > 500
    assert len(docx.Document(BytesIO(payload)).inline_shapes) == 1


@pytest.mark.parametrize("png_bytes", [b"", None])
def test_build_figure_docx_bytes_rejects_missing_figure(png_bytes) -> None:
    """An empty PNG made a "figure pack" with a heading and no figure; the app
    catches ValueError and reports "Word file not made" instead."""
    with pytest.raises(ValueError, match="figure"):
        build_figure_docx_bytes(png_bytes=png_bytes, caption="c", title="T", metadata={"a": 1})
