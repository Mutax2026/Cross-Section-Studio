"""Content checks in scripts/compare_figure_parity.py (blank page, swapped sheet)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_figure_parity as cfp  # noqa: E402

GWM_REF_FIG3 = cfp.REFERENCE_DIR / cfp.FIGURE_PAIRS["fig_3"][0]
GWM_REF_FIG4 = cfp.REFERENCE_DIR / cfp.FIGURE_PAIRS["fig_4"][0]
GWM_GEN_FIG3 = cfp.GENERATED_DIR / cfp.FIGURE_PAIRS["fig_3"][1]

needs_gwm = pytest.mark.skipif(
    not (GWM_REF_FIG3.is_file() and GWM_REF_FIG4.is_file() and GWM_GEN_FIG3.is_file()),
    reason="GWM parity fixtures absent",
)


def _write_png(path: Path, arr: np.ndarray) -> Path:
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).convert("L").save(path)
    return path


def _ink_page(h: int = 200, w: int = 260) -> np.ndarray:
    arr = np.full((h, w), 255, dtype=np.uint8)
    arr[20 : h - 20 : 10, 20 : w - 20] = 0  # horizontal rules
    arr[20 : h - 20, 20 : w - 20 : 15] = 0  # vertical rules
    return arr


def test_ink_fraction_blank_page_is_zero() -> None:
    assert cfp.ink_fraction(np.full((10, 10), 255.0)) == 0.0
    assert cfp.ink_fraction(np.full((10, 10), 250.0)) == 0.0  # near-white is not ink


def test_blank_candidate_fails_ink_floor(tmp_path: Path) -> None:
    ref = _write_png(tmp_path / "ref.png", _ink_page())
    blank = _write_png(tmp_path / "gen.png", np.full((200, 260), 255))
    metrics = cfp.compare_pair(ref, blank, mse_threshold=1e9)
    assert metrics["mse_ok"] is True  # MSE alone would pass it
    assert metrics["ink_ratio"] == 0.0
    assert metrics["ink_ok"] is False


def test_frame_only_candidate_fails_ink_floor(tmp_path: Path) -> None:
    ref = _write_png(tmp_path / "ref.png", _ink_page())
    frame = np.full((200, 260), 255, dtype=np.uint8)
    frame[0, :] = 0
    frame[-1, :] = 0
    frame[:, 0] = 0
    frame[:, -1] = 0
    gen = _write_png(tmp_path / "gen.png", frame)
    metrics = cfp.compare_pair(ref, gen, mse_threshold=1e9)
    assert metrics["ink_ratio"] < cfp.INK_RATIO_FLOOR
    assert metrics["ink_ok"] is False


def test_matching_candidate_passes_ink_floor(tmp_path: Path) -> None:
    ref = _write_png(tmp_path / "ref.png", _ink_page())
    gen = _write_png(tmp_path / "gen.png", _ink_page())
    metrics = cfp.compare_pair(ref, gen, mse_threshold=1e9)
    assert metrics["ink_ratio"] == pytest.approx(1.0)
    assert metrics["ink_ok"] is True


def test_blank_reference_fails_rather_than_dividing_by_zero(tmp_path: Path) -> None:
    ref = _write_png(tmp_path / "ref.png", np.full((60, 60), 255))
    gen = _write_png(tmp_path / "gen.png", _ink_page(60, 60))
    assert cfp.compare_pair(ref, gen, mse_threshold=1e9)["ink_ok"] is False


@needs_gwm
def test_real_gwm_pair_passes_content_checks() -> None:
    metrics = cfp.compare_pair(
        GWM_REF_FIG3,
        GWM_GEN_FIG3,
        mse_threshold=cfp.GWM_MSE_CEILING,
        expected_section="A",
    )
    assert metrics["mse_ok"] is True
    assert metrics["ink_ok"] is True
    assert metrics["section_labels"] == ["A"]
    assert metrics["reference_section_label"] == "A"
    assert metrics["label_ok"] is True


@needs_gwm
def test_blank_page_against_real_reference_fails(tmp_path: Path) -> None:
    blank = _write_png(tmp_path / "fig_3_cross_section_a_a.png", np.full((2550, 3300), 255))
    shutil.copy(GWM_GEN_FIG3.with_suffix(".svg"), blank.with_suffix(".svg"))
    metrics = cfp.compare_pair(
        GWM_REF_FIG3, blank, mse_threshold=cfp.GWM_MSE_CEILING, expected_section="A"
    )
    # The review's repro: a blank page beats the real export on MSE.
    assert metrics["mse_ok"] is True
    assert metrics["ink_ok"] is False


@needs_gwm
def test_swapped_sheet_fails_section_check() -> None:
    # Review repro: generated Fig 3 (A-A') against the Fig 4 (B-B') reference.
    metrics = cfp.compare_pair(
        GWM_REF_FIG4,
        GWM_GEN_FIG3,
        mse_threshold=cfp.GWM_MSE_CEILING,
        expected_section="B",
    )
    assert metrics["mse_ok"] is True  # MSE alone would pass the swap
    assert metrics["section_labels"] == ["A"]
    assert metrics["label_ok"] is False


@needs_gwm
def test_swapped_reference_fails_section_check() -> None:
    metrics = cfp.compare_pair(
        GWM_REF_FIG4,
        GWM_GEN_FIG3,
        mse_threshold=cfp.GWM_MSE_CEILING,
        expected_section="A",
    )
    assert metrics["reference_section_label"] == "B"
    assert metrics["label_ok"] is False


def test_missing_svg_sidecar_fails_section_check(tmp_path: Path) -> None:
    ref = _write_png(tmp_path / "Fig 3 Cross Section A-A'.png", _ink_page())
    gen = _write_png(tmp_path / "gen.png", _ink_page())
    metrics = cfp.compare_pair(ref, gen, mse_threshold=1e9, expected_section="A")
    assert metrics["section_labels"] is None
    assert metrics["label_ok"] is False


def test_svg_title_with_mismatched_letters_never_matches(tmp_path: Path) -> None:
    gen = tmp_path / "gen.png"
    gen.with_suffix(".svg").write_text("<!-- CROSS SECTION A - B' -->", encoding="utf-8")
    assert cfp.section_labels_from_svg(gen) == {"A-B"}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("2605011D_09_16-29-057-19 W4M_25-GWM_Fig 3 Cross Section A-A'.png", "A"),
        ("fig_7_cross_section_b_b.png", "B"),
        ("unrelated.png", None),
    ],
)
def test_section_label_from_reference_filename(name: str, expected: str | None) -> None:
    assert cfp.section_label_from_filename(Path(name)) == expected


def test_every_pair_has_an_expected_section() -> None:
    for key, (ref_name, _gen_name) in {**cfp.FIGURE_PAIRS, **cfp.P2_FIGURE_PAIRS}.items():
        assert cfp.section_label_from_filename(Path(ref_name)) == cfp.SECTION_LABELS[key]


def test_suite_gate_fails_blank_page_even_with_warn_only(tmp_path: Path) -> None:
    ref_dir = tmp_path / "ref"
    gen_dir = tmp_path / "gen"
    ref_dir.mkdir()
    gen_dir.mkdir()
    _write_png(ref_dir / "r.png", _ink_page())
    _write_png(gen_dir / "g.png", np.full((200, 260), 255))
    failures = cfp._run_suite(
        label="t",
        pairs={"x": ("r.png", "g.png")},
        keys=["x"],
        reference_dir=ref_dir,
        generated_dir=gen_dir,
        mse_threshold=1e9,
        require_same_size=False,
        size_tolerance_px=0,
        warn_mse=True,
    )
    assert any("ink_ratio" in f for f in failures)


@needs_gwm
def test_suite_gate_fails_swapped_mapping() -> None:
    failures = cfp._run_suite(
        label="gwm",
        pairs={"fig_4": (cfp.FIGURE_PAIRS["fig_4"][0], cfp.FIGURE_PAIRS["fig_3"][1])},
        keys=["fig_4"],
        reference_dir=cfp.REFERENCE_DIR,
        generated_dir=cfp.GENERATED_DIR,
        mse_threshold=cfp.GWM_MSE_CEILING,
        require_same_size=False,
        size_tolerance_px=0,
        warn_mse=False,
    )
    assert any("section" in f for f in failures)


@pytest.mark.parametrize(
    "title",
    [
        "CROSS SECTION A - A'",
        "Cross Section A-A'",
        "CROSS-SECTION A-A'",
        "CROSS SECTION A–A'",
        "CROSS SECTION A-A’",
        "CROSS SECTION A-A′",
        "CROSS SECTION A-A&#39;",
    ],
)
def test_section_label_accepts_title_variants(tmp_path: Path, title: str) -> None:
    """A correct figure must not fail the label check over case, dash or apostrophe style."""
    png = tmp_path / "fig.png"
    png.write_bytes(b"")
    png.with_suffix(".svg").write_text(f"<svg><text>{title}</text></svg>", encoding="utf-8")
    assert cfp.section_labels_from_svg(png) == {"A"}
