"""Compare platform-generated figures against reference PNG extracts.

Hard-fails missing pairs and empty/invalid dimensions. Default MSE gates use
suite ceilings (``GWM_MSE_CEILING`` / ``P2_MSE_CEILING``) as regression guards;
CAD pixel-match under ~2500 is not expected. Pass ``--mse-threshold`` to force
one threshold for all suites. ``--warn-only`` softens MSE only. SSIM is always
soft (reported, never fails). Optional ``--suite p2`` / ``all`` compares
Advantage P2 pairs when fixtures exist.

When native sizes differ, the generated image is fit into the reference frame
with letterboxing (aspect preserved, white pad) so letter-vs-tabloid exports
are not stretched. Pass ``--stretch`` to restore bilinear stretch-to-fit.

MSE alone cannot tell a figure from an empty page (a blank white page scores a
lower MSE than the real GWM Fig 3) or from the wrong sheet. Two content checks
therefore hard-fail regardless of ``--warn-only``:

* ink floor: the generated page must carry at least ``INK_RATIO_FLOOR`` of the
  reference page's ink (non-white pixel fraction);
* section label: the generated figure's SVG sidecar must title the section the
  pair expects (``CROSS SECTION A - A'`` for ``fig_3``) and the reference file
  name must name the same section, so a swapped sheet on either side fails.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REFERENCE_DIR = ROOT / "data" / "pdf_extract"
GENERATED_DIR = ROOT / "data" / "ecoventure_gwm"
P2_REFERENCE_DIR = ROOT / "data" / "pdf_extract_p2"
P2_GENERATED_DIR = ROOT / "data" / "Data2" / "test_output"

# Regression ceilings for CAD-vs-linear-fence floors (letterbox, hatches on).
# Aspirational visual target remains lower (~2500) but is not the default CI gate.
GWM_MSE_CEILING = 12000.0
P2_MSE_CEILING = 4500.0

FIGURE_PAIRS: dict[str, tuple[str, str]] = {
    "fig_3": (
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 3 Cross Section A-A'.png",
        "fig_3_cross_section_a_a.png",
    ),
    "fig_4": (
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 4 Cross Section B-B'.png",
        "fig_4_cross_section_b_b.png",
    ),
    "fig_5": (
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 5 Cross Section C-C'.png",
        "fig_5_cross_section_c_c.png",
    ),
    "fig_6": (
        "2605011D_09_16-29-057-19 W4M_25-GWM_Fig 6 Cross Section D-D'.png",
        "fig_6_cross_section_d_d.png",
    ),
}

# Content floor. A pixel counts as ink when its grey level is below
# ``INK_LEVEL``; the generated page must carry at least ``INK_RATIO_FLOOR`` times
# the reference page's ink fraction (both measured at native size, so letterbox
# padding does not count). Measured 2026-11-05 (generated/reference ink ratio):
#   matched pairs  gwm fig_3 2.232, fig_4 3.187, fig_5 2.410, fig_6 3.015,
#                  p2 fig_6 0.890, p2 fig_7 0.858  (lowest 0.858)
#   blank page     0.000 for every reference
# 0.25 leaves a 3.4x margin under the lowest real pair and still rejects a page
# that keeps only a frame or title block.
INK_LEVEL = 200
INK_RATIO_FLOOR = 0.25

# Pixel similarity cannot separate a swapped sheet from the right one: on the
# same date, 16x16 ink-grid correlation scored matched pairs 0.171-0.825 (GWM)
# and 0.292-0.295 (P2) but swapped pairs up to 0.587 (gwm ref fig_4 vs
# generated fig_6), and a numpy block SSIM (64x64 windows) scored matched
# 0.414-0.537 against swapped 0.375-0.520 and blank 0.510-0.677. So the swap
# check reads the section title from the generated figure's SVG sidecar.
SECTION_LABELS: dict[str, str] = {
    "fig_3": "A",
    "fig_4": "B",
    "fig_5": "C",
    "fig_6": "D",
    "p2_fig_6": "A",
    "p2_fig_7": "B",
}

# Title wording varies: any case, "CROSS-SECTION", en/em dashes, and a
# straight, curly or prime apostrophe after the second letter.
_SECTION_TITLE_RE = re.compile(
    r"CROSS[\s-]SECTION\s+([A-Z])\s*[-‐-―]\s*([A-Z])\s*['’′]",
    re.IGNORECASE,
)
# Reference extracts: "...Fig 3 Cross Section A-A'.png" / "fig_6_cross_section_a_a.png".
_SECTION_FILENAME_RE = re.compile(r"cross[ _]section[ _]([a-z])[-_ ]([a-z])", re.IGNORECASE)

P2_FIGURE_PAIRS: dict[str, tuple[str, str]] = {
    "p2_fig_6": (
        "fig_6_cross_section_a_a.png",
        "fig6_aa_lithology_chlorides.png",
    ),
    "p2_fig_7": (
        "fig_7_cross_section_b_b.png",
        "fig7_bb_lithology_chlorides.png",
    ),
}


def _load_grayscale(path: Path):
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float64)
    if arr.size == 0 or min(arr.shape[:2]) < 1:
        raise ValueError(f"empty or invalid image: {path}")
    return arr


def ink_fraction(arr) -> float:
    """Fraction of pixels darker than ``INK_LEVEL`` (0 for a blank page)."""
    return float((arr < INK_LEVEL).mean())


def section_labels_from_svg(generated_png: Path) -> set[str] | None:
    """Section letters titled in the SVG saved beside ``generated_png``.

    Returns ``None`` when the sidecar is missing. matplotlib writes each text
    artist's string as an XML comment, so the title survives path-rendered text.
    A well-formed title names the same letter twice (``A - A'``); a title whose
    two letters differ is returned as ``"A-B"`` so it can never match.
    """
    svg = generated_png.with_suffix(".svg")
    if not svg.is_file():
        return None
    text = html.unescape(svg.read_text(encoding="utf-8", errors="replace"))
    labels: set[str] = set()
    for start, end in _SECTION_TITLE_RE.findall(text):
        start, end = start.upper(), end.upper()
        labels.add(start if start == end else f"{start}-{end}")
    return labels


def section_label_from_filename(path: Path) -> str | None:
    """Section letter named in a reference file name, or ``None``."""
    match = _SECTION_FILENAME_RE.search(path.name)
    if match is None:
        return None
    start, end = match.group(1).upper(), match.group(2).upper()
    return start if start == end else f"{start}-{end}"


def _resize_to_match(reference, generated, *, stretch: bool = False):
    """Align arrays to the reference shape for MSE/SSIM.

    Default: scale generated to fit inside the reference frame (aspect preserved)
    and letterbox with white (255). ``stretch=True`` bilinear-stretches generated
    to the reference size (legacy behavior).
    """
    from PIL import Image
    import numpy as np

    ref_h, ref_w = int(reference.shape[0]), int(reference.shape[1])
    gen_h, gen_w = int(generated.shape[0]), int(generated.shape[1])
    if (gen_w, gen_h) == (ref_w, ref_h):
        return reference, generated.astype(np.float64, copy=False)

    gen_img = Image.fromarray(generated.astype("uint8"))
    if stretch:
        gen_resized = gen_img.resize((ref_w, ref_h), Image.Resampling.BILINEAR)
        return reference, np.asarray(gen_resized, dtype=np.float64)

    scale = min(ref_w / gen_w, ref_h / gen_h)
    fit_w = max(1, int(round(gen_w * scale)))
    fit_h = max(1, int(round(gen_h * scale)))
    # Guard against rounding past the reference frame on either axis.
    fit_w = min(fit_w, ref_w)
    fit_h = min(fit_h, ref_h)
    fitted = gen_img.resize((fit_w, fit_h), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (ref_w, ref_h), color=255)
    offset = ((ref_w - fit_w) // 2, (ref_h - fit_h) // 2)
    canvas.paste(fitted, offset)
    return reference, np.asarray(canvas, dtype=np.float64)


def compare_pair(
    reference: Path,
    generated: Path,
    *,
    mse_threshold: float,
    require_same_size: bool = False,
    size_tolerance_px: int = 0,
    stretch: bool = False,
    expected_section: str | None = None,
) -> dict[str, object]:
    if not reference.is_file():
        raise FileNotFoundError(f"Reference missing: {reference}")
    if not generated.is_file():
        raise FileNotFoundError(f"Generated missing: {generated}")

    ref_native = _load_grayscale(reference)
    gen_native = _load_grayscale(generated)
    ref_wh = (int(ref_native.shape[1]), int(ref_native.shape[0]))
    gen_wh = (int(gen_native.shape[1]), int(gen_native.shape[0]))
    size_ok = (
        abs(ref_wh[0] - gen_wh[0]) <= size_tolerance_px
        and abs(ref_wh[1] - gen_wh[1]) <= size_tolerance_px
    )
    # Always require non-empty valid dimensions (structural).
    dims_valid = min(ref_wh) > 0 and min(gen_wh) > 0

    ref, gen = _resize_to_match(ref_native, gen_native, stretch=stretch)
    mse = float(((ref - gen) ** 2).mean())
    result: dict[str, object] = {
        "reference": str(reference),
        "generated": str(generated),
        "native_reference_size": ref_wh,
        "native_generated_size": gen_wh,
        "size_ok": size_ok,
        "dims_valid": dims_valid,
        "mse": mse,
        "mse_ok": mse <= mse_threshold,
    }
    if require_same_size and not size_ok:
        result["mse_ok"] = False

    ref_ink = ink_fraction(ref_native)
    gen_ink = ink_fraction(gen_native)
    ink_ratio = gen_ink / ref_ink if ref_ink > 0 else 0.0
    result["reference_ink"] = ref_ink
    result["generated_ink"] = gen_ink
    result["ink_ratio"] = ink_ratio
    # A reference with no ink is itself broken; fail rather than divide by zero.
    result["ink_ok"] = ref_ink > 0 and ink_ratio >= INK_RATIO_FLOOR

    if expected_section is not None:
        labels = section_labels_from_svg(generated)
        result["section_labels"] = None if labels is None else sorted(labels)
        ref_label = section_label_from_filename(reference)
        result["reference_section_label"] = ref_label
        result["label_ok"] = labels == {expected_section} and ref_label == expected_section
    try:
        from skimage.metrics import structural_similarity as ssim  # type: ignore[import-untyped]

        score = float(ssim(ref, gen, data_range=255.0))
        result["ssim"] = score
        result["ssim_ok"] = score >= 0.25
    except ImportError:
        result["ssim"] = None
        result["ssim_ok"] = None
    return result


def _pairs_available(
    pairs: dict[str, tuple[str, str]],
    reference_dir: Path,
    generated_dir: Path,
) -> bool:
    return all(
        (reference_dir / ref_name).is_file() and (generated_dir / gen_name).is_file()
        for ref_name, gen_name in pairs.values()
    )


def _run_suite(
    *,
    label: str,
    pairs: dict[str, tuple[str, str]],
    keys: list[str],
    reference_dir: Path,
    generated_dir: Path,
    mse_threshold: float,
    require_same_size: bool,
    size_tolerance_px: int,
    warn_mse: bool,
    stretch: bool = False,
) -> list[str]:
    failures: list[str] = []
    for key in keys:
        if key not in pairs:
            continue
        ref_name, gen_name = pairs[key]
        try:
            metrics = compare_pair(
                reference_dir / ref_name,
                generated_dir / gen_name,
                mse_threshold=mse_threshold,
                require_same_size=require_same_size,
                size_tolerance_px=size_tolerance_px,
                stretch=stretch,
                expected_section=SECTION_LABELS.get(key),
            )
        except FileNotFoundError as exc:
            print(f"MISSING {label}/{key}: {exc}", file=sys.stderr)
            failures.append(f"missing:{label}/{key}")
            continue
        except ValueError as exc:
            print(f"INVALID {label}/{key}: {exc}", file=sys.stderr)
            failures.append(f"invalid:{label}/{key}")
            continue

        mse = metrics["mse"]
        ssim_val = metrics.get("ssim")
        native_ref = metrics["native_reference_size"]
        native_gen = metrics["native_generated_size"]
        line = (
            f"{label}/{key}: size {native_gen} vs ref {native_ref} "
            f"size_ok={metrics['size_ok']} mse={mse:.1f} mse_ok={metrics['mse_ok']} "
            f"ink_ratio={metrics['ink_ratio']:.3f} ink_ok={metrics['ink_ok']}"
        )
        if "label_ok" in metrics:
            line += (
                f" section={metrics['section_labels']} "
                f"ref_section={metrics['reference_section_label']} label_ok={metrics['label_ok']}"
            )
        if ssim_val is not None:
            line += f" ssim={ssim_val:.3f} ssim_ok={metrics['ssim_ok']}"
            if metrics.get("ssim_ok") is False:
                print(f"WARN {label}/{key}: ssim below threshold (soft)", file=sys.stderr)
        print(line)

        if not metrics.get("dims_valid", True):
            failures.append(f"{label}/{key} invalid dimensions")
        # Content checks are not MSE: --warn-only never softens them.
        if not metrics["ink_ok"]:
            failures.append(
                f"{label}/{key} ink_ratio={metrics['ink_ratio']:.3f} "
                f"< {INK_RATIO_FLOOR:g} (blank or near-blank page)"
            )
        if key in SECTION_LABELS and not metrics.get("label_ok", False):
            failures.append(
                f"{label}/{key} generated section {metrics.get('section_labels')} / "
                f"reference section {metrics.get('reference_section_label')!r} "
                f"!= expected {SECTION_LABELS[key]!r} (missing SVG sidecar or swapped sheet)"
            )
        if require_same_size and not metrics["size_ok"]:
            failures.append(
                f"{label}/{key} size {native_gen} != {native_ref}"
            )
        if not metrics["mse_ok"]:
            msg = f"{label}/{key} mse={mse:.1f}"
            if warn_mse:
                print(f"WARN {msg} (soft)", file=sys.stderr)
            else:
                failures.append(msg)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare platform PNGs to reference extracts")
    parser.add_argument(
        "--suite",
        choices=("gwm", "p2", "all"),
        default="gwm",
        help="Which figure suite to compare (p2 skipped when fixtures absent)",
    )
    parser.add_argument(
        "--figure",
        choices=(*FIGURE_PAIRS.keys(), *P2_FIGURE_PAIRS.keys(), "all"),
        default="all",
        help="Figure key within the selected suite(s)",
    )
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--generated-dir", type=Path, default=GENERATED_DIR)
    parser.add_argument("--p2-reference-dir", type=Path, default=P2_REFERENCE_DIR)
    parser.add_argument("--p2-generated-dir", type=Path, default=P2_GENERATED_DIR)
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=None,
        help=(
            "Force a single MSE threshold for all suites (overrides suite ceilings). "
            "Suite ceilings are regression guards; CAD pixel-match under ~2500 is not expected"
        ),
    )
    parser.add_argument(
        "--mse-threshold-gwm",
        type=float,
        default=None,
        help=f"GWM suite MSE ceiling (default: {GWM_MSE_CEILING:g})",
    )
    parser.add_argument(
        "--mse-threshold-p2",
        type=float,
        default=None,
        help=f"P2 suite MSE ceiling (default: {P2_MSE_CEILING:g})",
    )
    parser.add_argument(
        "--require-same-size",
        action="store_true",
        help="Hard-fail when native PNG dimensions differ (default: warn via size_ok only)",
    )
    parser.add_argument(
        "--size-tolerance-px",
        type=int,
        default=0,
        help="Pixel tolerance when --require-same-size is set",
    )
    parser.add_argument(
        "--stretch",
        action="store_true",
        help="Bilinear-stretch generated to reference size (legacy); default is letterbox",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Soften MSE failures to warnings; missing/invalid pairs still fail",
    )
    args = parser.parse_args()

    # Default: suite ceilings. --mse-threshold forces one gate for all suites.
    if args.mse_threshold is not None:
        gwm_mse = p2_mse = args.mse_threshold
    else:
        gwm_mse = (
            args.mse_threshold_gwm
            if args.mse_threshold_gwm is not None
            else GWM_MSE_CEILING
        )
        p2_mse = (
            args.mse_threshold_p2
            if args.mse_threshold_p2 is not None
            else P2_MSE_CEILING
        )

    failures: list[str] = []
    run_gwm = args.suite in ("gwm", "all")
    run_p2 = args.suite in ("p2", "all")

    if run_gwm:
        gwm_keys = list(FIGURE_PAIRS.keys()) if args.figure == "all" else [args.figure]
        failures.extend(
            _run_suite(
                label="gwm",
                pairs=FIGURE_PAIRS,
                keys=gwm_keys,
                reference_dir=args.reference_dir,
                generated_dir=args.generated_dir,
                mse_threshold=gwm_mse,
                require_same_size=args.require_same_size,
                size_tolerance_px=args.size_tolerance_px,
                warn_mse=args.warn_only,
                stretch=args.stretch,
            )
        )

    if run_p2:
        if not _pairs_available(P2_FIGURE_PAIRS, args.p2_reference_dir, args.p2_generated_dir):
            print("P2 parity skipped — reference/generated PNG fixtures absent")
        else:
            p2_keys = list(P2_FIGURE_PAIRS.keys()) if args.figure == "all" else [args.figure]
            failures.extend(
                _run_suite(
                    label="p2",
                    pairs=P2_FIGURE_PAIRS,
                    keys=p2_keys,
                    reference_dir=args.p2_reference_dir,
                    generated_dir=args.p2_generated_dir,
                    mse_threshold=p2_mse,
                    require_same_size=args.require_same_size,
                    size_tolerance_px=args.size_tolerance_px,
                    warn_mse=args.warn_only,
                    stretch=args.stretch,
                )
            )

    if failures:
        print(f"FAILED: {len(failures)} issue(s)", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
