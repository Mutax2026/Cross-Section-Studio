"""Tests for the multi-tab input workbook template."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import DataParser
from ingestion import DATA_ENTRY_PROFILE_ID, FormatDetector, ingest_workbook
from workbook_template import (
    build_input_template,
    build_input_template_bytes,
    export_cleaned_workbook_bytes,
    load_data_entry_sheets,
    load_project_metadata,
)


EXPECTED_SHEETS = {
    "Instructions",
    "Project",
    "Collars",
    "Lithology",
    "Water",
    "Environmental",
    "Screens",
    "Gradients",
    "Sections",
    "Example",
    "Data Entry",
}


def test_build_input_template_writes_multi_tabs(tmp_path: Path) -> None:
    path = build_input_template(tmp_path / "template.xlsx")
    import pandas as pd

    with pd.ExcelFile(path) as workbook:
        sheet_names = workbook.sheet_names
    assert EXPECTED_SHEETS.issubset(set(sheet_names))
    assert sheet_names[0] == "Instructions"
    assert sheet_names[1] == "Project"
    assert sheet_names[-1] == "Data Entry"


def test_build_input_template_bytes_matches_sheet_set() -> None:
    from io import BytesIO

    import pandas as pd

    payload = build_input_template_bytes()
    assert isinstance(payload, (bytes, bytearray))
    assert len(payload) > 1000
    workbook = pd.ExcelFile(BytesIO(payload))
    assert EXPECTED_SHEETS.issubset(set(workbook.sheet_names))


def test_format_detector_recognizes_data_entry_template(tmp_path: Path) -> None:
    path = build_input_template(tmp_path / "template.xlsx")
    detection = FormatDetector().detect(path)
    assert detection.profile_id == DATA_ENTRY_PROFILE_ID
    assert detection.confidence == 1.0


def test_ingest_workbook_data_entry_template(tmp_path: Path) -> None:
    path = build_input_template(tmp_path / "template.xlsx")
    result, report = ingest_workbook(path)
    assert report.profile_id == DATA_ENTRY_PROFILE_ID
    assert len(result.collars) == 3
    assert len(result.lithologies) >= 8
    assert len(result.environmental_readings) == 5
    assert "Data Entry" in report.optional_sheets_detected
    assert "Sections" in report.optional_sheets_detected
    assert len(result.section_specs) == 2
    assert result.section_specs[0].label == "A-A'"
    assert result.section_specs[0].hole_ids == ("MW-01", "MW-02", "MW-03")
    assert result.section_specs[1].hole_ids == ("MW-01", "MW-03")
    assert report.section_specs == list(result.section_specs)
    assert report.project_metadata["client_name"] == "C-GROUP ENERGY INC."
    assert report.project_metadata["prepared_by"] == "ECOVENTURE"


def test_data_entry_template_parses_in_dataparser(tmp_path: Path) -> None:
    path = build_input_template(tmp_path / "template.xlsx")
    result = DataParser().parse_file(path)
    assert len(result.collars) == 3
    assert len(result.lithologies) >= 8
    assert len(result.water_levels) == 6
    assert any(level.status == "dry" for level in result.water_levels)
    assert len(result.environmental_readings) == 5
    assert len(result.screen_intervals) == 2
    assert len(result.vertical_gradients) == 2
    chloride = [reading for reading in result.environmental_readings if reading.parameter == "Chloride"]
    assert len(chloride) == 4
    # The sample non-detect uses value + qualifier, not a "<5 mg/L" label
    # that would repeat the unit column (QA round 7).
    non_detects = [reading for reading in chloride if not reading.detected]
    assert len(non_detects) == 1 and non_detects[0].value == 5.0
    assert not any(reading.value_label for reading in result.environmental_readings)


def test_load_data_entry_project_metadata(tmp_path: Path) -> None:
    path = build_input_template(tmp_path / "template.xlsx")
    sheets = load_data_entry_sheets(path)
    assert sheets.project["client_name"] == "C-GROUP ENERGY INC."
    assert sheets.project["section_title"].startswith("A - A")
    project = load_project_metadata(path)
    assert project["client_name"] == "C-GROUP ENERGY INC."
    assert project["map_scale"] == "1:1000"


def test_example_sheet_is_not_required_for_parse(tmp_path: Path) -> None:
    """Example is reference-only; geology comes from named Collars/Lithology tabs."""
    path = build_input_template(tmp_path / "template.xlsx")
    import pandas as pd
    from openpyxl import load_workbook

    workbook = load_workbook(path)
    workbook.remove(workbook["Example"])
    workbook.save(path)
    result = DataParser().parse_file(path)
    assert len(result.collars) == 3
    collars = pd.read_excel(path, sheet_name="Collars")
    assert "MW-01" in set(collars["hole_id"].astype(str))


def test_hybrid_native_plus_data_entry_keeps_environmental(tmp_path: Path) -> None:
    path = build_input_template(tmp_path / "hybrid.xlsx")
    # Template already has native Collars/Lithology; overlays come from those sheets
    # (Data Entry geology sections are empty stubs).
    hybrid = DataParser().parse_file(path)
    assert len(hybrid.collars) == 3
    assert len(hybrid.environmental_readings) == 5
    assert len(hybrid.water_levels) == 6
    assert any(level.status == "dry" for level in hybrid.water_levels)
    # Ensure Data Entry still carries PROJECT for metadata path
    sheets = load_data_entry_sheets(path)
    assert sheets.project["client_name"] == "C-GROUP ENERGY INC."
    assert sheets.collars.empty
    assert sheets.lithology.empty


def test_ingest_native_multi_tab_keeps_overlays(tmp_path: Path) -> None:
    """Native detect path must not drop Water/Environmental when Collars exist."""
    from openpyxl import load_workbook

    path = build_input_template(tmp_path / "native_overlays.xlsx")
    workbook = load_workbook(path)
    # Remove Data Entry so FormatDetector chooses native_platform.
    workbook.remove(workbook["Data Entry"])
    workbook.save(path)
    detection = FormatDetector().detect(path)
    assert detection.profile_id == "native_platform"
    result, _report = ingest_workbook(path)
    assert len(result.collars) == 3
    assert len(result.water_levels) == 6
    assert any(level.status == "dry" for level in result.water_levels)
    assert len(result.environmental_readings) == 5
    assert len(result.screen_intervals) == 2


def test_load_project_metadata_merges_partial_project(tmp_path: Path) -> None:
    import pandas as pd

    path = build_input_template(tmp_path / "merge_meta.xlsx")
    # Wipe most Project values but keep one field — Data Entry should fill the rest.
    frame = pd.read_excel(path, sheet_name="Project")
    frame.loc[frame["field"] != "notes", "value"] = ""
    frame.loc[frame["field"] == "notes", "value"] = "Only notes on Project tab"
    with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        frame.to_excel(writer, sheet_name="Project", index=False)
    project = load_project_metadata(path)
    assert project["notes"] == "Only notes on Project tab"
    assert project["client_name"] == "C-GROUP ENERGY INC."


def test_export_cleaned_workbook_bytes_roundtrip() -> None:
    path = build_input_template()
    result, _report = ingest_workbook(path)
    payload = export_cleaned_workbook_bytes(
        result,
        project_metadata={"client_name": "TEST CLIENT", "figure_preset": "gwm_fence"},
    )
    assert isinstance(payload, (bytes, bytearray))
    assert len(payload) > 500
    cleaned, report2 = ingest_workbook(BytesIO(payload))
    # Project metadata must come back on re-import (it seeds the title block).
    assert report2.project_metadata.get("client_name") == "TEST CLIENT"
    assert len(cleaned.collars) == len(result.collars)
    assert len(cleaned.lithologies) >= 1
    # Every water row must survive (elevation_masl rows used to be rejected on
    # re-ingest because the export wrote both depth and elevation_masl).
    assert len(cleaned.water_levels) == len(result.water_levels)
    assert not [e for e in cleaned.errors if "not both" in e]
    assert len(cleaned.vertical_gradients) == len(result.vertical_gradients)
    assert len(cleaned.section_specs) == len(result.section_specs)
    project = load_project_metadata(BytesIO(payload))
    assert project.get("client_name") == "TEST CLIENT"
    assert project.get("figure_preset") == "gwm_fence"

def test_data_entry_hole_ids_that_start_with_a_banner_word_stay_data_rows() -> None:
    """Review F3: WATER-01 / WATERLOO-2 / PROJECT-7 were read as section banners,
    dropping the hole and every row beneath it with no error."""
    import pandas as pd

    from workbook_template import parse_data_entry_sheet

    frame = pd.DataFrame(
        [
            ["COLLARS", "", "", "", ""],
            ["hole_id", "easting", "northing", "elevation", "total_depth"],
            ["BH-01", 0, 0, 100, 10],
            ["WATER-01", 10, 0, 100, 10],
            ["PROJECT-7", 15, 0, 100, 10],
            ["BH-02", 20, 0, 100, 10],
            ["LITHOLOGY (required)", "", "", "", ""],
            ["hole_id", "from_depth", "to_depth", "lithology_code", ""],
            ["BH-01", 0, 10, "Clay", ""],
            ["WATER-01", 0, 10, "Clay", ""],
            ["WATERLOO-2", 0, 10, "Clay", ""],
            ["Screens-A", 0, 10, "Sand", ""],
            ["BH-02", 0, 10, "Clay", ""],
            ["Water (optional)", "", "", "", ""],
            ["hole_id", "depth", "", "", ""],
            ["WATER-01", 3, "", "", ""],
        ]
    )
    sheets = parse_data_entry_sheet(frame)
    assert list(sheets.collars["hole_id"]) == ["BH-01", "WATER-01", "PROJECT-7", "BH-02"]
    assert list(sheets.lithology["hole_id"]) == [
        "BH-01",
        "WATER-01",
        "WATERLOO-2",
        "Screens-A",
        "BH-02",
    ]
    assert list(sheets.water["hole_id"]) == ["WATER-01"]


def test_data_entry_banners_as_the_template_writes_them_still_open_sections() -> None:
    """Every banner on the template's Example sheet (same layout as Data Entry)
    still opens its section after the exact-banner match."""
    import pandas as pd

    from workbook_template import parse_data_entry_sheet

    workbook = pd.ExcelFile(BytesIO(build_input_template_bytes()))
    sheets = parse_data_entry_sheet(pd.read_excel(workbook, "Example", header=None))
    for name in ("collars", "lithology", "water", "environmental", "screens", "gradients"):
        assert not getattr(sheets, name).empty, name
    assert sheets.project  # "PROJECT / CLIENT METADATA" block


def _data_entry_frame(project: str, collars: str, lithology: str, water: str, extra_rows=()):
    import pandas as pd

    return pd.DataFrame(
        [
            [project, "value", "", "", ""],
            ["client_name", "Demo", "", "", ""],
            ["", "", "", "", ""],
            [collars, "", "", "", ""],
            ["hole_id", "easting", "northing", "elevation", "total_depth"],
            ["BH-1", 500000, 6000000, 650, 10],
            ["BH-2", 500050, 6000000, 651, 12],
            *extra_rows,
            ["", "", "", "", ""],
            [lithology, "", "", "", ""],
            ["hole_id", "from_depth", "to_depth", "lithology_code", ""],
            ["BH-1", 0, 5, "Clay", ""],
            ["BH-1", 5, 10, "Sand", ""],
            ["BH-2", 0, 12, "Clay", ""],
            ["", "", "", "", ""],
            [water, "", "", "", ""],
            ["hole_id", "depth", "", "", ""],
            ["BH-1", 3.2, "", "", ""],
            ["BH-2", 3.5, "", "", ""],
        ]
    )


@pytest.mark.parametrize(
    ("project", "collars", "lithology", "water"),
    [
        ("PROJECT / CLIENT METADATA", "COLLARS (required)", "LITHOLOGY (required)", "WATER (optional)"),
        ("PROJECT METADATA", "COLLARS", "LITHOLOGY", "WATER LEVELS"),
        ("Project", "COLLARS:", "LITHOLOGY:", "WATER:"),
        ("PROJECT:", "COLLARS (required):", "LITHOLOGY LOG", "Water Levels (optional)"),
        ("project metadata", "COLLARS - required", "Lithology Log (required)", "Water levels"),
        ("PROJECT", "Collars — required", "lithology data", "water / levels"),
        ("PROJECT", "collars ( required )", "LITHOLOGY – optional", "WATER -"),
        ("PROJECT", "COLLARS:", "Lithology:", "WATER"),
        ("PROJECT", "COLLARS:", "LITHOLOGY DATA", "WATER"),
        ("PROJECT", "COLLARS:", "Lithology intervals", "WATER"),
        ("PROJECT", "COLLARS:", "LITHOLOGY - required", "WATER"),
    ],
)
def test_data_entry_hand_typed_banner_spellings_open_their_sections(
    project: str, collars: str, lithology: str, water: str
) -> None:
    """Review: hand-typed banners ("COLLARS:", "WATER LEVELS", "LITHOLOGY LOG") were
    read as data rows, so the next header row remapped the previous section."""
    from workbook_template import parse_data_entry_sheet

    sheets = parse_data_entry_sheet(_data_entry_frame(project, collars, lithology, water))
    assert sheets.project == {"client_name": "Demo"}
    assert list(sheets.collars["hole_id"]) == ["BH-1", "BH-2"]
    assert list(sheets.collars["total_depth"]) == [10, 12]
    assert list(sheets.lithology["hole_id"]) == ["BH-1", "BH-1", "BH-2"]
    assert list(sheets.lithology["to_depth"]) == [5, 10, 12]
    assert list(sheets.water["hole_id"]) == ["BH-1", "BH-2"]
    assert list(sheets.water["depth"]) == [3.2, 3.5]


@pytest.mark.parametrize(
    "hole_id",
    [
        "WATER-01",
        "WATERLOO-2",
        "WATER_1",
        "WATER 1",
        "Water-A",
        "PROJECT-7",
        "Project_7",
        "COLLARS2",
        "LITHOLOGY-3",
        "SCREENS-A",
        "GRADIENTS1",
    ],
)
def test_data_entry_hole_ids_spelled_like_banners_stay_collar_rows(hole_id: str) -> None:
    from workbook_template import parse_data_entry_sheet

    frame = _data_entry_frame(
        "PROJECT / CLIENT METADATA",
        "COLLARS (required)",
        "LITHOLOGY (required)",
        "WATER (optional)",
        extra_rows=[[hole_id, 500100, 6000000, 652, 8]],
    )
    sheets = parse_data_entry_sheet(frame)
    assert list(sheets.collars["hole_id"]) == ["BH-1", "BH-2", hole_id]
    assert list(sheets.lithology["hole_id"]) == ["BH-1", "BH-1", "BH-2"]
    assert list(sheets.water["hole_id"]) == ["BH-1", "BH-2"]


@pytest.mark.parametrize("hole_id", ["WATER_1", "WATER 1", "COLLARS2", "PROJECT-7"])
def test_banner_pattern_rejects_hole_ids_even_without_numbers_beside_them(hole_id: str) -> None:
    """A hole id typed with its other cells still blank must not open a section."""
    from workbook_template import _PROJECT_BANNER, _SECTION_BANNERS

    assert not any(p.fullmatch(hole_id) for p in (*_SECTION_BANNERS.values(), _PROJECT_BANNER))


def test_data_entry_banner_like_first_cell_beside_numbers_is_a_data_row() -> None:
    """A hole named "Water Well" logged with coordinates is a collar, not a banner."""
    from workbook_template import parse_data_entry_sheet

    frame = _data_entry_frame(
        "PROJECT",
        "COLLARS",
        "LITHOLOGY",
        "WATER",
        extra_rows=[["Water Well", 500100, 6000000, 652, 8], ["Lithology: pit", 1, 2, 3, 4]],
    )
    sheets = parse_data_entry_sheet(frame)
    assert list(sheets.collars["hole_id"]) == ["BH-1", "BH-2", "Water Well", "Lithology: pit"]
    assert list(sheets.water["depth"]) == [3.2, 3.5]


def test_data_entry_hand_typed_banners_ingest_end_to_end(tmp_path) -> None:
    """The reviewer's Data-Entry-only workbook: every section loads (no
    "Missing required sheet(s)")."""
    import pandas as pd

    frame = _data_entry_frame(
        "PROJECT / CLIENT METADATA", "COLLARS:", "LITHOLOGY LOG", "Water Levels (optional)"
    )
    path = tmp_path / "hand_typed.xlsx"
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name="Data Entry", header=False, index=False)
    result, _ = ingest_workbook(path)
    assert len(result.collars) == 2
    assert len(result.lithologies) == 3
    assert len(result.water_levels) == 2
