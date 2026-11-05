"""Tests for projection.py spatial math."""

from __future__ import annotations

import numpy as np
import pytest

from models import Collar, DeviationReading, Lithology, Transect
from projection import (
    _TransectGeometry,
    _cumulative_distance,
    _find_closest_segment,
    build_minimum_curvature_survey,
    densify_survey,
    interpolate_survey_at_depth,
    project_boreholes,
    project_collar_to_transect,
    select_and_order_holes_near_transect,
    select_holes_near_transect,
    order_holes_along_transect,
    transect_length_m,
)


def test_project_point_on_first_segment(axis_transect: Transect) -> None:
    x_profile, offset = project_collar_to_transect(25.0, 0.0, axis_transect)
    assert x_profile == 25.0
    assert offset == 0.0


def test_project_point_on_second_segment(axis_transect: Transect) -> None:
    x_profile, offset = project_collar_to_transect(100.0, 20.0, axis_transect)
    assert x_profile == 120.0
    assert offset == 0.0


def test_closest_segment_at_corner() -> None:
    transect = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0)]
    b = np.array([100.0, 10.0])
    segment_index, t, projected, distance = _find_closest_segment(b, transect)
    assert segment_index == 1
    assert t == pytest.approx(0.2)
    assert projected[0] == pytest.approx(100.0)
    assert projected[1] == pytest.approx(10.0)
    assert distance == pytest.approx(0.0)


def test_cumulative_distance_matches_segment_sum() -> None:
    transect = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0)]
    distance = _cumulative_distance(transect, segment_index=1, t=0.4)
    assert distance == pytest.approx(120.0)


def test_project_boreholes_elevations(axis_collars, axis_lithologies, axis_transect) -> None:
    projected = project_boreholes(axis_collars, axis_lithologies, axis_transect)
    bh01 = projected[projected["hole_id"] == "BH-01"].iloc[0]
    assert bh01["top_elevation"] == 100.0
    assert bh01["bottom_elevation"] == 95.0
    assert projected["x_profile"].is_monotonic_increasing


def test_project_boreholes_perpendicular_offset() -> None:
    collars = [Collar(hole_id="BH-01", easting=0.0, northing=10.0, elevation=50.0, total_depth=10.0)]
    lithologies = [Lithology(hole_id="BH-01", from_depth=0.0, to_depth=5.0, lithology_code="Clay")]
    transect = Transect(points=[(0.0, 0.0), (100.0, 0.0)])
    _, offset = project_collar_to_transect(0.0, 10.0, transect)
    assert offset == 10.0

    projected = project_boreholes(collars, lithologies, transect)
    assert projected.iloc[0]["x_profile"] == 0.0


def test_project_many_matches_single_project(axis_transect: Transect) -> None:
    geometry = _TransectGeometry.from_transect(axis_transect)
    eastings = np.array([0.0, 25.0, 100.0, 100.0])
    northings = np.array([0.0, 0.0, 0.0, 20.0])
    x_profiles, offsets = geometry.project_many(eastings, northings)
    for index, (easting, northing) in enumerate(zip(eastings, northings)):
        x_single, offset_single = geometry.project(float(easting), float(northing))
        assert x_profiles[index] == pytest.approx(x_single)
        assert offsets[index] == pytest.approx(offset_single)
    assert x_profiles.tolist() == pytest.approx([0.0, 25.0, 100.0, 120.0])
    assert offsets.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_transect_length_m_sums_segments() -> None:
    transect = Transect(points=[(0.0, 0.0), (100.0, 0.0), (100.0, 50.0)])
    assert transect_length_m(transect) == pytest.approx(150.0)


def test_select_holes_near_transect_excludes_far_collar() -> None:
    collars = [
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-03", easting=25.0, northing=80.0, elevation=100.0, total_depth=10.0),
    ]
    transect = Transect(points=[(0.0, 0.0), (50.0, 0.0)])
    near = select_holes_near_transect(collars, transect, offset_threshold_m=50.0)
    assert near == ("BH-01", "BH-02")


def test_order_holes_along_transect_sorts_by_profile_distance() -> None:
    collars = [
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
    ]
    transect = Transect(points=[(0.0, 0.0), (50.0, 0.0)])
    ordered = order_holes_along_transect(collars, transect, ("BH-02", "BH-01"))
    assert ordered == ("BH-01", "BH-02")


def test_select_and_order_holes_near_transect_single_pass() -> None:
    collars = [
        Collar(hole_id="BH-02", easting=50.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0),
        Collar(hole_id="BH-03", easting=25.0, northing=80.0, elevation=100.0, total_depth=10.0),
    ]
    transect = Transect(points=[(0.0, 0.0), (50.0, 0.0)])
    ordered = select_and_order_holes_near_transect(collars, transect, offset_threshold_m=50.0)
    assert ordered == ("BH-01", "BH-02")


def test_minimum_curvature_shifts_deviated_hole_east() -> None:
    collar = Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0)
    readings = [
        DeviationReading(hole_id="BH-01", depth=10.0, inclination_deg=45.0, azimuth_deg=90.0),
    ]
    stations = build_minimum_curvature_survey(collar, readings)
    easting, northing, tvd = interpolate_survey_at_depth(stations, 10.0)
    assert easting > 0.0
    assert northing == pytest.approx(0.0, abs=0.01)
    assert tvd < 10.0


def test_project_boreholes_uses_deviation_survey() -> None:
    collar = Collar(hole_id="BH-01", easting=0.0, northing=0.0, elevation=100.0, total_depth=10.0)
    lithologies = [Lithology(hole_id="BH-01", from_depth=0.0, to_depth=10.0, lithology_code="Sand")]
    readings = [
        DeviationReading(hole_id="BH-01", depth=10.0, inclination_deg=45.0, azimuth_deg=90.0),
    ]
    transect = Transect(points=[(0.0, 0.0), (100.0, 0.0)])
    vertical = project_boreholes([collar], lithologies, transect)
    deviated = project_boreholes([collar], lithologies, transect, deviation_readings=readings)
    assert deviated.iloc[0]["x_profile"] > vertical.iloc[0]["x_profile"]
    assert deviated.iloc[0]["bottom_elevation"] > vertical.iloc[0]["bottom_elevation"]


def test_hole_exactly_on_the_offset_threshold_is_kept_on_a_rotated_transect() -> None:
    """A 37° transect gives an offset of 50.00000000000019 m for a hole that
    is exactly 50 m away; the threshold compare must tolerate that."""
    import math

    from projection import select_and_order_holes_near_transect

    angle = math.radians(37.0)
    direction = (math.cos(angle), math.sin(angle))
    normal = (-direction[1], direction[0])
    transect = Transect(points=[(0.0, 0.0), (200.0 * direction[0], 200.0 * direction[1])])
    collars = [
        Collar(
            hole_id="ON-LINE",
            easting=20.0 * direction[0],
            northing=20.0 * direction[1],
            elevation=100.0,
            total_depth=5.0,
        ),
        Collar(
            hole_id="ON-LIMIT",
            easting=100.0 * direction[0] + 50.0 * normal[0],
            northing=100.0 * direction[1] + 50.0 * normal[1],
            elevation=100.0,
            total_depth=5.0,
        ),
    ]
    assert select_and_order_holes_near_transect(collars, transect, 50.0) == ("ON-LINE", "ON-LIMIT")
    # Clearly past the limit is still excluded.
    assert select_and_order_holes_near_transect(collars, transect, 49.9) == ()


# --- review F6: duplicate measured depths in a deviation survey --------------------


def _end_of_hole(collar: Collar, readings: list[DeviationReading]) -> tuple[float, float, float]:
    return interpolate_survey_at_depth(build_minimum_curvature_survey(collar, readings), collar.total_depth)


def test_duplicate_survey_depth_is_order_independent_and_warned(caplog) -> None:
    collar = Collar(hole_id="DUP", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0)
    shallow = DeviationReading(hole_id="DUP", depth=10.0, inclination_deg=10.0, azimuth_deg=0.0)
    steep = DeviationReading(hole_id="DUP", depth=10.0, inclination_deg=80.0, azimuth_deg=0.0)
    with caplog.at_level("WARNING", logger="projection"):
        forward = _end_of_hole(collar, [shallow, steep])
        reverse = _end_of_hole(collar, [steep, shallow])
    assert forward == pytest.approx(reverse, abs=1e-12)
    # Documented winner: the reading closest to vertical (10°), as if it were alone.
    assert forward == pytest.approx(_end_of_hole(collar, [shallow]), abs=1e-12)
    _, northing, tvd = forward
    assert northing == pytest.approx(2.6069, abs=1e-4)
    assert tvd == pytest.approx(19.7974, abs=1e-4)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2 and all("'DUP'" in m and "10.00 m" in m for m in messages)


def test_identical_duplicate_survey_rows_collapse_silently(caplog) -> None:
    collar = Collar(hole_id="DUP", easting=0.0, northing=0.0, elevation=100.0, total_depth=20.0)
    rows = [
        DeviationReading(hole_id="DUP", depth=10.0, inclination_deg=30.0, azimuth_deg=0.0),
        DeviationReading(hole_id="DUP", depth=10.0, inclination_deg=30.0, azimuth_deg=360.0),
        # Vertical shots at one depth agree whatever the azimuth says.
        DeviationReading(hole_id="DUP", depth=15.0, inclination_deg=0.0, azimuth_deg=90.0),
        DeviationReading(hole_id="DUP", depth=15.0, inclination_deg=0.0, azimuth_deg=270.0),
    ]
    with caplog.at_level("WARNING", logger="projection"):
        stations = build_minimum_curvature_survey(collar, rows)
    assert [station.measured_depth for station in stations] == [0.0, 10.0, 15.0, 20.0]
    assert caplog.records == []


# --- review F13: minimum curvature at a 180° dogleg --------------------------------


def test_horizontal_180_degree_turn_adds_no_vertical_depth(caplog) -> None:
    collar = Collar(
        hole_id="H", easting=0.0, northing=0.0, elevation=500.0, total_depth=10.0,
        inclination_deg=90.0, azimuth_deg=0.0,
    )
    readings = [
        DeviationReading(hole_id="H", depth=0.0, inclination_deg=90.0, azimuth_deg=0.0),
        DeviationReading(hole_id="H", depth=10.0, inclination_deg=90.0, azimuth_deg=180.0),
    ]
    lithologies = [
        Lithology(hole_id="H", from_depth=0.0, to_depth=5.0, lithology_code="Sand"),
        Lithology(hole_id="H", from_depth=5.0, to_depth=10.0, lithology_code="Clay"),
    ]
    with caplog.at_level("WARNING", logger="projection"):
        projected = project_boreholes(
            [collar], lithologies, Transect(points=[(-50.0, 0.0), (50.0, 0.0)]), deviation_readings=readings
        )
        _, _, tvd = interpolate_survey_at_depth(build_minimum_curvature_survey(collar, readings), 10.0)
    assert tvd == pytest.approx(0.0, abs=1e-9)  # was 6.366 m
    assert projected["top_elevation"].tolist() == pytest.approx([500.0, 500.0], abs=1e-9)
    assert projected["bottom_elevation"].tolist() == pytest.approx([500.0, 500.0], abs=1e-9)  # was 498.1 / 493.6
    assert any("turns 180°" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("azimuth_deg", [180.0 - 1e-7, 180.0 + 1e-7])
def test_near_180_degree_horizontal_turn_stays_finite_and_level(azimuth_deg: float) -> None:
    """acos rounded these doglegs to exactly π and threw the hole ~52 000 km sideways.

    δ = 1e-7° (≈1.7e-9 rad) is still a defined arc: a level semicircle whose
    chord is ΔMD·2/π, bending east for 180° − δ and west for 180° + δ.
    """
    collar = Collar(hole_id="H", easting=0.0, northing=0.0, elevation=500.0, total_depth=10.0)
    readings = [
        DeviationReading(hole_id="H", depth=0.0, inclination_deg=90.0, azimuth_deg=0.0),
        DeviationReading(hole_id="H", depth=10.0, inclination_deg=90.0, azimuth_deg=azimuth_deg),
    ]
    easting, northing, tvd = interpolate_survey_at_depth(build_minimum_curvature_survey(collar, readings), 10.0)
    assert np.hypot(easting, northing) == pytest.approx(10.0 * 2.0 / np.pi, abs=1e-6)
    assert np.sign(easting) == (1.0 if azimuth_deg < 180.0 else -1.0)
    assert abs(northing) < 1e-6
    assert tvd == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("delta_deg", [1e-4, 1e-5, 1e-6])
def test_near_180_degree_turn_keeps_semicircle_chord(delta_deg: float) -> None:
    """Regression: a 1e-6 rad reversal band collapsed 179.99999° (δ ≈ 1.7e-7 rad)
    to a point while 179.9999° gave the semicircle; the chord is now continuous."""
    collar = Collar(hole_id="H", easting=0.0, northing=0.0, elevation=500.0, total_depth=100.0)
    readings = [
        DeviationReading(hole_id="H", depth=0.0, inclination_deg=90.0, azimuth_deg=0.0),
        DeviationReading(hole_id="H", depth=100.0, inclination_deg=90.0, azimuth_deg=180.0 - delta_deg),
    ]
    stations = build_minimum_curvature_survey(collar, readings)
    easting, northing, tvd = interpolate_survey_at_depth(stations, 100.0)
    beta = np.pi - np.radians(delta_deg)
    chord = 100.0 * np.sin(beta / 2) / (beta / 2)
    assert np.hypot(easting, northing) == pytest.approx(chord, abs=1e-9)
    assert easting == pytest.approx(100.0 * 2.0 / np.pi, abs=1e-4)  # 63.662 m semicircle
    assert northing == pytest.approx(0.0, abs=1e-3)
    assert tvd == pytest.approx(0.0, abs=1e-6)
    # The densified arc stays a level (near) half circle of radius ΔMD/β about E = radius.
    radius = 100.0 / beta
    for station in densify_survey(stations):
        assert np.hypot(station.easting - radius, station.northing) == pytest.approx(radius, abs=1e-3)
        assert station.true_vertical_depth == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("azimuth_deg", [180.0, 180.0 - 1e-12, 180.0 + 1e-12])
def test_reversal_within_rounding_of_180_stays_finite_and_level(azimuth_deg: float) -> None:
    """Within ~1e-10 rad of 180° the arc's plane is rounding noise: no offset."""
    collar = Collar(hole_id="H", easting=0.0, northing=0.0, elevation=500.0, total_depth=10.0)
    readings = [
        DeviationReading(hole_id="H", depth=0.0, inclination_deg=90.0, azimuth_deg=0.0),
        DeviationReading(hole_id="H", depth=10.0, inclination_deg=90.0, azimuth_deg=azimuth_deg),
    ]
    stations = build_minimum_curvature_survey(collar, readings)
    easting, northing, tvd = interpolate_survey_at_depth(stations, 10.0)
    assert (easting, northing, tvd) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert all(np.isfinite(s.true_vertical_depth) for s in densify_survey(stations))


def test_near_180_degree_horizontal_turn_is_a_level_semicircle() -> None:
    """Just short of a reversal the arc is a level, nearly half circle bending east.

    Dogleg β = 180° − δ; chord = ΔMD·sin(β/2)/(β/2) along t1 + t2, which sits δ/2
    north of due east.
    """
    collar = Collar(hole_id="H", easting=0.0, northing=0.0, elevation=500.0, total_depth=10.0)
    readings = [
        DeviationReading(hole_id="H", depth=0.0, inclination_deg=90.0, azimuth_deg=0.0),
        DeviationReading(hole_id="H", depth=10.0, inclination_deg=90.0, azimuth_deg=179.99),
    ]
    easting, northing, tvd = interpolate_survey_at_depth(build_minimum_curvature_survey(collar, readings), 10.0)
    delta = np.radians(0.01)
    beta = np.pi - delta
    chord = 10.0 * np.sin(beta / 2) / (beta / 2)
    assert easting == pytest.approx(chord * np.cos(delta / 2), abs=1e-6)
    assert northing == pytest.approx(chord * np.sin(delta / 2), abs=1e-6)
    assert tvd == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("start", "end", "md", "expected"),
    [
        # 0 → 30° inclination due north over 30 m: RF = tan(15°)/(15° in rad).
        ((0.0, 0.0), (30.0, 0.0), 30.0, (0.0, 7.676178925, 28.647889757)),
        # Azimuth wrap 350° → 10° at 30° inclination over 30 m (dogleg ≈ 9.97°).
        ((30.0, 350.0), (30.0, 10.0), 30.0, (0.0, 14.809442270, 26.046409936)),
    ],
)
def test_minimum_curvature_hand_checks(start, end, md, expected) -> None:
    collar = Collar(hole_id="M", easting=0.0, northing=0.0, elevation=100.0, total_depth=md)
    readings = [
        DeviationReading(hole_id="M", depth=0.0, inclination_deg=start[0], azimuth_deg=start[1]),
        DeviationReading(hole_id="M", depth=md, inclination_deg=end[0], azimuth_deg=end[1]),
    ]
    easting, northing, tvd = interpolate_survey_at_depth(build_minimum_curvature_survey(collar, readings), md)
    assert (easting, northing, tvd) == pytest.approx(expected, abs=1e-8)
