import numpy as np
import pytest

from general.solvers import cross_section


def test_compound_section_properties_match_piecewise_geometry():
    depth = np.array([0.0, 1.0, 2.0])
    width = np.array([[2.0, 4.0, 10.0]])

    area, top_width, moment, perimeter = cross_section.properties(
        depth, width, np.array([1.5])
    )

    assert area[0] == pytest.approx(5.75)
    assert top_width[0] == pytest.approx(7.0)
    assert moment[0] == pytest.approx(83.0 / 24.0)
    assert perimeter[0] == pytest.approx(
        2.0 + 2.0 * np.sqrt(2.0) + np.sqrt(10.0)
    )


def test_compound_section_area_depth_round_trip_for_many_cells():
    levels = np.array([0.0, 0.5, 1.5, 3.0])
    widths = np.array(
        [
            [2.0, 3.0, 8.0, 12.0],
            [4.0, 4.0, 7.0, 20.0],
        ]
    )
    water_depth = np.array([0.2, 4.0])
    area = cross_section.area(levels, widths, water_depth)
    recovered = cross_section.depth_from_area(levels, widths, area)
    assert recovered == pytest.approx(water_depth)


def test_compound_section_rejects_narrowing_or_misaligned_curves():
    with pytest.raises(ValueError, match="non-decreasing"):
        cross_section.validate_table(
            [0.0, 1.0, 2.0], [[2.0, 5.0, 4.0]]
        )
    with pytest.raises(ValueError, match="one curve per cell"):
        cross_section.validate_table(
            [0.0, 1.0], [[2.0, 3.0]], cell_count=2
        )


def test_asymmetric_survey_retains_exact_polyline_wetted_perimeter():
    offset = np.array([0.0, 2.0, 6.0, 10.0, 13.0])
    elevation = np.array([3.0, 0.0, 0.0, 1.0, 4.0])
    levels = np.array([0.0, 1.0, 3.0, 4.0])
    width, perimeter = cross_section.survey_table(
        offset, elevation, levels
    )

    assert width[0] == pytest.approx(4.0)
    assert width[1] == pytest.approx(26.0 / 3.0)
    assert perimeter[1] == pytest.approx(
        4.0 + np.sqrt(13.0) / 3.0 + np.sqrt(17.0)
    )
    surveyed_radius = cross_section.hydraulic_radius(
        levels,
        width[None, :],
        np.array([1.0]),
        perimeter[None, :],
    )
    inferred_symmetric_radius = cross_section.hydraulic_radius(
        levels, width[None, :], np.array([1.0])
    )
    assert surveyed_radius[0] != pytest.approx(
        inferred_symmetric_radius[0]
    )


def test_raw_survey_rejects_unrepresentable_horizontal_bench():
    with pytest.raises(ValueError, match="horizontal survey benches"):
        cross_section.survey_table(
            [0.0, 2.0, 5.0, 8.0, 10.0],
            [3.0, 0.0, 0.0, 2.0, 2.0],
            [0.0, 1.0, 2.0, 3.0],
        )


def test_raw_survey_rejects_disconnected_or_v_shaped_bottom():
    with pytest.raises(ValueError, match="one connected flat bottom"):
        cross_section.survey_table(
            [0.0, 2.0, 4.0, 6.0, 8.0],
            [3.0, 0.0, 2.0, 0.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
        )
    with pytest.raises(ValueError, match="one connected flat bottom"):
        cross_section.survey_table(
            [0.0, 4.0, 8.0],
            [3.0, 0.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
        )
