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
