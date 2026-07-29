"""Hydraulic properties for tabulated, symmetric compound cross-sections.

The table describes top width as a piecewise-linear function of water depth.
It is a compact solver representation of a reviewed cross-section survey:
storage and pressure are integrated exactly within every table interval, while
wetted perimeter assumes each width change is shared equally by the two banks.
Above the highest reviewed depth, vertical walls are used rather than silently
extrapolating bank slope.
"""

from __future__ import annotations

import numpy as np


def validate_table(depth_m, top_width_m, *, cell_count=None):
    """Return validated depth levels and per-cell top-width curves."""
    depth = np.asarray(depth_m, dtype=float)
    width = np.asarray(top_width_m, dtype=float)
    if depth.ndim != 1 or depth.size < 2:
        raise ValueError(
            "cross_section_depth_m must contain at least two depth levels"
        )
    if (
        not np.all(np.isfinite(depth))
        or depth[0] != 0.0
        or np.any(np.diff(depth) <= 0.0)
    ):
        raise ValueError(
            "cross_section_depth_m must start at zero and increase strictly"
        )
    if width.ndim != 2 or width.shape[1] != depth.size:
        raise ValueError(
            "cross_section_top_width_m must have shape (cells, depth levels)"
        )
    if cell_count is not None and width.shape[0] != cell_count:
        raise ValueError(
            "cross_section_top_width_m must contain one curve per cell"
        )
    if (
        not np.all(np.isfinite(width))
        or np.any(width <= 0.0)
        or np.any(np.diff(width, axis=1) < 0.0)
    ):
        raise ValueError(
            "cross-section top widths must be finite, positive, and "
            "non-decreasing with depth"
        )
    return depth, width


def _broadcast(depth_m, top_width_m, values):
    depth = np.asarray(depth_m, dtype=float)
    width = np.asarray(top_width_m, dtype=float)
    value = np.asarray(values, dtype=float)
    target_shape = np.broadcast_shapes(value.shape, width.shape[:-1])
    return (
        depth,
        np.broadcast_to(width, target_shape + (width.shape[-1],)),
        np.broadcast_to(value, target_shape),
    )


def properties(depth_m, top_width_m, water_depth_m):
    """Return area, top width, first hydrostatic moment, and perimeter.

    ``top_width_m`` may have any leading dimensions; ``water_depth_m`` is
    broadcast over those dimensions. The hydrostatic moment is
    ``integral_0^h (h-y) width(y) dy`` and therefore has units m³.
    """
    levels, widths, water_depth = _broadcast(
        depth_m, top_width_m, water_depth_m
    )
    if np.any(~np.isfinite(water_depth)) or np.any(water_depth < 0.0):
        raise ValueError("water depth must be finite and non-negative")

    area = np.zeros_like(water_depth)
    first_vertical_moment = np.zeros_like(water_depth)
    wetted_perimeter = widths[..., 0].copy()
    active_width = widths[..., 0].copy()

    for index, interval in enumerate(np.diff(levels)):
        increment = np.clip(water_depth - levels[index], 0.0, interval)
        width_low = widths[..., index]
        bank_expansion = (
            widths[..., index + 1] - width_low
        ) / interval
        area += width_low * increment + 0.5 * bank_expansion * increment**2
        first_vertical_moment += (
            levels[index] * width_low * increment
            + 0.5
            * (levels[index] * bank_expansion + width_low)
            * increment**2
            + bank_expansion * increment**3 / 3.0
        )
        wetted_perimeter += (
            2.0
            * np.sqrt(1.0 + (0.5 * bank_expansion) ** 2)
            * increment
        )
        active_width += bank_expansion * increment

    above = np.maximum(water_depth - levels[-1], 0.0)
    area += widths[..., -1] * above
    first_vertical_moment += widths[..., -1] * (
        levels[-1] * above + 0.5 * above**2
    )
    wetted_perimeter += 2.0 * above
    active_width = np.where(
        water_depth >= levels[-1], widths[..., -1], active_width
    )
    hydrostatic_moment = water_depth * area - first_vertical_moment
    return area, active_width, hydrostatic_moment, wetted_perimeter


def area(depth_m, top_width_m, water_depth_m):
    return properties(depth_m, top_width_m, water_depth_m)[0]


def top_width(depth_m, top_width_m, water_depth_m):
    return properties(depth_m, top_width_m, water_depth_m)[1]


def hydrostatic_moment(depth_m, top_width_m, water_depth_m):
    return properties(depth_m, top_width_m, water_depth_m)[2]


def hydraulic_radius(depth_m, top_width_m, water_depth_m):
    cross_section_area, _, _, perimeter = properties(
        depth_m, top_width_m, water_depth_m
    )
    radius = np.zeros_like(cross_section_area)
    np.divide(
        cross_section_area,
        perimeter,
        out=radius,
        where=perimeter > 0.0,
    )
    return radius


def depth_from_area(depth_m, top_width_m, cross_section_area_m2):
    """Invert the exact piecewise-linear width integral."""
    levels, widths, target_area = _broadcast(
        depth_m, top_width_m, cross_section_area_m2
    )
    if np.any(~np.isfinite(target_area)) or np.any(target_area < 0.0):
        raise ValueError("cross-section area must be finite and non-negative")

    remaining = target_area.copy()
    water_depth = np.zeros_like(target_area)
    for index, interval in enumerate(np.diff(levels)):
        width_low = widths[..., index]
        expansion = (
            widths[..., index + 1] - width_low
        ) / interval
        capacity = width_low * interval + 0.5 * expansion * interval**2
        used_area = np.minimum(remaining, capacity)
        discriminant = np.sqrt(width_low**2 + 2.0 * expansion * used_area)
        sloped_increment = 2.0 * used_area / (width_low + discriminant)
        vertical_increment = used_area / width_low
        increment = np.where(
            expansion > 1e-14, sloped_increment, vertical_increment
        )
        water_depth += increment
        remaining -= used_area

    water_depth += remaining / widths[..., -1]
    return water_depth
