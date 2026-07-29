"""Hydraulic properties for tabulated compound cross-sections.

The table describes top width as a piecewise-linear function of water depth.
It is a compact solver representation of a reviewed cross-section survey:
storage and pressure are integrated exactly within every table interval.
Wetted perimeter may either be supplied from an asymmetric survey polyline or
inferred by sharing each width change equally between the two banks. Above the
highest reviewed depth, vertical walls are used rather than silently
extrapolating bank geometry.
"""

from __future__ import annotations

import numpy as np


def validate_table(
    depth_m,
    top_width_m,
    *,
    cell_count=None,
    wetted_perimeter_m=None,
):
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
    perimeter = None
    if wetted_perimeter_m is not None:
        perimeter = np.asarray(wetted_perimeter_m, dtype=float)
        if perimeter.shape != width.shape:
            raise ValueError(
                "cross_section_wetted_perimeter_m must match top-width shape"
            )
        if (
            not np.all(np.isfinite(perimeter))
            or np.any(perimeter <= 0.0)
            or np.any(np.diff(perimeter, axis=1) < 0.0)
            or np.any(perimeter + 1e-12 < width)
        ):
            raise ValueError(
                "surveyed wetted perimeter must be finite, positive, "
                "non-decreasing, and at least the top width"
            )
    return depth, width, perimeter


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


def properties(
    depth_m,
    top_width_m,
    water_depth_m,
    wetted_perimeter_m=None,
):
    """Return area, top width, first hydrostatic moment, and perimeter.

    ``top_width_m`` may have any leading dimensions; ``water_depth_m`` is
    broadcast over those dimensions. The hydrostatic moment is
    ``integral_0^h (h-y) width(y) dy`` and therefore has units m³.
    """
    levels, widths, water_depth = _broadcast(
        depth_m, top_width_m, water_depth_m
    )
    perimeter_table = None
    if wetted_perimeter_m is not None:
        _, perimeter_table, _ = _broadcast(
            depth_m, wetted_perimeter_m, water_depth_m
        )
    if np.any(~np.isfinite(water_depth)) or np.any(water_depth < 0.0):
        raise ValueError("water depth must be finite and non-negative")

    area = np.zeros_like(water_depth)
    first_vertical_moment = np.zeros_like(water_depth)
    wetted_perimeter = (
        widths[..., 0].copy()
        if perimeter_table is None
        else perimeter_table[..., 0].copy()
    )
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
        if perimeter_table is None:
            wetted_perimeter += (
                2.0
                * np.sqrt(1.0 + (0.5 * bank_expansion) ** 2)
                * increment
            )
        else:
            perimeter_slope = (
                perimeter_table[..., index + 1]
                - perimeter_table[..., index]
            ) / interval
            wetted_perimeter += perimeter_slope * increment
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


def hydraulic_radius(
    depth_m,
    top_width_m,
    water_depth_m,
    wetted_perimeter_m=None,
):
    cross_section_area, _, _, perimeter = properties(
        depth_m,
        top_width_m,
        water_depth_m,
        wetted_perimeter_m,
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


def survey_properties_at_depth(offset_m, elevation_m, water_depth_m):
    """Return top width and wetted perimeter from one survey polyline.

    Elevation is internally shifted to the local minimum. A positive-width
    horizontal bottom is required. Horizontal segments above the bottom create
    discontinuous top width and are rejected by this continuous table model.
    """
    offset = np.asarray(offset_m, dtype=float)
    elevation = np.asarray(elevation_m, dtype=float)
    if (
        offset.ndim != 1
        or elevation.shape != offset.shape
        or len(offset) < 3
        or np.any(~np.isfinite(offset))
        or np.any(~np.isfinite(elevation))
        or np.any(np.diff(offset) <= 0.0)
    ):
        raise ValueError(
            "survey offsets must be strictly increasing and contain at least "
            "three finite offset/elevation points"
        )
    relative = elevation - np.min(elevation)
    bottom_points = np.flatnonzero(np.isclose(relative, 0.0, atol=1e-12))
    if (
        len(bottom_points) < 2
        or np.any(np.diff(bottom_points) != 1)
        or np.any(np.diff(relative[: bottom_points[0] + 1]) > 1e-12)
        or np.any(np.diff(relative[bottom_points[-1] :]) < -1e-12)
    ):
        raise ValueError(
            "surveyed sections must descend monotonically to one connected "
            "flat bottom and ascend monotonically afterward"
        )
    segment_dx = np.diff(offset)
    low = np.minimum(relative[:-1], relative[1:])
    high = np.maximum(relative[:-1], relative[1:])
    horizontal = np.isclose(low, high, rtol=0.0, atol=1e-12)
    bottom_horizontal = horizontal & np.isclose(low, 0.0, atol=1e-12)
    if not np.any(bottom_horizontal):
        raise ValueError(
            "surveyed sections require a positive-width horizontal bottom"
        )
    if np.any(horizontal & ~bottom_horizontal):
        raise ValueError(
            "horizontal survey benches above the bottom are discontinuous; "
            "reduce them to a reviewed stage-width curve instead"
        )
    stage = np.asarray(water_depth_m, dtype=float)
    if np.any(~np.isfinite(stage)) or np.any(stage < 0.0):
        raise ValueError("survey evaluation depth must be finite and non-negative")

    target_shape = stage.shape + (len(segment_dx),)
    stage_column = np.broadcast_to(stage[..., None], target_shape)
    fraction = np.zeros(target_shape, dtype=float)
    sloped = ~horizontal
    if np.any(sloped):
        fraction[..., sloped] = np.clip(
            (
                stage_column[..., sloped]
                - low[sloped]
            )
            / (high[sloped] - low[sloped]),
            0.0,
            1.0,
        )
    fraction[..., bottom_horizontal] = 1.0
    top_width_m = np.sum(fraction * segment_dx, axis=-1)
    segment_length = np.sqrt(segment_dx**2 + np.diff(relative) ** 2)
    wetted_perimeter_m = np.sum(fraction * segment_length, axis=-1)

    above = np.maximum(stage - np.max(relative), 0.0)
    top_width_m = np.where(
        stage > np.max(relative), offset[-1] - offset[0], top_width_m
    )
    wetted_perimeter_m += 2.0 * above
    return top_width_m, wetted_perimeter_m


def survey_table(offset_m, elevation_m, depth_levels_m):
    """Reduce one asymmetric survey polyline onto common depth levels."""
    levels = np.asarray(depth_levels_m, dtype=float)
    width, perimeter = survey_properties_at_depth(
        offset_m, elevation_m, levels
    )
    validate_table(
        levels,
        width[None, :],
        cell_count=1,
        wetted_perimeter_m=perimeter[None, :],
    )
    return width, perimeter
