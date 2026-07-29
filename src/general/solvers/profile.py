import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from general.solvers.contract import Domain, Domain2D
from general.solvers import cross_section as tabulated_section

MIN_DEPTH = 0.0


@dataclass(frozen=True)
class RiverProfile:
    """Cell-centered river profile inputs for the 1D kinematic wave model."""

    station_m: np.ndarray
    dx_m: np.ndarray
    slope: np.ndarray
    manning_n: np.ndarray
    initial_depth_m: np.ndarray | None = None
    rainfall_rate_m_per_min: np.ndarray | None = None
    labels: tuple[str, ...] = ()

    @property
    def length_m(self):
        return float(np.sum(self.dx_m))


def _as_float(row, key, *, required=True, default=None):
    value = row.get(key, default)
    if value in (None, ""):
        if required:
            raise ValueError(f"Missing required profile field: {key}")
        return default
    return float(value)


def _cell_widths_from_stations(station_m):
    station_m = np.asarray(station_m, dtype=float)
    if station_m.ndim != 1 or len(station_m) == 0:
        raise ValueError("station_m must contain at least one station")
    if len(station_m) == 1:
        raise ValueError("At least two station_m values are required to infer cell widths")
    if np.any(np.diff(station_m) <= 0):
        raise ValueError("station_m values must be strictly increasing")

    edges = np.empty(len(station_m) + 1, dtype=float)
    edges[1:-1] = 0.5 * (station_m[:-1] + station_m[1:])
    edges[0] = station_m[0] - 0.5 * (station_m[1] - station_m[0])
    edges[-1] = station_m[-1] + 0.5 * (station_m[-1] - station_m[-2])
    return np.diff(edges)


def _optional_array(values, expected_len, name, *, minimum=None):
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if len(arr) != expected_len:
        raise ValueError(f"{name} must have one value per station")
    if minimum is not None and np.any(arr < minimum):
        raise ValueError(f"{name} values must be >= {minimum}")
    return arr


def make_profile(station_m, slope, manning_n, initial_depth_m=None, rainfall_rate_m_per_min=None, labels=None):
    station_m = np.asarray(station_m, dtype=float)
    slope = np.asarray(slope, dtype=float)
    manning_n = np.asarray(manning_n, dtype=float)

    if not (len(station_m) == len(slope) == len(manning_n)):
        raise ValueError("station_m, slope, and manning_n must have the same length")
    if np.any(slope <= 0):
        raise ValueError("slope values must be positive")
    if np.any(manning_n <= 0):
        raise ValueError("manning_n values must be positive")

    initial = _optional_array(initial_depth_m, len(station_m), "initial_depth_m", minimum=0.0)
    if initial is not None:
        initial = np.maximum(initial, 0.0)

    rainfall = _optional_array(rainfall_rate_m_per_min, len(station_m), "rainfall_rate_m_per_min", minimum=0.0)

    if labels is None:
        labels = tuple("" for _ in station_m)
    else:
        labels = tuple(labels)
        if len(labels) != len(station_m):
            raise ValueError("labels must have one value per station")

    return RiverProfile(
        station_m=station_m,
        dx_m=_cell_widths_from_stations(station_m),
        slope=slope,
        manning_n=manning_n,
        initial_depth_m=initial,
        rainfall_rate_m_per_min=rainfall,
        labels=labels,
    )


def load_profile_csv(path):
    """Load a river profile CSV.

    Required columns: station_m, slope, manning_n.
    Optional columns: initial_depth_m, rainfall_rate_m_per_min, label.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("River profile CSV is empty")

    initial_values = [row.get("initial_depth_m", "") for row in rows]
    has_initial = any(value not in (None, "") for value in initial_values)
    rainfall_values = [row.get("rainfall_rate_m_per_min", "") for row in rows]
    has_rainfall = any(value not in (None, "") for value in rainfall_values)

    return make_profile(
        station_m=[_as_float(row, "station_m") for row in rows],
        slope=[_as_float(row, "slope") for row in rows],
        manning_n=[_as_float(row, "manning_n") for row in rows],
        initial_depth_m=[_as_float(row, "initial_depth_m", required=False, default=MIN_DEPTH) for row in rows]
        if has_initial
        else None,
        rainfall_rate_m_per_min=[_as_float(row, "rainfall_rate_m_per_min", required=False, default=0.0) for row in rows]
        if has_rainfall
        else None,
        labels=[row.get("label", "") for row in rows],
    )


def load_profile_json(path):
    """Load a river profile JSON file.

    Accepted forms are either a list of segment objects or an object with a
    ``segments`` list. Segment fields match the CSV columns.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("segments", data) if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("River profile JSON must contain a non-empty segment list")

    has_initial = any(row.get("initial_depth_m") is not None for row in rows)
    has_rainfall = any(row.get("rainfall_rate_m_per_min") is not None for row in rows)
    return make_profile(
        station_m=[_as_float(row, "station_m") for row in rows],
        slope=[_as_float(row, "slope") for row in rows],
        manning_n=[_as_float(row, "manning_n") for row in rows],
        initial_depth_m=[_as_float(row, "initial_depth_m", required=False, default=MIN_DEPTH) for row in rows]
        if has_initial
        else None,
        rainfall_rate_m_per_min=[_as_float(row, "rainfall_rate_m_per_min", required=False, default=0.0) for row in rows]
        if has_rainfall
        else None,
        labels=[str(row.get("label", "")) for row in rows],
    )


def load_profile(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_profile_csv(path)
    if suffix == ".json":
        return load_profile_json(path)
    raise ValueError(f"Unsupported river profile format: {suffix}")


def resample_profile(profile: RiverProfile, cells: int) -> RiverProfile:
    """Interpolate reviewed profile fields onto a derived numerical grid.

    The source observations are not modified or relabeled. Existing labels are
    retained only where a derived station coincides with a reviewed station.
    Total modeled reach length is preserved in the finite-volume cell widths.
    """
    if not isinstance(cells, (int, np.integer)) or cells < 2:
        raise ValueError("cells must be an integer of at least 2")
    if len(profile.station_m) < 2:
        raise ValueError("A profile needs at least two stations to be resampled")

    station_m = np.linspace(
        float(profile.station_m[0]),
        float(profile.station_m[-1]),
        int(cells),
    )

    def interpolate(values):
        if values is None:
            return None
        return np.interp(station_m, profile.station_m, values)

    labels = []
    for station in station_m:
        matches = np.flatnonzero(
            np.isclose(profile.station_m, station, rtol=0.0, atol=1e-9)
        )
        labels.append(
            profile.labels[matches[0]]
            if len(matches) and len(profile.labels) == len(profile.station_m)
            else ""
        )

    return RiverProfile(
        station_m=station_m,
        dx_m=np.full(int(cells), profile.length_m / int(cells)),
        slope=interpolate(profile.slope),
        manning_n=interpolate(profile.manning_n),
        initial_depth_m=interpolate(profile.initial_depth_m),
        rainfall_rate_m_per_min=interpolate(
            profile.rainfall_rate_m_per_min
        ),
        labels=tuple(labels),
    )


def domain_from_profile(
    profile: RiverProfile,
    *,
    channel_width_m=None,
    bankfull_depth_m=None,
    channel_bottom_width_m=None,
    side_slope_h_to_v=None,
    cross_section_depth_m=None,
    cross_section_top_width_m=None,
) -> Domain:
    """Build a Domain from a RiverProfile (uses per-cell slope and Manning n)."""
    if (channel_width_m is None) != (bankfull_depth_m is None):
        raise ValueError(
            "channel_width_m and bankfull_depth_m must be supplied together"
        )
    width = (
        None
        if channel_width_m is None
        else np.asarray(channel_width_m, dtype=float)
    )
    bankfull = (
        None
        if bankfull_depth_m is None
        else np.asarray(bankfull_depth_m, dtype=float)
    )
    if (channel_bottom_width_m is None) != (side_slope_h_to_v is None):
        raise ValueError(
            "channel_bottom_width_m and side_slope_h_to_v must be supplied together"
        )
    if channel_bottom_width_m is not None and width is None:
        raise ValueError("Trapezoidal geometry requires channel_width_m")
    if (cross_section_depth_m is None) != (
        cross_section_top_width_m is None
    ):
        raise ValueError(
            "cross_section_depth_m and cross_section_top_width_m must be "
            "supplied together"
        )
    if cross_section_depth_m is not None and (
        channel_bottom_width_m is not None or side_slope_h_to_v is not None
    ):
        raise ValueError(
            "Tabulated compound geometry cannot be combined with a trapezoid"
        )
    section_depth = None
    section_width = None
    if cross_section_depth_m is not None:
        section_depth, section_width = tabulated_section.validate_table(
            cross_section_depth_m,
            cross_section_top_width_m,
            cell_count=len(profile.station_m),
        )
    bottom_width = (
        None
        if channel_bottom_width_m is None
        else np.asarray(channel_bottom_width_m, dtype=float)
    )
    side_slope = (
        None
        if side_slope_h_to_v is None
        else np.asarray(side_slope_h_to_v, dtype=float)
    )
    for values, name in (
        (width, "channel_width_m"),
        (bankfull, "bankfull_depth_m"),
        (bottom_width, "channel_bottom_width_m"),
    ):
        if values is not None and (
            values.shape != profile.station_m.shape
            or np.any(~np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError(f"{name} must contain one finite positive value per station")
    if side_slope is not None and (
        side_slope.shape != profile.station_m.shape
        or np.any(~np.isfinite(side_slope))
        or np.any(side_slope < 0)
    ):
        raise ValueError(
            "side_slope_h_to_v must contain one finite non-negative value per station"
        )
    if bottom_width is not None and np.any(bottom_width > width):
        raise ValueError("channel_bottom_width_m cannot exceed channel_width_m")
    return Domain(
        x_m=profile.station_m,
        dx_m=profile.dx_m,
        slope=profile.slope,
        manning_n=profile.manning_n,
        channel_width_m=width,
        bankfull_depth_m=bankfull,
        channel_bottom_width_m=bottom_width,
        side_slope_h_to_v=side_slope,
        cross_section_depth_m=section_depth,
        cross_section_top_width_m=section_width,
    )


def load_channel_geometry(path, station_m):
    """Interpolate reviewed channel width and bankfull depth onto model stations."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Channel geometry CSV is empty")
    rows.sort(key=lambda row: float(row["station_m"]))
    geometry_stations = np.asarray([float(row["station_m"]) for row in rows])
    widths = np.asarray([float(row["width_m"]) for row in rows])
    bankfull = np.asarray([float(row["bankfull_depth_m"]) for row in rows])
    if np.any(~np.isfinite(geometry_stations)) or np.any(np.diff(geometry_stations) <= 0):
        raise ValueError("Geometry stations must be finite and strictly increasing")
    if np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("Channel widths must be finite and positive")
    if np.any(~np.isfinite(bankfull)) or np.any(bankfull <= 0):
        raise ValueError("Bankfull depths must be finite and positive")
    stations = np.asarray(station_m, dtype=float)
    return (
        np.interp(stations, geometry_stations, widths),
        np.interp(stations, geometry_stations, bankfull),
    )


def load_compound_cross_sections(path, station_m):
    """Interpolate reviewed stage-width curves onto model stations.

    The CSV contains ``station_m,depth_m,top_width_m``. Every surveyed station
    must provide the same strictly increasing depth levels beginning at zero.
    Width is interpolated longitudinally at each depth level.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Compound cross-section CSV is empty")
    grouped = {}
    try:
        for row in rows:
            station = float(row["station_m"])
            depth = float(row["depth_m"])
            width = float(row["top_width_m"])
            grouped.setdefault(station, []).append((depth, width))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Compound cross-section CSV must contain numeric station_m, "
            "depth_m, and top_width_m columns"
        ) from exc

    survey_stations = np.asarray(sorted(grouped), dtype=float)
    if (
        len(survey_stations) < 2
        or np.any(~np.isfinite(survey_stations))
        or np.any(np.diff(survey_stations) <= 0.0)
    ):
        raise ValueError(
            "Compound cross-sections require at least two finite, distinct "
            "survey stations"
        )

    common_depth = None
    survey_widths = []
    for station in survey_stations:
        samples = sorted(grouped[float(station)])
        depths = np.asarray([sample[0] for sample in samples], dtype=float)
        widths = np.asarray([sample[1] for sample in samples], dtype=float)
        if len(np.unique(depths)) != len(depths):
            raise ValueError(
                f"Compound section at station {station:g} has duplicate depths"
            )
        if common_depth is None:
            common_depth = depths
        elif not np.array_equal(depths, common_depth):
            raise ValueError(
                "Every compound section must use identical depth levels"
            )
        survey_widths.append(widths)

    survey_widths = np.asarray(survey_widths, dtype=float)
    tabulated_section.validate_table(
        common_depth, survey_widths, cell_count=len(survey_stations)
    )
    target_stations = np.asarray(station_m, dtype=float)
    interpolated_widths = np.column_stack(
        [
            np.interp(
                target_stations, survey_stations, survey_widths[:, level]
            )
            for level in range(len(common_depth))
        ]
    )
    return common_depth, interpolated_widths


def domain2d_from_profile(
    profile: RiverProfile,
    width_m: float,
    cross_cells: int,
    *,
    channel_width_m=None,
    bankfull_depth_m=None,
    floodplain_slope: float = 0.02,
) -> Domain2D:
    """Build a 2-D channel/floodplain domain from a longitudinal profile.

    Longitudinal slope and roughness are repeated across the channel. The
    legacy default is flat across ``width_m``. Supplying ``channel_width_m``
    and ``bankfull_depth_m`` creates a centred parabolic channel whose banks
    meet the bankfull elevation, with planar floodplain rising laterally at
    ``floodplain_slope`` beyond the reviewed channel width.
    """
    if not np.isfinite(width_m) or width_m <= 0:
        raise ValueError("width_m must be finite and positive")
    if cross_cells < 1:
        raise ValueError("cross_cells must be at least 1")
    if not np.isfinite(floodplain_slope) or floodplain_slope <= 0:
        raise ValueError("floodplain_slope must be finite and positive")

    dy = float(width_m) / int(cross_cells)
    y_m = np.linspace(0.5 * dy, float(width_m) - 0.5 * dy, int(cross_cells))
    shape = (len(profile.station_m), int(cross_cells))
    if (channel_width_m is None) != (bankfull_depth_m is None):
        raise ValueError(
            "channel_width_m and bankfull_depth_m must be supplied together"
        )
    if channel_width_m is None:
        channel_width = np.full(len(profile.station_m), float(width_m))
        bankfull_depth = np.zeros(len(profile.station_m))
    else:
        channel_width = np.asarray(channel_width_m, dtype=float)
        bankfull_depth = np.asarray(bankfull_depth_m, dtype=float)
        if channel_width.ndim == 0:
            channel_width = np.full(len(profile.station_m), float(channel_width))
        if bankfull_depth.ndim == 0:
            bankfull_depth = np.full(
                len(profile.station_m), float(bankfull_depth)
            )
        if (
            channel_width.shape != profile.station_m.shape
            or np.any(~np.isfinite(channel_width))
            or np.any(channel_width <= 0)
            or np.any(channel_width >= width_m)
        ):
            raise ValueError(
                "channel_width_m must contain one positive width smaller than the 2-D domain"
            )
        if (
            bankfull_depth.shape != profile.station_m.shape
            or np.any(~np.isfinite(bankfull_depth))
            or np.any(bankfull_depth <= 0)
        ):
            raise ValueError(
                "bankfull_depth_m must contain one finite positive depth per station"
            )

    bed_profile = np.zeros(len(profile.station_m), dtype=float)
    if len(bed_profile) > 1:
        station_spacing = np.diff(profile.station_m)
        face_slope = 0.5 * (profile.slope[:-1] + profile.slope[1:])
        bed_profile[1:] = -np.cumsum(face_slope * station_spacing)
    offset = np.abs(y_m[None, :] - 0.5 * float(width_m))
    half_channel_width = 0.5 * channel_width[:, None]
    channel_fraction = np.minimum(offset / half_channel_width, 1.0)
    outside_distance = np.maximum(offset - half_channel_width, 0.0)
    lateral_bed = (
        bankfull_depth[:, None] * channel_fraction**2
        + floodplain_slope * outside_distance
    )
    bed = bed_profile[:, None] + lateral_bed
    slope_y = np.zeros(shape)
    if len(y_m) > 1:
        slope_y[:, 1:] = -(bed[:, 1:] - bed[:, :-1]) / np.diff(y_m)[None, :]
        slope_y[:, 0] = slope_y[:, 1]

    return Domain2D(
        x_m=np.asarray(profile.station_m, dtype=float).copy(),
        y_m=y_m,
        dx_m=np.asarray(profile.dx_m, dtype=float).copy(),
        dy_m=np.full(int(cross_cells), dy),
        slope_x=np.broadcast_to(np.asarray(profile.slope)[:, None], shape).copy(),
        slope_y=slope_y,
        manning_n=np.broadcast_to(np.asarray(profile.manning_n)[:, None], shape).copy(),
        bed_elevation_m=bed,
    )
