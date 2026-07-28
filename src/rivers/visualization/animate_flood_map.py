"""Create an animated topographic flood map from a simulation time series.

The canonical CSV provides one depth per river station (for a 2-D run, the
runner writes the cross-channel mean). This visualization applies that depth
uniformly across each station's estimated wetted cross-section and interpolates
between adjacent stations. It is a screening visualization, not a replacement
for the full 2-D field report or a terrain-resolving inundation model.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivers.ingest.common import haversine_m


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FLOODPLAIN_SLOPE = 0.02
DEFAULT_MAX_WIDTH_M = 5_000.0


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def load_time_series(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the canonical CSV written by ``run_simulation.py``."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("Simulation time-series CSV is empty") from exc
        if len(header) < 3 or header[0] not in {"t", "t_min"}:
            raise ValueError("Expected a t_min column followed by at least two river stations")
        stations = np.asarray([float(value) for value in header[1:]], dtype=float)
        rows = [row for row in reader if row]

    if not rows:
        raise ValueError("Simulation time-series CSV contains no frames")
    times = np.asarray([float(row[0]) for row in rows], dtype=float)
    depths = np.asarray([[float(value) for value in row[1:]] for row in rows], dtype=float)
    if depths.shape != (len(times), len(stations)):
        raise ValueError("Every time-series row must contain one depth per station")
    if np.any(~np.isfinite(stations)) or np.any(np.diff(stations) <= 0):
        raise ValueError("Simulation station coordinates must be finite and strictly increasing")
    if np.any(~np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError("Simulation times must be finite and non-decreasing")
    if np.any(~np.isfinite(depths)) or np.any(depths < 0):
        raise ValueError("Simulation depths must be finite and non-negative")
    return stations, times, depths


def load_markers(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ordered river centerline markers and their cumulative stations."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("At least two ordered centerline markers are required")

    coordinates = np.asarray(
        [[float(row["lat"]), float(row["lon"])] for row in rows], dtype=float
    )
    if np.any(~np.isfinite(coordinates)):
        raise ValueError("Marker coordinates must be finite")

    explicit = [row.get("station_m", "") for row in rows]
    if all(value not in (None, "") for value in explicit):
        stations = np.asarray([float(value) for value in explicit], dtype=float)
    else:
        stations = np.zeros(len(rows), dtype=float)
        for index in range(1, len(rows)):
            stations[index] = stations[index - 1] + haversine_m(
                coordinates[index - 1, 0],
                coordinates[index - 1, 1],
                coordinates[index, 0],
                coordinates[index, 1],
            )
    if np.any(~np.isfinite(stations)) or np.any(np.diff(stations) <= 0):
        raise ValueError("Marker stations must be finite and strictly increasing")
    return stations, coordinates


def load_geometry(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load channel width and bankfull depth samples."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Channel geometry CSV is empty")

    rows.sort(key=lambda row: float(row["station_m"]))
    stations = np.asarray([float(row["station_m"]) for row in rows], dtype=float)
    widths = np.asarray([float(row["width_m"]) for row in rows], dtype=float)
    bankfull = np.asarray([float(row["bankfull_depth_m"]) for row in rows], dtype=float)
    if np.any(~np.isfinite(stations)) or np.any(np.diff(stations) <= 0):
        raise ValueError("Geometry stations must be finite and strictly increasing")
    if np.any(~np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("Channel widths must be positive finite values")
    if np.any(~np.isfinite(bankfull)) or np.any(bankfull <= 0):
        raise ValueError("Bankfull depths must be positive finite values")
    return stations, widths, bankfull


def interpolate_centerline(
    stations: np.ndarray,
    marker_stations: np.ndarray,
    marker_coordinates: np.ndarray,
) -> np.ndarray:
    """Interpolate latitude/longitude onto the solver's station coordinates."""
    tolerance = max(1.0, marker_stations[-1] * 1e-6)
    if stations[0] < marker_stations[0] - tolerance or stations[-1] > marker_stations[-1] + tolerance:
        raise ValueError("Simulation stations extend beyond the supplied centerline markers")
    clipped = np.clip(stations, marker_stations[0], marker_stations[-1])
    return np.column_stack(
        (
            np.interp(clipped, marker_stations, marker_coordinates[:, 0]),
            np.interp(clipped, marker_stations, marker_coordinates[:, 1]),
        )
    )


def mapped_centerline_samples(
    simulation_stations: np.ndarray,
    centerline_stations: np.ndarray,
    centerline_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain mapped bends and insert every solver station into the map path."""
    interpolate_centerline(
        simulation_stations,
        centerline_stations,
        centerline_coordinates,
    )
    interior = centerline_stations[
        (centerline_stations > simulation_stations[0])
        & (centerline_stations < simulation_stations[-1])
    ]
    mapped_stations = np.unique(np.concatenate((simulation_stations, interior)))
    mapped_coordinates = np.column_stack(
        (
            np.interp(mapped_stations, centerline_stations, centerline_coordinates[:, 0]),
            np.interp(mapped_stations, centerline_stations, centerline_coordinates[:, 1]),
        )
    )
    keep = [0]
    for index in range(1, len(mapped_stations)):
        previous = mapped_coordinates[keep[-1]]
        current = mapped_coordinates[index]
        separation_m = haversine_m(previous[0], previous[1], current[0], current[1])
        if separation_m >= 0.01:
            keep.append(index)
    if keep[-1] != len(mapped_stations) - 1:
        keep[-1] = len(mapped_stations) - 1
    return mapped_stations[keep], mapped_coordinates[keep]


def inundation_widths(
    depths_m: np.ndarray,
    channel_widths_m: np.ndarray,
    bankfull_depths_m: np.ndarray,
    floodplain_slope: float,
    max_width_m: float,
) -> np.ndarray:
    """Estimate wetted widths using a symmetric planar floodplain cross-section."""
    overbank_depth = np.maximum(depths_m - bankfull_depths_m, 0.0)
    widths = channel_widths_m + 2.0 * overbank_depth / floodplain_slope
    return np.minimum(widths, max_width_m)


def _cross_section_edges(centerline: np.ndarray, widths_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Offset each station left and right of its local centerline tangent."""
    reference_lat = float(np.mean(centerline[:, 0]))
    reference_lon = float(np.mean(centerline[:, 1]))
    meters_per_lat_degree = 111_132.0
    meters_per_lon_degree = 111_320.0 * math.cos(math.radians(reference_lat))
    xy = np.column_stack(
        (
            (centerline[:, 1] - reference_lon) * meters_per_lon_degree,
            (centerline[:, 0] - reference_lat) * meters_per_lat_degree,
        )
    )

    tangents = np.empty_like(xy)
    tangents[0] = xy[1] - xy[0]
    tangents[-1] = xy[-1] - xy[-2]
    if len(xy) > 2:
        tangents[1:-1] = xy[2:] - xy[:-2]
    norms = np.linalg.norm(tangents, axis=1)
    if np.any(norms == 0):
        raise ValueError("Adjacent mapped simulation stations must not share coordinates")
    tangents /= norms[:, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    offsets = normals * (widths_m[:, None] / 2.0)

    def to_lat_lon(points: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                reference_lat + points[:, 1] / meters_per_lat_degree,
                reference_lon + points[:, 0] / meters_per_lon_degree,
            )
        )

    return to_lat_lon(xy + offsets), to_lat_lon(xy - offsets)


def build_frames(
    stations: np.ndarray,
    times: np.ndarray,
    depths: np.ndarray,
    centerline: np.ndarray,
    channel_widths_m: np.ndarray,
    bankfull_depths_m: np.ndarray,
    floodplain_slope: float,
    max_width_m: float,
) -> list[dict[str, object]]:
    """Build map polygons for every time step and centerline segment."""
    frames = []
    for time_min, frame_depths in zip(times, depths):
        widths = inundation_widths(
            frame_depths,
            channel_widths_m,
            bankfull_depths_m,
            floodplain_slope,
            max_width_m,
        )
        left, right = _cross_section_edges(centerline, widths)
        segments = []
        for index in range(len(stations) - 1):
            depth = float(0.5 * (frame_depths[index] + frame_depths[index + 1]))
            width = float(0.5 * (widths[index] + widths[index + 1]))
            bankfull = float(0.5 * (bankfull_depths_m[index] + bankfull_depths_m[index + 1]))
            segments.append(
                {
                    "coordinates": [
                        left[index].tolist(),
                        left[index + 1].tolist(),
                        right[index + 1].tolist(),
                        right[index].tolist(),
                    ],
                    "depth_m": depth,
                    "width_m": width,
                    "overbank_depth_m": max(depth - bankfull, 0.0),
                    "station_start_m": float(stations[index]),
                    "station_end_m": float(stations[index + 1]),
                }
            )
        frames.append(
            {
                "time_min": float(time_min),
                "max_depth_m": float(np.max(frame_depths)),
                "max_width_m": float(np.max(widths)),
                "segments": segments,
            }
        )
    return frames


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _portable_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _default_summary_path(timeseries_path: Path) -> Path:
    suffix = "_timeseries.csv"
    if timeseries_path.name.endswith(suffix):
        return timeseries_path.with_name(timeseries_path.name[: -len(suffix)] + "_summary.json")
    return timeseries_path.with_suffix(".summary.json")


def _named_map_inputs(river: str) -> tuple[Path, Path]:
    if river == "example":
        directory = REPO_ROOT / "real_world_rivers" / "tools"
        return directory / "example_markers.csv", directory / "example_geometry.csv"
    slug = river.replace("-", "_")
    directory = REPO_ROOT / "real_world_rivers" / "curated"
    return directory / f"{slug}_markers.csv", directory / f"{slug}_geometry.csv"


def resolve_map_inputs(args: argparse.Namespace) -> tuple[Path, Path, dict[str, object]]:
    """Resolve explicit, run-summary, or named-river map inputs in that order."""
    summary_path = args.summary or _default_summary_path(args.timeseries)
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    markers = args.markers
    geometry = args.geometry
    map_inputs = summary.get("map_inputs", {})
    if markers is None and map_inputs.get("markers"):
        markers = _repo_path(map_inputs["markers"])
    if geometry is None and map_inputs.get("geometry"):
        geometry = _repo_path(map_inputs["geometry"])

    river = args.river or summary.get("river")
    if (markers is None or geometry is None) and river:
        named_markers, named_geometry = _named_map_inputs(str(river))
        markers = markers or named_markers
        geometry = geometry or named_geometry

    if markers is None or geometry is None:
        raise ValueError(
            "Could not locate map inputs; provide --river or both --markers and --geometry"
        )
    markers = Path(markers)
    geometry = Path(geometry)
    if not markers.exists():
        raise ValueError(f"Centerline marker file does not exist: {markers}")
    if not geometry.exists():
        raise ValueError(f"Channel geometry file does not exist: {geometry}")
    return markers, geometry, summary


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    html, body, #app { width: 100%; height: 100%; margin: 0; }
    body { background: #edf1f2; color: #172126; }
    #app { display: grid; grid-template-rows: auto minmax(0, 1fr); }
    header { display: grid; grid-template-columns: minmax(180px, 1fr) auto minmax(280px, 1.6fr); gap: 18px; align-items: center; padding: 12px 16px; background: #fff; border-bottom: 1px solid #ccd5d8; z-index: 1000; }
    h1 { margin: 0; font-size: 17px; line-height: 1.2; letter-spacing: 0; }
    .subtitle { margin-top: 4px; color: #58666d; font-size: 12px; }
    .metrics { display: flex; gap: 18px; }
    .metric { min-width: 84px; }
    .metric-label { display: block; color: #65747b; font-size: 10px; text-transform: uppercase; }
    .metric-value { display: block; margin-top: 2px; font-size: 15px; font-variant-numeric: tabular-nums; }
    .timeline { display: grid; grid-template-columns: 40px minmax(120px, 1fr) 90px; gap: 10px; align-items: center; }
    button, select { height: 34px; border: 1px solid #aebbc0; border-radius: 4px; background: #fff; color: #172126; }
    button { width: 40px; cursor: pointer; font-size: 15px; }
    button:hover, select:hover { border-color: #51656e; background: #f3f6f7; }
    input[type="range"] { width: 100%; accent-color: #087f8c; }
    #map { width: 100%; height: 100%; background: #dce4e5; }
    .legend { background: rgba(255, 255, 255, 0.94); border: 1px solid #bdc8cc; border-radius: 4px; padding: 9px 10px; color: #26343a; font-size: 11px; line-height: 1.35; box-shadow: 0 1px 4px rgba(0,0,0,0.14); }
    .legend-ramp { width: 180px; height: 9px; margin: 5px 0 3px; background: linear-gradient(90deg, #4cc9f0, #087f8c, #f6c453, #d1495b); }
    .legend-labels { display: flex; justify-content: space-between; font-variant-numeric: tabular-nums; }
    .method { margin-top: 6px; padding-top: 6px; border-top: 1px solid #d7dfe1; color: #647279; max-width: 180px; }
    @media (max-width: 760px) {
      header { grid-template-columns: 1fr; gap: 10px; padding: 10px 12px; }
      .metrics { justify-content: space-between; }
      .timeline { grid-template-columns: 40px minmax(100px, 1fr) 80px; }
    }
  </style>
</head>
<body>
<div id="app">
  <header>
    <div>
      <h1 id="title">__TITLE__</h1>
      <div class="subtitle" id="subtitle"></div>
    </div>
    <div class="metrics" aria-live="polite">
      <div class="metric"><span class="metric-label">Time</span><span class="metric-value" id="time-value"></span></div>
      <div class="metric"><span class="metric-label">Max depth</span><span class="metric-value" id="depth-value"></span></div>
      <div class="metric"><span class="metric-label">Max width</span><span class="metric-value" id="width-value"></span></div>
    </div>
    <div class="timeline">
      <button id="play" type="button" aria-label="Pause animation" title="Pause">&#10074;&#10074;</button>
      <input id="frame" type="range" min="0" step="1" aria-label="Simulation time">
      <select id="speed" aria-label="Animation speed" title="Animation speed">
        <option value="2">2x</option>
        <option value="1" selected>1x</option>
        <option value="0.5">0.5x</option>
      </select>
    </div>
  </header>
  <main id="map" aria-label="Animated flood extent topographic map"></main>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script id="flood-data" type="application/json">__PAYLOAD__</script>
<script>
  const data = JSON.parse(document.getElementById('flood-data').textContent);
  const map = L.map('map', { preferCanvas: true });
  L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
    maxZoom: 17,
    attribution: 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, SRTM | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (CC-BY-SA)'
  }).addTo(map);

  const centerline = L.polyline(data.centerline, { color: '#172126', weight: 1.5, opacity: 0.75 }).addTo(map);
  map.fitBounds(centerline.getBounds().pad(0.12), { maxZoom: 14 });
  const floodLayer = L.layerGroup().addTo(map);
  const slider = document.getElementById('frame');
  const playButton = document.getElementById('play');
  const speed = document.getElementById('speed');
  slider.max = String(data.frames.length - 1);
  slider.value = '0';

  const palette = ['#4cc9f0', '#087f8c', '#f6c453', '#d1495b'];
  function depthColor(depth) {
    const ratio = Math.max(0, Math.min(0.999, depth / Math.max(data.max_depth_m, 1e-9)));
    return palette[Math.floor(ratio * palette.length)];
  }

  function drawFrame(index) {
    const frame = data.frames[index];
    floodLayer.clearLayers();
    frame.segments.forEach((segment) => {
      const polygon = L.polygon(segment.coordinates, {
        color: depthColor(segment.depth_m),
        fillColor: depthColor(segment.depth_m),
        fillOpacity: 0.68,
        opacity: 0.95,
        weight: 1
      }).addTo(floodLayer);
      polygon.bindTooltip(
        `Depth ${segment.depth_m.toFixed(2)} m<br>Extent width ${segment.width_m.toFixed(0)} m<br>Station ${segment.station_start_m.toFixed(0)}-${segment.station_end_m.toFixed(0)} m`,
        { sticky: true }
      );
    });
    slider.value = String(index);
    document.getElementById('time-value').textContent = `${frame.time_min.toFixed(1)} min`;
    document.getElementById('depth-value').textContent = `${frame.max_depth_m.toFixed(2)} m`;
    document.getElementById('width-value').textContent = `${frame.max_width_m.toFixed(0)} m`;
  }

  let playing = true;
  let timer = null;
  function setPlaying(next) {
    playing = next;
    playButton.innerHTML = playing ? '&#10074;&#10074;' : '&#9654;';
    playButton.setAttribute('aria-label', playing ? 'Pause animation' : 'Play animation');
    playButton.title = playing ? 'Pause' : 'Play';
    window.clearInterval(timer);
    if (playing) {
      timer = window.setInterval(() => {
        const nextFrame = (Number(slider.value) + 1) % data.frames.length;
        drawFrame(nextFrame);
      }, data.frame_interval_ms / Number(speed.value));
    }
  }

  playButton.addEventListener('click', () => setPlaying(!playing));
  slider.addEventListener('input', () => drawFrame(Number(slider.value)));
  speed.addEventListener('change', () => setPlaying(playing));
  document.getElementById('subtitle').textContent = data.subtitle;

  const legend = L.control({ position: 'bottomleft' });
  legend.onAdd = () => {
    const element = L.DomUtil.create('div', 'legend');
    element.innerHTML = `<strong>Water depth</strong><div class="legend-ramp"></div><div class="legend-labels"><span>0 m</span><span>${data.max_depth_m.toFixed(2)} m</span></div><div class="method">Uniform depth across each estimated cross-section</div>`;
    return element;
  };
  legend.addTo(map);
  drawFrame(0);
  setPlaying(true);
</script>
</body>
</html>
"""


def render_html(payload: dict[str, object], output_path: Path) -> Path:
    """Write a portable HTML animation; only basemap tiles require a network."""
    title = html.escape(str(payload["title"]))
    encoded = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    document = HTML_TEMPLATE.replace("__TITLE__", title).replace("__PAYLOAD__", encoded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate a run_simulation depth CSV on an interactive topographic river map."
    )
    parser.add_argument("timeseries", type=Path, help="Timeseries CSV written by run_simulation.py")
    parser.add_argument("--summary", type=Path, help="Paired run summary JSON (auto-detected by default)")
    parser.add_argument("--river", help="Named river when the run summary has no map inputs")
    parser.add_argument("--markers", type=Path, help="Ordered centerline marker CSV for a custom river")
    parser.add_argument("--geometry", type=Path, help="Channel geometry CSV for a custom river")
    parser.add_argument("--output", type=Path, help="Output HTML path")
    parser.add_argument(
        "--floodplain-slope",
        type=_positive_float,
        default=DEFAULT_FLOODPLAIN_SLOPE,
        help="Symmetric lateral rise/run used above bankfull (default: 0.02)",
    )
    parser.add_argument(
        "--max-width-m",
        type=_positive_float,
        default=DEFAULT_MAX_WIDTH_M,
        help="Safety cap for estimated inundation width (default: 5000)",
    )
    parser.add_argument(
        "--frame-interval-ms",
        type=_positive_float,
        default=700.0,
        help="Playback interval for each recorded frame (default: 700)",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    stations, times, depths = load_time_series(args.timeseries)
    marker_path, geometry_path, summary = resolve_map_inputs(args)
    marker_stations, marker_coordinates = load_markers(marker_path)
    geometry_stations, geometry_widths, geometry_bankfull = load_geometry(geometry_path)
    mapped_stations, centerline = mapped_centerline_samples(
        stations, marker_stations, marker_coordinates
    )
    mapped_depths = np.asarray(
        [np.interp(mapped_stations, stations, frame) for frame in depths], dtype=float
    )
    channel_widths = np.interp(mapped_stations, geometry_stations, geometry_widths)
    bankfull_depths = np.interp(mapped_stations, geometry_stations, geometry_bankfull)
    frames = build_frames(
        mapped_stations,
        times,
        mapped_depths,
        centerline,
        channel_widths,
        bankfull_depths,
        args.floodplain_slope,
        args.max_width_m,
    )

    river = str(args.river or summary.get("river") or args.timeseries.stem)
    model = str(summary.get("model", summary.get("solver", "model"))).replace("_", " ")
    event = str(summary.get("event", "simulation")).replace("-", " ")
    payload = {
        "title": f"{river.replace('-', ' ').title()} flood extent",
        "subtitle": f"{model.title()} | {event.title()}",
        "centerline": centerline.tolist(),
        "frames": frames,
        "max_depth_m": float(np.max(mapped_depths)),
        "frame_interval_ms": float(args.frame_interval_ms),
        "method": {
            "cross_section_depth": "uniform",
            "floodplain_slope": float(args.floodplain_slope),
            "max_width_m": float(args.max_width_m),
            "markers": _portable_path(marker_path),
            "geometry": _portable_path(geometry_path),
        },
    }
    output_path = args.output or args.timeseries.with_name(
        args.timeseries.stem.removesuffix("_timeseries") + "_flood_map.html"
    )
    return render_html(payload, output_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        output_path = run(args)
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"Done. Animated flood map: {output_path}")


if __name__ == "__main__":
    main()
