"""Generate a self-contained flood-outcome report from saved model artifacts.

The reporter deliberately consumes the CSV/JSON files written by the simulation
harness instead of importing a solver. This keeps reporting independent from the
numerical model implementation.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TimeSeries:
    stations_m: np.ndarray
    times_min: np.ndarray
    depth_m: np.ndarray


def _finite_array(values, name):
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def load_time_series(path):
    """Load and validate the simulation harness depth time series."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 2 or len(rows[0]) < 3:
        raise ValueError("Time-series CSV needs a header and at least two spatial cells")
    if rows[0][0].strip() not in {"t", "t_min", "time", "time_min"}:
        raise ValueError("First time-series column must identify time in minutes")

    try:
        stations = _finite_array([float(value) for value in rows[0][1:]], "stations")
        times = _finite_array([float(row[0]) for row in rows[1:]], "times")
        depth = _finite_array(
            [[float(value) for value in row[1:]] for row in rows[1:]],
            "depth",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Time-series CSV contains a non-numeric value: {exc}") from exc

    expected_columns = len(rows[0])
    if any(len(row) != expected_columns for row in rows[1:]):
        raise ValueError("Every time-series row must have one depth per station")
    if np.any(np.diff(stations) <= 0):
        raise ValueError("Time-series stations must be strictly increasing")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Time-series times must be strictly increasing")
    if np.any(times < 0):
        raise ValueError("Time-series times must be non-negative")
    if np.any(depth < 0):
        raise ValueError("Time-series depth values must be non-negative")
    if depth.shape != (len(times), len(stations)):
        raise ValueError("Time-series depth shape does not match its coordinates")

    return TimeSeries(stations_m=stations, times_min=times, depth_m=depth)


def load_summary(path):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Run summary must contain a JSON object")
    return data


def _float_field(row, name):
    value = row.get(name)
    if value in (None, ""):
        raise ValueError(f"Geometry CSV is missing {name}")
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Geometry field {name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"Geometry field {name} must be finite")
    return result


def load_bankfull_depths(path, stations_m):
    """Interpolate a reviewed bankfull-depth profile onto model stations."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Geometry CSV is empty")

    geometry_stations = _finite_array(
        [_float_field(row, "station_m") for row in rows],
        "geometry stations",
    )
    bankfull = _finite_array(
        [_float_field(row, "bankfull_depth_m") for row in rows],
        "bankfull depths",
    )
    if np.any(np.diff(geometry_stations) <= 0):
        raise ValueError("Geometry stations must be strictly increasing")
    if np.any(bankfull <= 0):
        raise ValueError("Bankfull depths must be positive")
    if geometry_stations[0] > stations_m[0] or geometry_stations[-1] < stations_m[-1]:
        raise ValueError("Geometry stations must cover the full simulation reach")
    return np.interp(stations_m, geometry_stations, bankfull)


def cell_widths(stations_m):
    edges = np.empty(len(stations_m) + 1, dtype=float)
    edges[1:-1] = 0.5 * (stations_m[:-1] + stations_m[1:])
    edges[0] = stations_m[0] - 0.5 * (stations_m[1] - stations_m[0])
    edges[-1] = stations_m[-1] + 0.5 * (stations_m[-1] - stations_m[-2])
    return np.diff(edges)


def _optional_number(summary, key):
    value = summary.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def calculate_outcomes(series, summary=None, threshold_m=None, threshold_source=None):
    """Calculate model-neutral depth and reach-exceedance outcomes."""
    summary = summary or {}
    peak_flat_index = int(np.argmax(series.depth_m))
    peak_time_index, peak_station_index = np.unravel_index(
        peak_flat_index, series.depth_m.shape
    )
    peak_by_station = np.max(series.depth_m, axis=0)
    depth_change = series.depth_m - series.depth_m[0]

    metrics = {
        "max_depth_m": float(series.depth_m[peak_time_index, peak_station_index]),
        "peak_time_min": float(series.times_min[peak_time_index]),
        "peak_station_m": float(series.stations_m[peak_station_index]),
        "max_depth_change_m": float(np.max(depth_change)),
        "duration_min": float(series.times_min[-1] - series.times_min[0]),
        "reach_length_m": float(np.sum(cell_widths(series.stations_m))),
        "station_count": int(len(series.stations_m)),
        "snapshot_count": int(len(series.times_min)),
        "peak_depth_by_station_m": peak_by_station.tolist(),
    }
    warnings = [
        "Screening output only: a 1-D depth result does not determine a 2-D inundation boundary."
    ]

    threshold = None if threshold_m is None else _finite_array(threshold_m, "threshold")
    if threshold is not None:
        if threshold.ndim == 0:
            threshold = np.full(len(series.stations_m), float(threshold))
        if threshold.shape != series.stations_m.shape:
            raise ValueError("Depth threshold must have one value per station")
        if np.any(threshold < 0):
            raise ValueError("Depth thresholds must be non-negative")

        exceedance = series.depth_m - threshold[np.newaxis, :]
        exceeded = exceedance > 0
        widths = cell_widths(series.stations_m)
        affected_length = exceeded @ widths
        affected_fraction = affected_length / float(np.sum(widths))
        first_indices = np.flatnonzero(np.any(exceeded, axis=1))
        max_length_index = int(np.argmax(affected_length))
        metrics.update(
            {
                "threshold": {
                    "source": threshold_source or "uniform_depth_threshold",
                    "depth_m_by_station": threshold.tolist(),
                },
                "max_exceedance_depth_m": float(max(0.0, np.max(exceedance))),
                "max_exceedance_length_m": float(affected_length[max_length_index]),
                "max_exceedance_fraction": float(affected_fraction[max_length_index]),
                "max_exceedance_time_min": float(series.times_min[max_length_index]),
                "first_exceedance_time_min": (
                    float(series.times_min[first_indices[0]]) if len(first_indices) else None
                ),
            }
        )
    else:
        metrics["threshold"] = None
        warnings.append(
            "No bankfull geometry or explicit depth threshold was supplied; "
            "threshold-exceedance outcomes were not calculated."
        )

    mass_balance = {
        key: _optional_number(summary, key)
        for key in (
            "mass_inflow",
            "mass_source",
            "mass_outflow",
            "mass_balance_error",
        )
    }
    if all(value is None for value in mass_balance.values()):
        warnings.append("The run summary did not provide mass-balance diagnostics.")

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "metrics": metrics,
        "mass_balance": mass_balance,
        "warnings": warnings,
    }


def _json_for_script(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


def _format_metric(value, unit="", digits=3):
    if value is None:
        return "Not assessed"
    return f"{value:,.{digits}f}{unit}"


def render_report(series, outcome, summary, title):
    metrics = outcome["metrics"]
    threshold = metrics["threshold"]
    report_data = {
        "stations": series.stations_m.tolist(),
        "times": series.times_min.tolist(),
        "depth": series.depth_m.tolist(),
        "peakByStation": metrics["peak_depth_by_station_m"],
        "threshold": None if threshold is None else threshold["depth_m_by_station"],
    }

    threshold_card = (
        _format_metric(metrics.get("max_exceedance_length_m"), " m", 1)
        if threshold is not None
        else "Not assessed"
    )
    first_exceedance = _format_metric(
        metrics.get("first_exceedance_time_min"), " min", 1
    )
    warning_items = "".join(
        f"<li>{html.escape(message)}</li>" for message in outcome["warnings"]
    )
    mass_rows = "".join(
        "<tr><th scope=\"row\">{}</th><td>{}</td></tr>".format(
            html.escape(key.replace("_", " ").title()),
            html.escape(_format_metric(value, " m²", 4)),
        )
        for key, value in outcome["mass_balance"].items()
    )
    diagnostic_keys = {
        "mass_inflow",
        "mass_source",
        "mass_outflow",
        "mass_balance_error",
    }
    run_rows = "".join(
        "<tr><th scope=\"row\">{}</th><td>{}</td></tr>".format(
            html.escape(str(key).replace("_", " ").title()),
            html.escape(str(value)),
        )
        for key, value in summary.items()
        if key not in diagnostic_keys
        and (isinstance(value, (str, int, float, bool)) or value is None)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  color-scheme: light dark;
  --bg: #f5f7f8; --panel: #ffffff; --text: #13212a; --muted: #596a73;
  --border: #d8e0e4; --water: #087da8; --water-soft: #a9dbea;
  --threshold: #c24d2c; --warning: #fff4d6; --warning-text: #5d4700;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #10181d; --panel: #172329; --text: #eef5f7; --muted: #afc0c7;
    --border: #34474f; --water: #58c5e8; --water-soft: #214f61;
    --threshold: #ff987a; --warning: #3b3217; --warning-text: #ffe8a3;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--text);
  font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}}
main {{ width: min(1120px, calc(100% - 32px)); margin: 0 auto; padding: 32px 0 56px; }}
h1 {{ margin: 0 0 6px; font-size: clamp(1.7rem, 4vw, 2.5rem); font-weight: 650; }}
h2 {{ margin: 0 0 16px; font-size: 1.15rem; }}
p {{ margin: 0; }}
.subtitle {{ color: var(--muted); margin-bottom: 24px; }}
.metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }}
.metric, section {{
  background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
}}
.metric {{ padding: 16px; }}
.metric span {{ display: block; color: var(--muted); font-size: .82rem; }}
.metric strong {{ display: block; margin-top: 4px; font-size: 1.35rem; font-weight: 650; }}
section {{ padding: 20px; margin-top: 16px; }}
.controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }}
label {{ font-weight: 600; }}
input[type="range"] {{ flex: 1 1 320px; accent-color: var(--water); }}
#time-value {{ min-width: 95px; color: var(--muted); }}
svg {{ width: 100%; height: auto; display: block; overflow: visible; }}
.axis {{ stroke: var(--muted); stroke-width: 1; }}
.grid {{ stroke: var(--border); stroke-width: 1; }}
.profile {{ fill: none; stroke: var(--water); stroke-width: 3; }}
.envelope {{ fill: var(--water-soft); opacity: .55; }}
.threshold {{ fill: none; stroke: var(--threshold); stroke-width: 2; stroke-dasharray: 7 5; }}
.chart-label {{ fill: var(--muted); font-size: 12px; }}
.legend {{ display: flex; gap: 18px; flex-wrap: wrap; color: var(--muted); font-size: .85rem; }}
.swatch {{ display: inline-block; width: 22px; height: 3px; margin: 0 6px 3px 0; background: var(--water); }}
.swatch.threshold {{ background: var(--threshold); }}
.warning {{ background: var(--warning); color: var(--warning-text); border-color: transparent; }}
.warning ul {{ margin: 0; padding-left: 20px; }}
.tables {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
th, td {{ padding: 8px 0; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }}
th {{ color: var(--muted); font-weight: 500; padding-right: 16px; }}
footer {{ margin-top: 18px; color: var(--muted); font-size: .82rem; }}
@media (max-width: 720px) {{
  .metrics, .tables {{ grid-template-columns: 1fr; }}
  main {{ width: min(100% - 20px, 1120px); padding-top: 20px; }}
  section {{ padding: 14px; }}
}}
</style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <p class="subtitle">Model-output screening summary · depths in meters · time in minutes</p>

  <div class="metrics" aria-label="Key outcomes">
    <div class="metric"><span>Maximum modeled depth</span><strong>{_format_metric(metrics["max_depth_m"], " m")}</strong></div>
    <div class="metric"><span>Peak location and time</span><strong>{_format_metric(metrics["peak_station_m"], " m", 1)} · {_format_metric(metrics["peak_time_min"], " min", 1)}</strong></div>
    <div class="metric"><span>Maximum threshold-exceeding reach</span><strong>{threshold_card}</strong></div>
  </div>

  <section>
    <h2>Depth through time</h2>
    <div class="controls">
      <label for="time-slider">Snapshot</label>
      <input id="time-slider" type="range" min="0" max="{len(series.times_min) - 1}" value="0" step="1">
      <output id="time-value" for="time-slider">{series.times_min[0]:.1f} min</output>
    </div>
    <svg id="profile-chart" viewBox="0 0 960 390" role="img" aria-labelledby="chart-title chart-desc">
      <title id="chart-title">Modeled water depth along the reach</title>
      <desc id="chart-desc">Use the snapshot slider to compare water depth by river station over time.</desc>
    </svg>
    <div class="legend">
      <span><i class="swatch"></i>Selected snapshot</span>
      <span>Shaded area: maximum modeled depth</span>
      {"<span><i class=\"swatch threshold\"></i>Depth threshold</span>" if threshold is not None else ""}
    </div>
  </section>

  <section class="warning" aria-labelledby="limitations-title">
    <h2 id="limitations-title">Interpretation limits</h2>
    <ul>{warning_items}</ul>
  </section>

  <section>
    <h2>Diagnostics and run context</h2>
    <div class="tables">
      <table><caption>Mass balance</caption><tbody>{mass_rows}</tbody></table>
      <table><caption>Run metadata</caption><tbody>{run_rows or '<tr><td>No run summary supplied.</td></tr>'}</tbody></table>
    </div>
  </section>

  <footer>
    First threshold exceedance: {first_exceedance}. Report schema version {REPORT_SCHEMA_VERSION}.
  </footer>
</main>
<script>
const report = {_json_for_script(report_data)};
const svg = document.getElementById("profile-chart");
const slider = document.getElementById("time-slider");
const timeValue = document.getElementById("time-value");
const NS = "http://www.w3.org/2000/svg";
const chartW = 960, chartH = 390;
const leftMargin = 72, rightMargin = 24, topMargin = 24, bottomMargin = 58;
const plotW = chartW - leftMargin - rightMargin;
const plotH = chartH - topMargin - bottomMargin;
const maxStation = report.stations[report.stations.length - 1];
const minStation = report.stations[0];
const thresholdMax = report.threshold ? Math.max(...report.threshold) : 0;
const maxDepth = Math.max(1e-9, ...report.peakByStation, thresholdMax) * 1.08;
const x = value => leftMargin + (value - minStation) / (maxStation - minStation) * plotW;
const y = value => topMargin + plotH - value / maxDepth * plotH;
const add = (name, attrs, parent = svg) => {{
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  parent.appendChild(node);
  return node;
}};
const path = values => values.map((value, index) =>
  `${{index ? "L" : "M"}} ${{x(report.stations[index]).toFixed(2)}} ${{y(value).toFixed(2)}}`
).join(" ");

for (let i = 0; i <= 4; i++) {{
  const value = maxDepth * i / 4;
  add("line", {{x1:leftMargin, x2:chartW-rightMargin, y1:y(value), y2:y(value), class:"grid"}});
  const label = add("text", {{x:leftMargin-10, y:y(value)+4, "text-anchor":"end", class:"chart-label"}});
  label.textContent = value.toFixed(maxDepth < 0.1 ? 3 : 2);
}}
add("line", {{x1:leftMargin, x2:chartW-rightMargin, y1:topMargin+plotH, y2:topMargin+plotH, class:"axis"}});
add("line", {{x1:leftMargin, x2:leftMargin, y1:topMargin, y2:topMargin+plotH, class:"axis"}});
for (const station of [minStation, (minStation + maxStation) / 2, maxStation]) {{
  const stationLabel = add("text", {{
    x:x(station), y:topMargin+plotH+22, "text-anchor":"middle", class:"chart-label"
  }});
  stationLabel.textContent = station.toLocaleString(undefined, {{maximumFractionDigits:0}});
}}
const envelopeD = path(report.peakByStation) +
  ` L ${{x(maxStation)}} ${{y(0)}} L ${{x(minStation)}} ${{y(0)}} Z`;
add("path", {{d:envelopeD, class:"envelope"}});
if (report.threshold) add("path", {{d:path(report.threshold), class:"threshold"}});
const profile = add("path", {{d:path(report.depth[0]), class:"profile"}});
const xLabel = add("text", {{x:leftMargin+plotW/2, y:chartH-12, "text-anchor":"middle", class:"chart-label"}});
xLabel.textContent = "River station (m)";
const yLabel = add("text", {{
  x:18, y:topMargin+plotH/2, transform:`rotate(-90 18 ${{topMargin+plotH/2}})`,
  "text-anchor":"middle", class:"chart-label"
}});
yLabel.textContent = "Water depth (m)";
const update = () => {{
  const index = Number(slider.value);
  profile.setAttribute("d", path(report.depth[index]));
  timeValue.textContent = `${{report.times[index].toFixed(1)}} min`;
}};
slider.addEventListener("input", update);
</script>
</body>
</html>
"""


def _default_summary_path(timeseries_path):
    stem = timeseries_path.stem
    if stem.endswith("_timeseries"):
        candidate = timeseries_path.with_name(stem[: -len("_timeseries")] + "_summary.json")
        if candidate.exists():
            return candidate
    return None


def generate_report(
    timeseries_path,
    *,
    summary_path=None,
    geometry_path=None,
    depth_threshold_m=None,
    output_path=None,
    outcome_path=None,
    title=None,
):
    timeseries_path = Path(timeseries_path)
    series = load_time_series(timeseries_path)
    summary_path = Path(summary_path) if summary_path else _default_summary_path(timeseries_path)
    summary = load_summary(summary_path)

    if geometry_path is not None and depth_threshold_m is not None:
        raise ValueError("Choose bankfull geometry or a uniform depth threshold, not both")
    if geometry_path is not None:
        threshold = load_bankfull_depths(geometry_path, series.stations_m)
        threshold_source = "bankfull_depth_profile"
    elif depth_threshold_m is not None:
        if not math.isfinite(depth_threshold_m) or depth_threshold_m < 0:
            raise ValueError("Uniform depth threshold must be a finite non-negative value")
        threshold = float(depth_threshold_m)
        threshold_source = "uniform_depth_threshold"
    else:
        threshold = None
        threshold_source = None

    outcome = calculate_outcomes(series, summary, threshold, threshold_source)
    outcome["sources"] = {
        "timeseries": str(timeseries_path),
        "summary": None if summary_path is None else str(summary_path),
        "geometry": None if geometry_path is None else str(geometry_path),
    }

    report_title = title or f"{timeseries_path.stem.replace('_', ' ').title()} flood outcomes"
    output_path = Path(output_path) if output_path else timeseries_path.with_name(
        timeseries_path.stem.removesuffix("_timeseries") + "_report.html"
    )
    outcome_path = Path(outcome_path) if outcome_path else output_path.with_suffix(
        ".outcomes.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outcome_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_report(series, outcome, summary, report_title),
        encoding="utf-8",
    )
    outcome_path.write_text(
        json.dumps(outcome, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path, outcome_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a model-neutral flood-outcome report from a saved depth time series."
    )
    parser.add_argument("timeseries", type=Path, help="Simulation depth time-series CSV")
    parser.add_argument("--summary", type=Path, help="Run summary JSON; auto-detected when adjacent")
    threshold = parser.add_mutually_exclusive_group()
    threshold.add_argument(
        "--geometry",
        type=Path,
        help="CSV with station_m and bankfull_depth_m covering the reach",
    )
    threshold.add_argument(
        "--depth-threshold",
        type=float,
        help="Uniform reporting threshold in meters",
    )
    parser.add_argument("--output", type=Path, help="Output report HTML")
    parser.add_argument("--outcomes", type=Path, help="Output machine-readable outcomes JSON")
    parser.add_argument("--title", help="Human-readable report title")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        report_path, outcome_path = generate_report(
            args.timeseries,
            summary_path=args.summary,
            geometry_path=args.geometry,
            depth_threshold_m=args.depth_threshold,
            output_path=args.output,
            outcome_path=args.outcomes,
            title=args.title,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"Flood report: {report_path}")
    print(f"Outcome data: {outcome_path}")


if __name__ == "__main__":
    main()
