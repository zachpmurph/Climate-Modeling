"""Generate the Flood Explorer website's data artifacts using the real solvers.

Modes:

    python website/build_scenarios.py               # default: validation data only
    python website/build_scenarios.py --references  # parity references for the JS solver ports
    python website/build_scenarios.py --atlas       # the (retired) Regions atlas data

The Regions tab was removed from the site while the model's accuracy is being
established, so the atlas is no longer built by default. Its builder is kept
behind --atlas so Regions can be restored once validation supports it. The
default now writes only data/validation.json; --references writes the small
fixed cases the Node parity test checks against the JS ports.

All quantities are meters and minutes internally (Manning n = SI / 60).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from general.solvers import linear_advection as la  # noqa: E402
from general.solvers import saint_venant_1d as sv1  # noqa: E402
from general.solvers import saint_venant_2d as sv2  # noqa: E402
from general.solvers.profile import load_profile, make_profile  # noqa: E402

WEBSITE_DIR = Path(__file__).resolve().parent
DATA_DIR = WEBSITE_DIR / "data"
SCENARIO_DIR = DATA_DIR / "scenarios"
REFERENCE_DIR = WEBSITE_DIR / "test" / "reference"

MAX_FRAMES = 145
MM_PER_HOUR_TO_M_PER_MIN = 1.0 / 1000.0 / 60.0


# ── regions ────────────────────────────────────────────────────────────────

def _columbia_region():
    raw = load_profile(REPO_ROOT / "data" / "real_world_rivers" / "columbia_hanford_profile.csv")
    # The source profile's first cell carries the ingestion slope floor (1e-6):
    # the dam pool upstream of Priest Rapids is nearly flat in the DEM. A
    # kinematic-wave cell with a near-zero slope cannot convey flow and becomes a
    # bottomless reservoir (tens of meters of ponded depth), which is a model
    # artifact rather than river behavior. For this screening atlas we replace the
    # floored value with the adjacent cell's slope, and say so in the region text.
    slope = raw.slope.copy()
    slope[0] = slope[1]
    profile = make_profile(
        station_m=raw.station_m,
        slope=slope,
        manning_n=raw.manning_n,
        initial_depth_m=raw.initial_depth_m,
        rainfall_rate_m_per_min=raw.rainfall_rate_m_per_min,
        labels=raw.labels,
    )
    return {
        "id": "columbia_hanford",
        "name": "Columbia River — Hanford Reach",
        "kind": "real",
        "solver": "kinematic_wave",
        "description": (
            "A 90 km reach of the Columbia River in Washington State, from Priest "
            "Rapids Dam to Pasco. Built from ingested real-world data: DEM-derived "
            "slopes, USGS gauge context, and reviewed channel roughness. The flat "
            "dam-pool slope in the first cell of the source profile is replaced "
            "with the adjacent cell's slope for kinematic screening."
        ),
        "source": "data/real_world_rivers/columbia_hanford_profile.csv (ingested pipeline output; first-cell slope adjusted)",
        "profile": profile,
        "sim_minutes": 2880.0,   # 48 h
        "rain_scale": 1.0,
    }


def _steep_creek_region():
    n_cells = 61
    stations = np.linspace(0.0, 3000.0, n_cells)
    slope = np.linspace(0.03, 0.015, n_cells)
    manning_si = 0.045
    profile = make_profile(
        station_m=stations,
        slope=slope,
        manning_n=np.full(n_cells, manning_si / 60.0),
        initial_depth_m=np.full(n_cells, 0.15),
    )
    return {
        "id": "steep_creek",
        "name": "Steep Mountain Creek (synthetic)",
        "kind": "synthetic",
        "solver": "saint_venant",
        "description": (
            "A synthetic 3 km mountain creek: steep (3% easing to 1.5%), rough "
            "(SI Manning n 0.045), shallow baseflow. Responds fast and hard to "
            "storms — the flash-flood archetype. Run with the 1-D Saint-Venant "
            "(full dynamic wave) solver."
        ),
        "source": "synthetic reach defined in website/build_scenarios.py",
        "profile": profile,
        "sim_minutes": 360.0,    # 6 h
        "rain_scale": 1.0,
    }


def _lowland_region():
    n_cells = 81
    stations = np.linspace(0.0, 20000.0, n_cells)
    slope = np.linspace(6e-4, 2e-4, n_cells)
    manning_si = 0.03
    profile = make_profile(
        station_m=stations,
        slope=slope,
        manning_n=np.full(n_cells, manning_si / 60.0),
        initial_depth_m=np.full(n_cells, 1.2),
    )
    return {
        "id": "lowland_meander",
        "name": "Lowland Meander (synthetic)",
        "kind": "synthetic",
        "solver": "kinematic_wave",
        "description": (
            "A synthetic 20 km lowland river: gentle slopes (0.06% easing to "
            "0.02%), moderate roughness (SI Manning n 0.03), 1.2 m baseflow depth. "
            "Slow to rise and slow to drain — floods arrive late and linger."
        ),
        "source": "synthetic reach defined in website/build_scenarios.py",
        "profile": profile,
        "sim_minutes": 1440.0,   # 24 h
        "rain_scale": 1.0,
    }


def _floodplain_2d_region():
    """A straight compound-channel reach solved with the verified 2-D solver.

    The cross-section is a main channel flanked by benches that start dry, so
    an event's lateral spread onto the floodplain is the visible result -- the
    thing a 1-D cross-section average cannot show. A rectangular flat-bed
    channel with wall boundaries would produce a y-invariant solution (vertical
    stripes), which is why the bed is explicitly compound.
    """
    nx, ny = 48, 24
    length_m, width_m = 1200.0, 300.0
    slope_x = 0.002
    manning_si = 0.035
    manning_n = manning_si / 60.0
    bank_height_m = 1.5
    half_width_m = 60.0
    transition_m = 60.0
    channel_depth_m = 1.0

    x, dx = sv2._grid(length_m, nx)
    y, dy = sv2._grid(width_m, ny)
    centre = 0.5 * (y[0] + y[-1])
    lateral = bank_height_m * np.clip((np.abs(y - centre) - half_width_m) / transition_m, 0.0, 1.0)
    bed = (-slope_x * x)[:, None] + lateral[None, :]

    # Normal-flow initial state: uniform depth in the channel, dry benches, and
    # a matching unit discharge so the baseline event starts near steady state
    # instead of surging while the reach accelerates from rest.
    surface = (-slope_x * x + channel_depth_m)[:, None]
    h_init = np.maximum(surface - bed, 0.0)
    hu_init = (1.0 / manning_n) * np.power(h_init, 5.0 / 3.0) * np.sqrt(slope_x)

    return {
        "id": "floodplain_2d",
        "name": "Compound Floodplain Reach (synthetic 2-D)",
        "kind": "synthetic",
        "solver": "saint_venant_2d",
        "dimensions": 2,
        "description": (
            "A synthetic 1.2 km reach with a compound cross-section: a main "
            "channel with sloping banks, flanked by floodplain benches standing "
            "1.5 m higher, on a 0.2% slope in a 300 m valley with confining "
            "walls. Solved with the repository's verified 2-D shallow-water "
            "solver, so you can watch water leave the channel and spread "
            "laterally instead of only rising in a cross-section average. Once "
            "the valley fills wall to wall, extra flow shows up as depth rather "
            "than width — which is why the severe storm and the surge flood a "
            "similar area at very different depths."
        ),
        "source": "synthetic compound-channel reach defined in website/build_scenarios.py",
        "grid": {
            "x": x, "y": y, "dx": dx, "dy": dy, "bed": bed,
            "h_init": h_init, "hu_init": hu_init,
            "manning_n": np.full((nx, ny), manning_n),
            "channel_depth_m": channel_depth_m,
            "bank_height_m": bank_height_m,
        },
        "sim_minutes": 180.0,
        # The column-wise normal-flow guess is not an equilibrium of the 2-D
        # equations: lateral momentum exchange and the open outlet settle it
        # into a slightly different steady profile over the first ~60 min. Every
        # event therefore starts from a spun-up steady state, so "baseline"
        # really is flat and each event's change is the event's, not the
        # reach still settling. Convergence was checked by running to 480 min:
        # the profile is unchanged to 3 decimals from t=120 onward.
        "spin_up_min": 180.0,
        "rain_scale": 1.0,
        "frames": 31,
    }


def _baseline_inflow_2d(region):
    """Per-column normal-flow unit discharge at the inlet (m^2/min)."""
    return region["grid"]["hu_init"][0, :].copy()


def _baseline_inflow(profile):
    """Manning-equilibrium inflow at the mid-reach initial depth: keeps the reach
    near steady state under the baseline event."""
    mid = len(profile.station_m) // 2
    depth = 0.01 if profile.initial_depth_m is None else float(profile.initial_depth_m[mid])
    return float(la.q(np.array([depth]), profile.slope[mid], profile.manning_n[mid])[0])


def _events(q0, sim_minutes, rain_scale):
    rain = lambda mm_per_hour: mm_per_hour * MM_PER_HOUR_TO_M_PER_MIN * rain_scale
    third = sim_minutes / 3.0
    return [
        {
            "id": "baseline",
            "name": "Baseline flow",
            "narrative": "Normal conditions: steady upstream inflow, no storm. The reference the other events are compared against.",
            "left_inflow": q0,
            "rain_rate": 0.0,
            "rain_start": 0.0,
            "rain_end": None,
        },
        {
            "id": "moderate_storm",
            "name": "Moderate storm",
            "narrative": "A 10 mm/h storm over the whole reach for a third of the simulation, on top of normal inflow.",
            "left_inflow": q0,
            "rain_rate": rain(10.0),
            "rain_start": 0.0,
            "rain_end": third,
        },
        {
            "id": "severe_storm",
            "name": "Severe storm",
            "narrative": "A 30 mm/h downpour for half the simulation while upstream inflow runs 50% above normal.",
            "left_inflow": 1.5 * q0,
            "rain_rate": rain(30.0),
            "rain_start": 0.0,
            "rain_end": sim_minutes / 2.0,
        },
        {
            "id": "flash_flood",
            "name": "Upstream flood surge",
            "narrative": "No rain, but upstream inflow jumps to four times normal — a sustained release or upstream flood wave moving through.",
            "left_inflow": 4.0 * q0,
            "rain_rate": 0.0,
            "rain_start": 0.0,
            "rain_end": None,
        },
        {
            "id": "prolonged_rain",
            "name": "Prolonged rain",
            "narrative": "A long soaking rain: 8 mm/h for two thirds of the simulation at normal inflow.",
            "left_inflow": q0,
            "rain_rate": rain(8.0),
            "rain_start": 0.0,
            "rain_end": 2.0 * third,
        },
    ]


def _events_2d(q0, sim_minutes, rain_scale):
    """Event set for the 2-D reach.

    Unlike the 1-D regions, direct rain on this reach is not a meaningful
    stressor: 30 mm/h falling on 0.36 km2 is a few m3/min against an inflow of
    ~11,000 m3/min, so a rain-only event would be indistinguishable from
    baseline. A storm reaches a reach like this as *discharge from upstream*,
    so the storm events raise the inlet hydrograph and carry the direct rain as
    well. The multiples are chosen to walk the reach through the states a
    floodplain map exists to show: in-bank, at-bank, spilling, inundated.
    """
    rain = lambda mm_per_hour: mm_per_hour * MM_PER_HOUR_TO_M_PER_MIN * rain_scale
    third = sim_minutes / 3.0
    return [
        {
            "id": "baseline",
            "name": "Baseline flow",
            "narrative": "Normal conditions: steady upstream inflow, no storm. Water stays in the main channel and the floodplain benches are dry.",
            "left_inflow": q0,
            "rain_rate": 0.0,
            "rain_start": 0.0,
            "rain_end": None,
        },
        {
            "id": "moderate_storm",
            "name": "Moderate storm",
            "narrative": "A storm over the upstream catchment lifts inflow 60% above normal, plus 10 mm/h falling directly on the reach. The river runs higher but stays between its banks.",
            "left_inflow": 1.6 * q0,
            "rain_rate": rain(10.0),
            "rain_start": 0.0,
            "rain_end": third,
        },
        {
            "id": "severe_storm",
            "name": "Severe storm",
            "narrative": "A severe catchment storm pushes inflow to 2.6 times normal with 30 mm/h on the reach. The river tops its banks and water spreads onto the floodplain.",
            "left_inflow": 2.6 * q0,
            "rain_rate": rain(30.0),
            "rain_start": 0.0,
            "rain_end": sim_minutes / 2.0,
        },
        {
            "id": "flash_flood",
            "name": "Upstream flood surge",
            "narrative": "No rain here, but inflow jumps to four times normal — an upstream flood wave or dam release arriving. The floodplain goes under across the full reach.",
            "left_inflow": 4.0 * q0,
            "rain_rate": 0.0,
            "rain_start": 0.0,
            "rain_end": None,
        },
        {
            "id": "prolonged_rain",
            "name": "Prolonged rain",
            "narrative": "A long soaking rain: 8 mm/h on the reach for two thirds of the run, with inflow 30% above normal as the catchment slowly responds.",
            "left_inflow": 1.3 * q0,
            "rain_rate": rain(8.0),
            "rain_start": 0.0,
            "rain_end": 2.0 * third,
        },
    ]


def _record_interval(sim_minutes):
    interval = sim_minutes / (MAX_FRAMES - 1)
    # Round up to a tidy number of minutes.
    return max(1.0, math.ceil(interval))


def _run_scenario(region, event):
    profile = region["profile"]
    interval = _record_interval(region["sim_minutes"])
    if region["solver"] == "kinematic_wave":
        result = la.run_model(
            profile,
            t_final_min=region["sim_minutes"],
            left_inflow_flux=event["left_inflow"],
            record_interval_min=interval,
            rainfall_rate_m_per_min=event["rain_rate"],
            rainfall_start_min=event["rain_start"],
            rainfall_end_min=event["rain_end"],
        )
        discharge_history = None
    elif region["solver"] == "saint_venant":
        rate = event["rain_rate"]
        rain_end = event["rain_end"]

        def rainfall(x, t):
            active = rate > 0 and t >= event["rain_start"] and (rain_end is None or t < rain_end)
            return np.full_like(x, rate if active else 0.0)

        raw = sv1.run_model(
            None,
            region["sim_minutes"],
            record_interval=interval,
            h_init=profile.initial_depth_m,
            left_inflow=event["left_inflow"],
            rainfall=rainfall,
            x_m=profile.station_m,
            dx_m=profile.dx_m,
            slope=profile.slope,
            manning_n=profile.manning_n,
        )
        result = {
            "station_m": raw["x"],
            "dx_m": raw["dx_m"],
            "times": raw["times"],
            "depth_history": raw["h_history"],
            "depth_initial": raw["h_initial"],
            "depth_final": raw["h_final"],
            "mass_inflow": raw["mass_inflow"],
            "mass_source": raw["mass_source"],
            "mass_outflow": raw["mass_outflow"],
        }
        discharge_history = raw["q_history"]
    else:
        raise ValueError(f"Unknown solver {region['solver']}")
    return result, discharge_history


def _metrics(result, discharge_history, profile, solver):
    depth = np.asarray(result["depth_history"])
    times = np.asarray(result["times"])
    stations = np.asarray(result["station_m"])
    flat_peak = int(np.argmax(depth))
    ti, si = np.unravel_index(flat_peak, depth.shape)
    if discharge_history is not None:
        downstream_q = np.asarray(discharge_history)[:, -1]
    else:
        downstream_q = la.q(depth[:, -1], profile.slope[-1], profile.manning_n[-1])
    storage_initial = float(np.sum(np.asarray(result["depth_initial"]) * np.asarray(result["dx_m"])))
    storage_final = float(np.sum(np.asarray(result["depth_final"]) * np.asarray(result["dx_m"])))
    balance_error = (storage_final - storage_initial) - (
        result["mass_inflow"] + result["mass_source"] - result["mass_outflow"]
    )
    return {
        "peak_depth_m": float(depth[ti, si]),
        "peak_time_min": float(times[ti]),
        "peak_station_m": float(stations[si]),
        "initial_max_depth_m": float(np.max(result["depth_initial"])),
        "final_max_depth_m": float(np.max(result["depth_final"])),
        "downstream_hydrograph_m2_per_min": [float(v) for v in downstream_q],
        "mass_inflow": float(result["mass_inflow"]),
        "mass_source": float(result["mass_source"]),
        "mass_outflow": float(result["mass_outflow"]),
        "mass_balance_error": float(balance_error),
        "solver": solver,
    }


def _spin_up_2d(region, q0):
    """Settle the reach into steady state under the baseline inflow.

    Returns the converged state, which every event then starts from.
    """
    grid = region["grid"]
    raw = sv2.run_model(
        T_final=region["spin_up_min"],
        record_interval=region["spin_up_min"],
        h_init=grid["h_init"],
        hu_init=grid["hu_init"],
        left_inflow=q0,
        rainfall=lambda x, y, t: 0.0,
        x_m=grid["x"], y_m=grid["y"], dx_m=grid["dx"], dy_m=grid["dy"],
        manning_n=grid["manning_n"],
        bed_elevation_m=grid["bed"],
    )
    return {
        "h": raw["h_final"].copy(),
        "hu": raw["hu_final"].copy(),
        "hv": raw["hv_final"].copy(),
        # Land that is dry at steady state: the only ground an event can flood.
        "dry_mask": raw["h_final"] <= 0.0,
    }


def _run_scenario_2d(region, event, start):
    grid = region["grid"]
    interval = region["sim_minutes"] / (region["frames"] - 1)
    rate = event["rain_rate"]
    rain_start = event["rain_start"]
    rain_end = event["rain_end"]

    def rainfall(x, y, t):
        active = rate > 0 and t >= rain_start and (rain_end is None or t < rain_end)
        return rate if active else 0.0

    return sv2.run_model(
        T_final=region["sim_minutes"],
        record_interval=interval,
        h_init=start["h"],
        hu_init=start["hu"],
        hv_init=start["hv"],
        left_inflow=np.asarray(event["left_inflow"], dtype=float),
        rainfall=rainfall,
        x_m=grid["x"], y_m=grid["y"], dx_m=grid["dx"], dy_m=grid["dy"],
        manning_n=grid["manning_n"],
        bed_elevation_m=grid["bed"],
    )


# A cell counts as flooded once it carries this much water; below it the depth
# is a numerical film rather than something a person would call flooding.
FLOOD_DEPTH_M = 0.05


def _metrics_2d(raw, region, start):
    """Metrics for a 2-D scenario, computed from the full-precision arrays.

    The depth field written to disk is rounded for file size; these numbers are
    not, so the mass-balance figure the site reports as "numerical quality" is
    the solver's, not the rounding's.
    """
    depth = raw["h_history"]
    times = raw["times"]
    area = raw["dx_m"][:, None] * raw["dy_m"][None, :]
    bench = start["dry_mask"]
    bench_area = float(np.sum(area[bench]))

    flat_peak = int(np.argmax(depth))
    ti, xi, yi = np.unravel_index(flat_peak, depth.shape)

    # Downstream discharge through the outlet face: sum of unit discharge times
    # cell width -> m^3/min (a volume rate), not the 1-D m^2/min per unit width.
    downstream_q = np.sum(raw["hu_history"][:, -1, :] * raw["dy_m"][None, :], axis=1)
    flooded = np.array([
        float(np.sum(area[bench & (frame > FLOOD_DEPTH_M)])) for frame in depth
    ])

    storage_initial = float(np.sum(raw["h_initial"] * area))
    storage_final = float(np.sum(raw["h_final"] * area))
    balance_error = (storage_final - storage_initial) - (
        raw["mass_inflow"] + raw["mass_source"] - raw["mass_outflow"]
        + raw["mass_floor_correction"]
    )
    peak_index = int(np.argmax(flooded))
    return {
        "peak_depth_m": float(depth[ti, xi, yi]),
        "peak_time_min": float(times[ti]),
        "peak_station_m": float(raw["x"][xi]),
        "peak_across_m": float(raw["y"][yi]),
        "initial_max_depth_m": float(np.max(raw["h_initial"])),
        "final_max_depth_m": float(np.max(raw["h_final"])),
        "downstream_hydrograph_m3_per_min": [float(v) for v in downstream_q],
        "flooded_area_history_m2": [float(v) for v in flooded],
        "peak_flooded_area_m2": float(flooded.max()),
        "peak_flooded_time_min": float(times[peak_index]),
        "floodplain_area_m2": bench_area,
        "peak_flooded_fraction": float(flooded.max() / bench_area) if bench_area > 0 else 0.0,
        "flood_depth_threshold_m": FLOOD_DEPTH_M,
        "mass_inflow": float(raw["mass_inflow"]),
        "mass_source": float(raw["mass_source"]),
        "mass_outflow": float(raw["mass_outflow"]),
        "mass_floor_correction": float(raw["mass_floor_correction"]),
        "mass_balance_error": float(balance_error),
        "solver": "saint_venant_2d",
    }


def _round_field(values, digits=4):
    return [round(float(v), digits) for v in np.asarray(values).ravel()]


def _validate(depth_history):
    depth = np.asarray(depth_history)
    if not np.all(np.isfinite(depth)):
        raise ValueError("scenario produced non-finite depths")
    if np.any(depth < 0):
        raise ValueError("scenario produced negative depths")
    if depth.shape[0] > MAX_FRAMES + 1:
        raise ValueError(f"scenario produced {depth.shape[0]} frames (budget {MAX_FRAMES})")


CALIBRATION_LIMITATION = (
    "Roughness and geometry are literature/ingested estimates; the model is "
    "verified numerically but not calibrated to observed floods."
)
LIMITATIONS_1D = [
    "Screening output from a 1-D model: depths are cross-section averages, not a 2-D inundation boundary.",
    CALIBRATION_LIMITATION,
]
LIMITATIONS_2D = [
    "The inundation map is a 2-D solution on an idealised straight compound channel, not a survey of a real floodplain: it shows how water spreads over this geometry, not where a particular town would flood.",
    "The bed is a smooth prismatic cross-section — no levees, bridges, buildings, culverts, or infiltration.",
    CALIBRATION_LIMITATION,
]


def _entry_common(region, extra):
    entry = {
        "id": region["id"],
        "name": region["name"],
        "kind": region["kind"],
        "solver": region["solver"],
        "dimensions": region.get("dimensions", 1),
        "description": region["description"],
        "source": region["source"],
        "events": [],
    }
    entry.update(extra)
    return entry


def _event_payload(event, region):
    return {
        "id": event["id"],
        "name": event["name"],
        "narrative": event["narrative"],
        "rain_rate_m_per_min": event["rain_rate"],
        "rain_start_min": event["rain_start"],
        "rain_end_min": event["rain_end"],
        "sim_minutes": region["sim_minutes"],
    }


def _build_region_1d(region, index):
    profile = region["profile"]
    q0 = _baseline_inflow(profile)
    entry = _entry_common(region, {
        "length_m": float(np.sum(profile.dx_m)),
        "cells": int(len(profile.station_m)),
        "baseline_inflow_m2_per_min": q0,
    })
    for event in _events(q0, region["sim_minutes"], region["rain_scale"]):
        result, discharge_history = _run_scenario(region, event)
        _validate(result["depth_history"])
        metrics = _metrics(result, discharge_history, profile, region["solver"])
        payload = {
            "region": {k: entry[k] for k in
                       ("id", "name", "kind", "solver", "dimensions", "description", "source")},
            "event": {**_event_payload(event, region),
                      "left_inflow_m2_per_min": event["left_inflow"]},
            "station_m": [float(v) for v in result["station_m"]],
            "times_min": [float(v) for v in result["times"]],
            "depth_history": [[float(v) for v in row] for row in result["depth_history"]],
            "metrics": metrics,
            "limitations": LIMITATIONS_1D,
        }
        _emit(region, event, entry, payload)
    index["regions"].append(entry)


def _build_region_2d(region, index):
    grid = region["grid"]
    q0 = _baseline_inflow_2d(region)
    total_q0 = float(np.sum(q0 * grid["dy"]))
    start = _spin_up_2d(region, q0)
    print(f"  {region['id']}: spun up {region['spin_up_min']:.0f} min -> "
          f"max depth {start['h'].max():.3f} m, "
          f"{int(start['dry_mask'].sum())} of {start['h'].size} cells dry")
    entry = _entry_common(region, {
        "length_m": float(np.sum(grid["dx"])),
        "width_m": float(np.sum(grid["dy"])),
        "cells": int(grid["h_init"].size),
        "nx": int(len(grid["x"])),
        "ny": int(len(grid["y"])),
        "baseline_inflow_m3_per_min": total_q0,
    })
    for event in _events_2d(q0, region["sim_minutes"], region["rain_scale"]):
        raw = _run_scenario_2d(region, event, start)
        _validate(raw["h_history"])
        metrics = _metrics_2d(raw, region, start)
        payload = {
            "region": {k: entry[k] for k in
                       ("id", "name", "kind", "solver", "dimensions", "description", "source")},
            "event": {**_event_payload(event, region),
                      "left_inflow_m3_per_min": float(np.sum(np.asarray(event["left_inflow"]) * grid["dy"])),
                      "inflow_multiple_of_baseline": float(
                          np.sum(np.asarray(event["left_inflow"]) * grid["dy"]) / total_q0)},
            "nx": entry["nx"],
            "ny": entry["ny"],
            "x_m": [float(v) for v in raw["x"]],
            "y_m": [float(v) for v in raw["y"]],
            "dx_m": [float(v) for v in raw["dx_m"]],
            "dy_m": [float(v) for v in raw["dy_m"]],
            "bed_elevation_m": _round_field(raw["bed_elevation_m"]),
            "bank_height_m": grid["bank_height_m"],
            "times_min": [float(v) for v in raw["times"]],
            # Rounded for file size only; every number in `metrics` above comes
            # from the full-precision arrays.
            "depth_history": [_round_field(frame) for frame in raw["h_history"]],
            "metrics": metrics,
            "limitations": LIMITATIONS_2D,
        }
        _emit(region, event, entry, payload,
              extra=f"flooded {metrics['peak_flooded_area_m2'] / 10000:.1f} ha "
                    f"({metrics['peak_flooded_fraction'] * 100:.0f}% of floodplain)")
    index["regions"].append(entry)


def _emit(region, event, entry, payload, extra=""):
    filename = f"{region['id']}__{event['id']}.json"
    path = SCENARIO_DIR / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    entry["events"].append({
        "id": event["id"],
        "name": event["name"],
        "file": f"data/scenarios/{filename}",
        "peak_depth_m": payload["metrics"]["peak_depth_m"],
    })
    size_kb = path.stat().st_size / 1024
    print(f"  {region['id']} / {event['id']}: peak {payload['metrics']['peak_depth_m']:.3f} m "
          f"at t={payload['metrics']['peak_time_min']:.0f} min "
          f"({len(payload['times_min'])} frames, {size_kb:.0f} KB) {extra}")


def build_atlas():
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    regions = [_columbia_region(), _steep_creek_region(), _lowland_region(),
               _floodplain_2d_region()]
    index = {"generated_by": "website/build_scenarios.py", "regions": []}

    for region in regions:
        if region.get("dimensions", 1) == 2:
            _build_region_2d(region, index)
        else:
            _build_region_1d(region, index)

    (DATA_DIR / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {DATA_DIR / 'index.json'}")


# ── parity references for the JS ports ─────────────────────────────────────

def build_references():
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Kinematic wave, uniform reach with inflow.
    n_cells = 11
    profile = make_profile(
        station_m=np.linspace(0.0, 1000.0, n_cells),
        slope=np.full(n_cells, 0.001),
        manning_n=np.full(n_cells, 0.0006),
    )
    result = la.run_model(profile, t_final_min=30.0, left_inflow_flux=0.05,
                          record_interval_min=5.0, base_depth_m=0.05)
    _write_reference("kinematic_uniform.json", {
        "solver": "kinematic_wave",
        "profile": _profile_dict(profile),
        "options": {"tFinalMin": 30.0, "leftInflowFlux": 0.05,
                    "recordIntervalMin": 5.0, "baseDepthM": 0.05},
        "expected": _kw_expected(result),
    })

    # Kinematic wave, varying profile + profile rain column + windowed uniform rain.
    stations = np.array([0.0, 500.0, 1200.0, 2000.0, 3000.0])
    profile = make_profile(
        station_m=stations,
        slope=np.array([0.0012, 0.001, 0.0008, 0.0007, 0.0006]),
        manning_n=np.array([5.8e-4, 6.3e-4, 6.7e-4, 7.0e-4, 7.5e-4]),
        initial_depth_m=np.full(5, 0.04),
        rainfall_rate_m_per_min=np.array([0.0, 1e-6, 1e-6, 1e-6, 0.0]),
    )
    result = la.run_model(profile, t_final_min=40.0, left_inflow_flux=0.02,
                          record_interval_min=2.0, rainfall_rate_m_per_min=2e-5,
                          rainfall_start_min=5.0, rainfall_end_min=20.0)
    _write_reference("kinematic_varying.json", {
        "solver": "kinematic_wave",
        "profile": _profile_dict(profile),
        "options": {"tFinalMin": 40.0, "leftInflowFlux": 0.02, "recordIntervalMin": 2.0,
                    "rainfallRateMPerMin": 2e-5, "rainfallStartMin": 5.0, "rainfallEndMin": 20.0},
        "expected": _kw_expected(result),
    })

    # Saint-Venant, uniform grid, constant inflow, windowed uniform rain.
    n_cells = 100
    x = np.linspace(0.05, 9.95, n_cells)
    dx = np.full(n_cells, 0.1)
    slope = np.full(n_cells, 0.05)
    manning = np.full(n_cells, 0.05 / 60.0)
    h0 = np.full(n_cells, 0.02)
    rain_rate = 1e-5

    def rain(x_arr, t):
        return np.full_like(x_arr, rain_rate if t < 5.0 else 0.0)

    raw = sv1.run_model(None, 10.0, record_interval=1.0, h_init=h0, left_inflow=0.02,
                        rainfall=rain, x_m=x, dx_m=dx, slope=slope, manning_n=manning)
    _write_reference("saint_venant_uniform.json", {
        "solver": "saint_venant",
        "grid": {"x_m": x.tolist(), "dx_m": dx.tolist(),
                 "slope": slope.tolist(), "manning_n": manning.tolist()},
        "options": {"tFinalMin": 10.0, "recordIntervalMin": 1.0, "leftInflow": 0.02,
                    "hInit": h0.tolist(), "rain": {"rate": rain_rate, "endMin": 5.0}},
        "expected": _sv_expected(raw),
    })

    # Saint-Venant, non-uniform coarse grid.
    x = np.array([0.0, 50.0, 120.0, 200.0, 300.0])
    dx = np.array([50.0, 60.0, 75.0, 90.0, 100.0])
    slope = np.array([0.01, 0.008, 0.006, 0.005, 0.004])
    manning = np.full(5, 8e-4)
    h0 = np.array([0.3, 0.28, 0.26, 0.25, 0.24])

    def rain2(x_arr, t):
        return np.full_like(x_arr, 1e-5 if t < 8.0 else 0.0)

    raw = sv1.run_model(None, 20.0, record_interval=2.0, h_init=h0, left_inflow=0.5,
                        rainfall=rain2, x_m=x, dx_m=dx, slope=slope, manning_n=manning)
    _write_reference("saint_venant_nonuniform.json", {
        "solver": "saint_venant",
        "grid": {"x_m": x.tolist(), "dx_m": dx.tolist(),
                 "slope": slope.tolist(), "manning_n": manning.tolist()},
        "options": {"tFinalMin": 20.0, "recordIntervalMin": 2.0, "leftInflow": 0.5,
                    "hInit": h0.tolist(), "rain": {"rate": 1e-5, "endMin": 8.0}},
        "expected": _sv_expected(raw),
    })

    _build_2d_references()


def _build_2d_references():
    """Reference cases for the 2-D port.

    Deliberately tiny so a parity failure is debuggable by hand, and chosen to
    exercise the places a port actually diverges: hydrostatic reconstruction
    over a non-flat bed, advancing wet/dry fronts, and the wall/inflow
    ghost-cell construction.

    Not covered, by construction: the draining limiter's theta < 1 branch. At
    the enforced cfl <= 0.5 the ratio of one step's outgoing volume to a cell's
    available volume is bounded by the CFL number, so theta stays at 1.0 in
    every case here (measured, not assumed). The donor-cell scaling therefore
    multiplies by exactly 1.0 in both implementations and cannot make them
    disagree; the branch is a positivity guard, not a live code path.
    """
    # 1. Flat bed, uniform inflow, windowed rain. Exercises the plain Rusanov
    #    path and the mass ledger.
    x, dx = sv2._grid(40.0, 8)
    y, dy = sv2._grid(24.0, 6)
    bed = np.zeros((8, 6))
    h0 = np.full((8, 6), 0.30)
    _write_2d_reference("saint_venant_2d_flat.json", x, y, dx, dy, bed,
                        h_init=h0, manning_n=np.full((8, 6), 0.035 / 60.0),
                        t_final=1.5, record_interval=0.5, left_inflow=4.0,
                        rain_rate=2e-4, rain_end=1.0)

    # 2. Compound cross-section with dry benches: the well-balanced hydrostatic
    #    reconstruction and the wetting front over non-flat bed.
    x, dx = sv2._grid(300.0, 10)
    y, dy = sv2._grid(160.0, 8)
    centre = 0.5 * (y[0] + y[-1])
    lateral = 1.2 * np.clip((np.abs(y - centre) - 30.0) / 40.0, 0.0, 1.0)
    bed = (-0.003 * x)[:, None] + lateral[None, :]
    surface = (-0.003 * x + 0.8)[:, None]
    h0 = np.maximum(surface - bed, 0.0)
    manning = np.full((10, 8), 0.04 / 60.0)
    hu0 = (1.0 / manning) * np.power(h0, 5.0 / 3.0) * np.sqrt(0.003)
    _write_2d_reference("saint_venant_2d_compound.json", x, y, dx, dy, bed,
                        h_init=h0, hu_init=hu0, manning_n=manning,
                        t_final=2.0, record_interval=0.5,
                        left_inflow=(3.0 * hu0[0, :]).tolist(),
                        rain_rate=0.0, rain_end=0.0)

    # 3. Collapsing pond on a sloping bed with no inflow: a wet/dry front
    #    spreading into initially dry cells in both directions at once.
    x, dx = sv2._grid(60.0, 6)
    y, dy = sv2._grid(40.0, 5)
    bed = (-0.02 * x)[:, None] + np.zeros((1, 5))
    h0 = np.zeros((6, 5))
    h0[1:3, 1:4] = 0.25
    _write_2d_reference("saint_venant_2d_draining.json", x, y, dx, dy, bed,
                        h_init=h0, manning_n=np.full((6, 5), 0.03 / 60.0),
                        t_final=1.0, record_interval=0.25, left_inflow=0.0,
                        rain_rate=0.0, rain_end=0.0)


def _write_2d_reference(name, x, y, dx, dy, bed, *, h_init, manning_n,
                        t_final, record_interval, left_inflow, rain_rate,
                        rain_end, hu_init=None):
    def rainfall(x_arr, y_arr, t):
        return rain_rate if (rain_rate > 0 and t < rain_end) else 0.0

    raw = sv2.run_model(
        T_final=t_final,
        record_interval=record_interval,
        h_init=h_init,
        hu_init=0.0 if hu_init is None else hu_init,
        left_inflow=np.asarray(left_inflow, dtype=float) if isinstance(left_inflow, list) else left_inflow,
        rainfall=rainfall,
        x_m=x, y_m=y, dx_m=dx, dy_m=dy,
        manning_n=manning_n,
        bed_elevation_m=bed,
    )
    _write_reference(name, {
        "solver": "saint_venant_2d",
        "grid": {
            "x_m": x.tolist(), "y_m": y.tolist(),
            "dx_m": dx.tolist(), "dy_m": dy.tolist(),
            "bed_elevation_m": bed.ravel().tolist(),
            "manning_n": np.asarray(manning_n).ravel().tolist(),
        },
        "options": {
            "tFinalMin": t_final,
            "recordIntervalMin": record_interval,
            "leftInflow": left_inflow,
            "hInit": np.asarray(h_init).ravel().tolist(),
            "huInit": None if hu_init is None else np.asarray(hu_init).ravel().tolist(),
            "rain": {"rate": rain_rate, "endMin": rain_end},
        },
        "expected": {
            "nx": int(len(x)),
            "ny": int(len(y)),
            "times": [float(v) for v in raw["times"]],
            "depth_history": [[float(v) for v in frame.ravel()] for frame in raw["h_history"]],
            "discharge_x_history": [[float(v) for v in frame.ravel()] for frame in raw["hu_history"]],
            "discharge_y_history": [[float(v) for v in frame.ravel()] for frame in raw["hv_history"]],
            "mass_inflow": float(raw["mass_inflow"]),
            "mass_source": float(raw["mass_source"]),
            "mass_outflow": float(raw["mass_outflow"]),
            "mass_floor_correction": float(raw["mass_floor_correction"]),
        },
    })


def _profile_dict(profile):
    return {
        "station_m": profile.station_m.tolist(),
        "dx_m": profile.dx_m.tolist(),
        "slope": profile.slope.tolist(),
        "manning_n": profile.manning_n.tolist(),
        "initial_depth_m": None if profile.initial_depth_m is None else profile.initial_depth_m.tolist(),
        "rainfall_rate_m_per_min": None if profile.rainfall_rate_m_per_min is None else profile.rainfall_rate_m_per_min.tolist(),
    }


def _kw_expected(result):
    return {
        "times": [float(v) for v in result["times"]],
        "depth_history": [[float(v) for v in row] for row in result["depth_history"]],
        "mass_inflow": float(result["mass_inflow"]),
        "mass_source": float(result["mass_source"]),
        "mass_outflow": float(result["mass_outflow"]),
    }


def _sv_expected(raw):
    return {
        "times": [float(v) for v in raw["times"]],
        "depth_history": [[float(v) for v in row] for row in raw["h_history"]],
        "discharge_history": [[float(v) for v in row] for row in raw["q_history"]],
        "mass_inflow": float(raw["mass_inflow"]),
        "mass_source": float(raw["mass_source"]),
        "mass_outflow": float(raw["mass_outflow"]),
        "mass_floor_correction": float(raw["mass_floor_correction"]),
    }


def _write_reference(name, payload):
    path = REFERENCE_DIR / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"  reference: {path.name}")


# ── observed-event validation suite ────────────────────────────────────────

VALIDATION_DIR = REPO_ROOT / "real_world_rivers" / "validation"

# Below this observed coefficient of variation, NSE's variance denominator is so
# small that the score stops being a reliable read on the model (see metric note).
LOW_VARIABILITY_COV_PERCENT = 5.0

# A cell counts as flooding-relevant nowhere here; validation scores discharge.
# River display order and short ids, data-driven from the case config's river.
def _river_id(river_name):
    return river_name.lower().replace(" river", "").replace(" ", "_").strip("_")


def _extended_routing_lag(stem):
    """Routing-lag error from the 48-hour extended-window rerun, if one exists.

    The extended-window study showed much of each baseline's apparent lag was a
    truncated-window artifact (Truckee −375 → −60 min once the window includes
    the peak and recession), so the baseline figure must never stand alone.
    """
    path = VALIDATION_DIR / f"{stem}_extended_48h.results.json"
    if not path.exists():
        return None
    reach = json.loads(path.read_text(encoding="utf-8")).get("reach_diagnosis") or {}
    return reach.get("routing_lag_error_min")


def _validation_event(stem, *, predeclared=False):
    """Reshape one 2-D case result: series, steadiness, and the tracked diagnostics.

    Nothing is recomputed from the solver. CoV is derived from the observed
    series; every other field is copied from the results file, including the
    error/reach diagnosis guards verbatim (they mark oracle-only quantities that
    must never be read as a permitted correction).
    """
    result = json.loads((VALIDATION_DIR / f"{stem}.results.json").read_text(encoding="utf-8"))
    if result.get("solver") != "saint_venant_2d":
        raise ValueError(
            f"{stem}: validation is 2-D only, but its result solver is {result.get('solver')!r}"
        )
    status_l = str(result.get("status", "")).lower()
    if "calibrated" in status_l and "uncalibrated" not in status_l:
        raise ValueError(f"{stem}: a calibrated status ({result['status']}) cannot be published as validation")

    series = result["series"]
    observed = np.asarray(series["observed_downstream_m3_per_min"], dtype=float)
    predicted = np.asarray(series["predicted_downstream_m3_per_min"], dtype=float)
    residual = observed - predicted
    count = len(observed)
    first, second = count // 3, 2 * count // 3
    mean_observed = float(observed.mean()) if count else float("nan")
    cov = float(100.0 * observed.std() / mean_observed) if mean_observed else float("nan")

    terrain = result.get("terrain_representation") or {}
    err = result.get("error_diagnosis") or {}
    reach = result.get("reach_diagnosis") or {}
    counterfactual = err.get("volume_only_counterfactual") or {}
    observed_volume = reach.get("observed_downstream_volume_m3")
    upstream_volume = reach.get("upstream_volume_m3")
    unexplained = reach.get("unexplained_downstream_volume_m3")
    # How much more (or less) water leaves the reach than enters it. Above 1 means
    # water joins between the gauges — flow the boundary-driven model was never
    # given. This is the single strongest predictor of where the model misses.
    down_up_volume = (
        float(observed_volume / upstream_volume)
        if observed_volume and upstream_volume else None
    )

    window = result["case"]["observation_window"]
    return {
        "id": stem,
        "name": result["case"]["name"],
        "date": window[0][:10],
        "status": result["status"],
        "purpose": result["case"].get("purpose", ""),
        "predeclared": predeclared,
        "cov_percent": cov,
        "low_variability": cov < LOW_VARIABILITY_COV_PERCENT,
        "observed_mean_m3_per_min": mean_observed,
        "times_min": [float(v) for v in series.get("times_min", series.get("observed_times_min"))],
        "observed_m3_per_min": [float(v) for v in observed],
        "predicted_m3_per_min": [float(v) for v in predicted],
        "scores": result["scores"],
        "terrain": {
            "name": terrain.get("name"),
            "x_cells": terrain.get("x_cells"),
            "y_cells": terrain.get("y_cells"),
            "limitation": terrain.get("limitation"),
        },
        "diagnostics": {
            "volume_ratio": err.get("volume_ratio"),
            "amplitude_ratio": err.get("amplitude_ratio"),
            "peak_lag_min": err.get("peak_lag_min"),
            "volume_only_counterfactual": {
                "scaled_nse": (counterfactual.get("scores") or {}).get("nse"),
                "scale": counterfactual.get("scale_applied_to_prediction"),
                "guard": counterfactual.get("guard"),
            } if counterfactual else None,
            "observed_routing_lag_min": reach.get("observed_routing_lag_min"),
            "modeled_routing_lag_min": reach.get("modeled_routing_lag_min"),
            "routing_lag_error_min": reach.get("routing_lag_error_min"),
            "routing_lag_error_min_extended": _extended_routing_lag(stem),
            "downstream_over_upstream_volume": down_up_volume,
            "unexplained_volume_fraction": (
                float(unexplained / observed_volume)
                if observed_volume and unexplained is not None else None
            ),
            "reach_guard": reach.get("guard"),
        },
        "residual_thirds_m3_per_min": {
            "early": float(residual[:first].mean()),
            "middle": float(residual[first:second].mean()),
            "late": float(residual[second:].mean()),
        },
    }


# Predeclared hypotheses for the three cross-river holdouts, summarising the
# predictions recorded in docs/cross_river_fit_investigation.md *before* scoring.
# Eel and Willamette are the tributary-rich pair; Chattahoochee is the regulated
# pulse-routing analogue for Glen Canyon Dam.
HOLDOUT_HYPOTHESES = {
    "eel_fort_seward_scotia_2019-02-26": {
        "river": "Eel River",
        "prediction": "Tributary-rich coastal flood in the same storm as the Russian event — predicted to underestimate (large intervening volume gain) while keeping hydrograph shape.",
        "confirmed_expectation": "underestimate",
    },
    "willamette_albany_salem_2012-01-18": {
        "river": "Willamette River",
        "prediction": "Tributary-rich Pacific-Northwest flood — predicted to underestimate strongly from ungauged intervening inflow, with correlation still high.",
        "confirmed_expectation": "underestimate",
    },
    "chattahoochee_buford_norcross_2019-07-01": {
        "river": "Chattahoochee River",
        "prediction": "Regulated dam-pulse reach, a routing analogue for Glen Canyon — predicted to behave like Colorado: modest bias, fast routing, no large volume gain.",
        "confirmed_expectation": "colorado-like",
    },
}


def _fit_regime():
    """Per-river volume accounting that explains where the model fits and misses.

    Mechanism first (a boundary-driven model cannot route water it was never
    given), correlation second, with codex's own warning that these purposive
    river tests rank hypotheses rather than establish a regression.
    """
    fr = json.loads((VALIDATION_DIR / "fit_regime_analysis.results.json").read_text(encoding="utf-8"))
    rivers = [
        {
            "river": row["river"],
            "events": row["event_count"],
            "nse": row["median_nse"],
            "pearson_r": row["median_pearson_r"],
            "percent_bias": row["median_percent_bias"],
            "predicted_over_observed_volume": row["median_predicted_to_observed_volume_ratio"],
            "downstream_over_upstream_volume": row["median_observed_downstream_to_upstream_volume_ratio"],
        }
        for row in fr["river_summary"]
    ]
    rivers.sort(key=lambda r: r["downstream_over_upstream_volume"])
    return {
        "title": "Why the fit varies across rivers",
        "lede": (
            "The model is driven only by the upstream gauge. Where the downstream gauge carries "
            "roughly the same water that entered (ratio near 1), it scores well; where much more "
            "water leaves than entered — tributaries and runoff joining between the gauges — it "
            "underestimates, because it was never given that water. The columns below are all "
            "observed or modelled volumes; none is fitted."
        ),
        "rivers": rivers,
        "associations": fr.get("natural_flow_associations"),
        "conclusions": fr.get("conclusions"),
        "excluded_rivers": fr.get("excluded_rivers"),
    }


def _holdout_group():
    """The three predeclared cross-river holdouts, framed as prediction-then-score."""
    events = []
    for stem, meta in HOLDOUT_HYPOTHESES.items():
        event = _validation_event(stem, predeclared=True)
        event["prediction"] = meta["prediction"]
        events.append(event)
    return {
        "title": "Predeclared holdout rivers",
        "lede": (
            "Three rivers named and predicted before their downstream data was scored, to test "
            "the intervening-flow explanation rather than illustrate it. Two tributary-rich rivers "
            "were predicted to underestimate; a regulated dam-pulse river was predicted to behave "
            "like the Colorado. All three came in as predicted."
        ),
        "criterion": json.loads(
            (VALIDATION_DIR / "fit_regime_analysis.results.json").read_text(encoding="utf-8")
        ).get("matched_holdout_test", {}).get("criterion"),
        "events": events,
    }


def _tributary_case():
    """Snoqualmie baseline vs the same reach forced with observed tributary gauges.

    An explicit two-source before/after (the suite still scores the baseline).
    The guards are the point: correlation falls slightly while water balance
    improves sharply, so a single-metric objective would reject a physically
    correct forcing; and a large residual remains that is not licence to fit.
    """
    base = _validation_event("snoqualmie_snoqualmie_carnation_2009-01-07")
    trib = _validation_event("snoqualmie_snoqualmie_carnation_2009-01-07_observed_tributaries")
    trib_raw = json.loads(
        (VALIDATION_DIR / "snoqualmie_snoqualmie_carnation_2009-01-07_observed_tributaries.results.json")
        .read_text(encoding="utf-8")
    )
    reach = trib_raw.get("reach_diagnosis") or {}
    return {
        "title": "Supplying the missing water: observed tributaries",
        "lede": (
            "The Snoqualmie miss is the intervening-flow story made concrete. Re-running the same "
            "reach with two independently observed tributary gauges — Raging River (USGS-12145500) "
            "and Tolt River (USGS-12148500), located on the reach via the USGS NLDI network, with "
            "no magnitude or timing taken from the downstream gauge — recovers most of the deficit."
        ),
        "tributary_sources": ["Raging River · USGS-12145500", "Tolt River · USGS-12148500"],
        "baseline": base,
        "with_tributaries": trib,
        "supplied_volume_m3": reach.get("modeled_net_change_from_upstream_m3"),
        "unexplained_volume_m3": reach.get("unexplained_downstream_volume_m3"),
        "guards": [
            "Correlation actually falls (0.798 → 0.749) while bias improves from −32.6% to −13.6% and "
            "NSE from 0.23 to 0.47: a single correlation objective would have rejected a physically "
            "necessary forcing. Water balance and shape are different questions.",
            "About 49.8 million m³ of downstream volume is still unexplained after both gauges. That "
            "residual can be ungauged runoff, gauge-to-confluence travel, storage, controls, geometry, "
            "or measurement error — it is not licence to fit an extra source from the downstream gauge.",
        ],
    }


def build_validation():
    """Publish the observed-data comparison: Saint-Venant 2-D across eight rivers.

    Source: real_world_rivers/validation/, produced by
    src/rivers/validation/run_suite.py (and run_case_2d.py). Nothing is
    recomputed from the solver; this reshapes the tracked 2-D results and adds
    the steadiness context (CoV) the honest reading of NSE needs.

    The suite's standing claim is now `parameter_policy`: every event runs
    Saint-Venant 2-D only, no fitted parameter, no 1-D fallback. _validation_event
    verifies the 2-D-only half per case (and refuses any calibrated status); this
    keeps the published page from outrunning what the suite actually did.
    """
    suite = json.loads((VALIDATION_DIR / "validation_suite.json").read_text(encoding="utf-8"))
    summary = json.loads((VALIDATION_DIR / "validation_suite.results.json").read_text(encoding="utf-8"))

    # Group the 12 cases by river, preserving suite order.
    rivers_by_id = {}
    order = []
    for config_name in suite["cases"]:
        stem = config_name[:-len(".json")]
        config = json.loads((VALIDATION_DIR / config_name).read_text(encoding="utf-8"))
        case = config["case"]
        river_name = case["river"]
        rid = _river_id(river_name)
        predeclared = "selection_protocol" in case
        if rid not in rivers_by_id:
            order.append(rid)
            rivers_by_id[rid] = {
                "id": rid,
                "name": river_name,
                "gauges": f"{case['upstream_gauge']} → {case['downstream_gauge']}",
                "predeclared_policy": case.get("selection_protocol") if predeclared else None,
                "events": [],
            }
        rivers_by_id[rid]["events"].append(_validation_event(stem, predeclared=predeclared))

    rivers = []
    for rid in order:
        river = rivers_by_id[rid]
        river["kind"] = (
            "predeclared_transfer_test" if river["predeclared_policy"]
            else "multi_event_river" if len(river["events"]) > 1
            else "independent_river"
        )
        rivers.append(river)

    payload = {
        "generated_by": "website/build_scenarios.py --validation",
        "suite_name": suite["suite"]["name"],
        "parameter_policy": suite["parameter_policy"],
        "score_summary": summary.get("score_summary"),
        "event_count": sum(len(r["events"]) for r in rivers),
        "river_count": len(rivers),
        "rivers": rivers,
        "metric_note": {
            "title": "Why a good model can score a terrible NSE",
            "text": (
                "Nash–Sutcliffe (NSE) measures the model's error against the variance of "
                "the observations themselves. When a river's flow barely changes across the "
                "window, that variance is tiny and NSE stops being a reliable guide to the "
                "model: a small, physically reasonable error can still drive a large negative "
                "score. Each event below shows its observed variability (CoV); when it is only "
                "a few percent, read correlation and bias instead of NSE."
            ),
            "cov_threshold_percent": LOW_VARIABILITY_COV_PERCENT,
        },
        "diagnostics_note": {
            "title": "How each event is diagnosed",
            "text": (
                "For every event the pipeline reports where the error lives: the volume ratio "
                "and amplitude ratio separate a level error from a shape error, the routing-lag "
                "error compares modelled and observed travel time, and a volume-only "
                "counterfactual shows what the score would be if the prediction were rescaled to "
                "the observed total. That counterfactual uses held-out downstream volume as an "
                "oracle — it is a diagnostic, never a permitted correction — and its guard text "
                "is shown with it."
            ),
        },
        "routing_note": {
            "title": "Routing lag: read the extended window",
            "text": (
                "An early diagnostic scored each event over a short window that often ended near "
                "the flood peak, so part of the apparent fast-routing was a truncated-window "
                "artifact. Re-scoring every event with 48 more hours appended shrinks the error "
                "(Truckee January fell from −375 to −60 minutes) but does not remove it: 13 of 15 "
                "extended cases still route early, 10 by at least an hour. Each event shows both "
                "figures; trust the extended one."
            ),
        },
        "fit_regime": _fit_regime(),
        "holdouts": _holdout_group(),
        "tributary_case": _tributary_case(),
    }
    path = DATA_DIR / "validation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KB) — "
          f"{payload['river_count']} rivers, {payload['event_count']} events, 2-D only")
    for river in rivers:
        print(f"  {river['name']}  ({river['kind']})")
        for event in river["events"]:
            s = event["scores"]
            flag = "  [low-variability: NSE unreliable]" if event["low_variability"] else ""
            print(f"    {event['date']}  NSE {s['nse']:+.3f}  bias {s['percent_bias']:+.1f}%  "
                  f"r {s['pearson_r']:.3f}  CoV {event['cov_percent']:.1f}%{flag}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build Flood Explorer data artifacts")
    parser.add_argument("--references", action="store_true",
                        help="Write JS-parity reference cases for the solver ports")
    parser.add_argument("--validation", action="store_true",
                        help="Rebuild only the observed-event validation data (the default)")
    parser.add_argument("--atlas", action="store_true",
                        help="Rebuild the retired Regions atlas data (not shown on the site)")
    args = parser.parse_args(argv)
    if args.references:
        build_references()
    elif args.atlas:
        build_atlas()
    else:
        # Default: validation only. The Regions atlas is retired from the site.
        build_validation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
