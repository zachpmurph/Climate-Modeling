/* Regions tab: browse precomputed scenarios (real Python-solver output). */

import {
  animator,
  compareBars,
  depthProfileChart,
  formatMinutes,
  formatNumber,
  hydrographChart,
  metricTiles,
} from "./charts.js";
import { crossSectionChart, inundationMap } from "./charts2d.js";

const state = {
  index: null,
  region: null,
  eventId: null,
  animation: null,
};

function showError(message) {
  const card = document.getElementById("atlas-error");
  card.hidden = false;
  card.textContent = message;
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

function renderRegionList() {
  const list = document.getElementById("region-list");
  list.innerHTML = "";
  for (const region of state.index.regions) {
    const card = document.createElement("button");
    card.className = "region-card";
    card.type = "button";
    card.innerHTML = `
      <span class="badge ${region.kind}"></span>
      <h3></h3>
      <p></p>
      <span class="region-facts"></span>`;
    const badge = card.querySelector(".badge");
    badge.textContent = region.dimensions === 2
      ? "2-D synthetic"
      : (region.kind === "real" ? "real data" : "synthetic");
    if (region.dimensions === 2) badge.classList.add("two-d");
    card.querySelector("h3").textContent = region.name;
    card.querySelector("p").textContent = region.description;
    const extent = region.dimensions === 2
      ? `${formatNumber(region.length_m / 1000, 1)} × ${formatNumber(region.width_m, 0)} m · ${region.nx}×${region.ny} cells`
      : `${formatNumber(region.length_m / 1000, 1)} km · ${region.cells} cells`;
    card.querySelector(".region-facts").textContent =
      `${extent} · solver: ${region.solver}`;
    card.onclick = () => selectRegion(region, card);
    list.appendChild(card);
  }
}

function selectRegion(region, card) {
  document.querySelectorAll(".region-card").forEach((node) => node.classList.remove("selected"));
  if (card) card.classList.add("selected");
  state.region = region;
  state.eventId = null;
  renderEventChips();
  loadScenario(region.events[0]);
}

function renderEventChips() {
  const holder = document.getElementById("event-chips");
  holder.innerHTML = "";
  for (const event of state.region.events) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    chip.textContent = event.name;
    chip.dataset.eventId = event.id;
    chip.onclick = () => loadScenario(event);
    holder.appendChild(chip);
  }
}

async function loadScenario(eventEntry) {
  try {
    const scenario = await fetchJson(eventEntry.file);
    state.eventId = eventEntry.id;
    document.querySelectorAll("#event-chips .chip").forEach((chip) => {
      chip.classList.toggle("active", chip.dataset.eventId === eventEntry.id);
    });
    renderScenario(scenario);
  } catch (error) {
    showError(`Could not load scenario data (${error.message}). If you opened index.html directly from disk, serve the folder instead: python -m http.server`);
  }
}

function show(id, visible) {
  document.getElementById(id).hidden = !visible;
}

/** 1-D scenario: depth profile along the reach + unit-discharge hydrograph. */
function renderScenario1d(scenario) {
  const metrics = scenario.metrics;
  metricTiles(document.getElementById("scenario-metrics"), [
    { label: "Peak depth", value: `${formatNumber(metrics.peak_depth_m, 2)} m`,
      sub: `vs ${formatNumber(metrics.initial_max_depth_m, 2)} m initially` },
    { label: "Peak arrives", value: formatMinutes(metrics.peak_time_min),
      sub: `at ${formatNumber(metrics.peak_station_m / 1000, 1)} km downstream` },
    { label: "Upstream inflow", value: `${formatNumber(scenario.event.left_inflow_m2_per_min, 1)}`,
      sub: "m²/min per m width" },
    { label: "Mass-balance error", value: formatNumber(metrics.mass_balance_error, 3),
      sub: "m² (numerical quality)" },
  ]);

  const depth = depthProfileChart(document.getElementById("depth-chart"), {
    stations: scenario.station_m,
    depthHistory: scenario.depth_history,
  });
  document.getElementById("hydrograph-title").textContent = "Downstream discharge";
  const hydro = hydrographChart(document.getElementById("hydrograph-chart"), {
    times: scenario.times_min,
    values: metrics.downstream_hydrograph_m2_per_min,
    unit: "discharge (m²/min)",
  });
  return (i) => {
    depth.setFrame(i);
    hydro.setCursor(scenario.times_min[i]);
  };
}

/** 2-D scenario: inundation map, cross-section, volumetric hydrograph, area. */
function renderScenario2d(scenario) {
  const metrics = scenario.metrics;
  const grid = {
    nx: scenario.nx,
    ny: scenario.ny,
    x_m: scenario.x_m,
    y_m: scenario.y_m,
    bedElevation: scenario.bed_elevation_m,
  };
  const hectares = metrics.peak_flooded_area_m2 / 10000;
  metricTiles(document.getElementById("scenario-metrics"), [
    { label: "Peak depth", value: `${formatNumber(metrics.peak_depth_m, 2)} m`,
      sub: `${formatNumber(metrics.peak_station_m, 0)} m downstream, ${formatNumber(metrics.peak_across_m, 0)} m across` },
    { label: "Floodplain under water", value: `${formatNumber(hectares, 1)} ha`,
      sub: `${formatNumber(metrics.peak_flooded_fraction * 100, 0)}% of the normally dry ground` },
    { label: "Peak arrives", value: formatMinutes(metrics.peak_time_min),
      sub: hectares > 0
        ? `floodplain widest at ${formatMinutes(metrics.peak_flooded_time_min)}`
        : "the channel contains the flow" },
    { label: "Upstream inflow", value: `${formatNumber(scenario.event.inflow_multiple_of_baseline, 2)}× baseline`,
      sub: `${formatNumber(scenario.event.left_inflow_m3_per_min, 0)} m³/min` },
    { label: "Mass-balance error", value: formatNumber(metrics.mass_balance_error, 3),
      sub: "m³ (numerical quality)" },
  ]);

  const map = inundationMap(document.getElementById("map-chart"), {
    grid,
    depthHistory: scenario.depth_history,
    wetTolM: metrics.flood_depth_threshold_m,
  });

  // Section through the deepest point the event reaches, so the picture shows
  // the water actually leaving the channel.
  let stationIndex = 0;
  let best = Infinity;
  for (let i = 0; i < scenario.x_m.length; i += 1) {
    const gap = Math.abs(scenario.x_m[i] - metrics.peak_station_m);
    if (gap < best) {
      best = gap;
      stationIndex = i;
    }
  }
  document.getElementById("section-title").textContent =
    `Cross-section ${formatNumber(scenario.x_m[stationIndex], 0)} m downstream (through the deepest point)`;
  const section = crossSectionChart(document.getElementById("section-chart"), {
    grid,
    depthHistory: scenario.depth_history,
    stationIndex,
    maxDepth: map.peakDepth,
  });

  document.getElementById("hydrograph-title").textContent = "Discharge past the downstream end";
  const hydro = hydrographChart(document.getElementById("hydrograph-chart"), {
    times: scenario.times_min,
    values: metrics.downstream_hydrograph_m3_per_min,
    unit: "discharge (m³/min)",
  });
  const areaChart = hydrographChart(document.getElementById("area-chart"), {
    times: scenario.times_min,
    values: metrics.flooded_area_history_m2.map((v) => v / 10000),
    unit: "flooded area (ha)",
    minAxisMax: 1,
  });

  return (i) => {
    map.setFrame(i);
    section.setFrame(i);
    hydro.setCursor(scenario.times_min[i]);
    areaChart.setCursor(scenario.times_min[i]);
  };
}

function renderScenario(scenario) {
  const view = document.getElementById("scenario-view");
  view.hidden = false;

  document.getElementById("scenario-title").textContent =
    `${scenario.region.name} — ${scenario.event.name}`;
  document.getElementById("scenario-narrative").textContent = scenario.event.narrative;

  const isTwoD = scenario.region.dimensions === 2;
  show("block-1d-depth", !isTwoD);
  show("block-2d-map", isTwoD);
  show("block-2d-section", isTwoD);
  show("block-2d-area", isTwoD);

  if (state.animation) state.animation.stop();
  const onFrame = isTwoD ? renderScenario2d(scenario) : renderScenario1d(scenario);
  state.animation = animator({
    frameCount: scenario.times_min.length,
    times: scenario.times_min,
    onFrame,
    playButton: document.getElementById("play-toggle"),
    slider: document.getElementById("frame-slider"),
    label: document.getElementById("frame-label"),
  });
  state.animation.play();

  compareBars(
    document.getElementById("event-compare"),
    state.region.events.map((event) => ({ id: event.id, name: event.name, value: event.peak_depth_m })),
    scenario.event.id,
  );

  const limitations = document.getElementById("scenario-limitations");
  limitations.innerHTML = "<strong>Limitations</strong><ul></ul>";
  const list = limitations.querySelector("ul");
  for (const item of scenario.limitations) {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  }

  view.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

export async function initAtlas() {
  try {
    state.index = await fetchJson("data/index.json");
    renderRegionList();
    const firstCard = document.querySelector(".region-card");
    if (state.index.regions.length > 0) {
      selectRegion(state.index.regions[0], firstCard);
    }
  } catch (error) {
    showError(`Could not load the scenario index (${error.message}). Serve this folder over HTTP: python -m http.server`);
  }
}
