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
    card.querySelector(".badge").textContent = region.kind === "real" ? "real data" : "synthetic";
    card.querySelector("h3").textContent = region.name;
    card.querySelector("p").textContent = region.description;
    card.querySelector(".region-facts").textContent =
      `${formatNumber(region.length_m / 1000, 1)} km · ${region.cells} cells · solver: ${region.solver}`;
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

function renderScenario(scenario) {
  const view = document.getElementById("scenario-view");
  view.hidden = false;

  document.getElementById("scenario-title").textContent =
    `${scenario.region.name} — ${scenario.event.name}`;
  document.getElementById("scenario-narrative").textContent = scenario.event.narrative;

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

  if (state.animation) state.animation.stop();
  const depth = depthProfileChart(document.getElementById("depth-chart"), {
    stations: scenario.station_m,
    depthHistory: scenario.depth_history,
  });
  const hydro = hydrographChart(document.getElementById("hydrograph-chart"), {
    times: scenario.times_min,
    values: scenario.metrics.downstream_hydrograph_m2_per_min,
    unit: "discharge (m²/min)",
  });
  state.animation = animator({
    frameCount: scenario.times_min.length,
    times: scenario.times_min,
    onFrame: (i) => {
      depth.setFrame(i);
      hydro.setCursor(scenario.times_min[i]);
    },
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
