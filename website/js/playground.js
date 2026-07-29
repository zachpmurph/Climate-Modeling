/* Playground tab: run the parity-tested JS solver ports on user-defined
 * conditions, in the browser. */

import { kinematicWave, saintVenant1D, manningFlux } from "./solvers.js";
import {
  compoundChannelBed,
  normalFlowState,
  saintVenant2D,
  uniformAxis,
  CFL_2D,
  G_M_PER_MIN2,
} from "./solvers2d.js";
import {
  animator,
  depthProfileChart,
  formatMinutes,
  formatNumber,
  hydrographChart,
  metricTiles,
} from "./charts.js";
import { crossSectionChart, inundationMap } from "./charts2d.js";

const MM_PER_HOUR_TO_M_PER_MIN = 1 / 1000 / 60;

// A 2-D run costs (time steps x cells) cell updates and the browser does about
// 4.5 million per second. Refusing above this keeps the worst accepted run to a
// few seconds rather than freezing the tab; measured, not guessed.
const WORK_BUDGET_CELL_UPDATES = 20e6;
const FLOOD_DEPTH_M = 0.05;

let animation = null;

function readForm(form) {
  const data = new FormData(form);
  const num = (name) => Number(data.get(name));
  return {
    model: data.get("model"),
    length: num("length"),
    cells: Math.round(num("cells")),
    slope: num("slope"),
    manningSI: num("manning_si"),
    depth0: num("depth0"),
    inflow: num("inflow"),
    rainMmPerHour: num("rain"),
    rainMinutes: num("rain_minutes"),
    minutes: num("minutes"),
    // 2-D only
    width: num("width"),
    cellsY: Math.round(num("cells_y")),
    halfWidth: num("half_width"),
    bankHeight: num("bank_height"),
    channelDepth: num("channel_depth"),
    inflowMultiple: num("inflow_multiple"),
  };
}

function validate(p) {
  const problems = [];
  if (!(p.length >= 100 && p.length <= 100000)) problems.push("reach length must be 100–100000 m");
  if (!(p.cells >= 10 && p.cells <= 400)) problems.push("cells must be 10–400");
  if (!(p.slope >= 5e-5 && p.slope <= 0.1)) problems.push("slope must be 5e-5–0.1");
  if (!(p.manningSI >= 0.01 && p.manningSI <= 0.15)) problems.push("Manning n must be 0.01–0.15 (SI)");
  if (!(p.rainMmPerHour >= 0 && p.rainMmPerHour <= 150)) problems.push("rain must be 0–150 mm/h");
  if (!(p.minutes >= 5 && p.minutes <= 1440)) problems.push("duration must be 5–1440 min");

  if (p.model === "saint_venant_2d") {
    if (!(p.cells >= 10 && p.cells <= 120)) problems.push("2-D runs allow 10–120 cells along the reach");
    if (!(p.width >= 40 && p.width <= 2000)) problems.push("valley width must be 40–2000 m");
    if (!(p.cellsY >= 6 && p.cellsY <= 40)) problems.push("cells across the valley must be 6–40");
    if (!(p.halfWidth >= 5 && p.halfWidth < p.width / 2)) {
      problems.push("channel half-width must be at least 5 m and less than half the valley width");
    }
    if (!(p.bankHeight >= 0.1 && p.bankHeight <= 10)) problems.push("bank height must be 0.1–10 m");
    if (!(p.channelDepth >= 0.05 && p.channelDepth <= 10)) problems.push("channel depth must be 0.05–10 m");
    if (!(p.inflowMultiple >= 0 && p.inflowMultiple <= 20)) problems.push("inflow multiple must be 0–20");
    return problems;
  }

  if (!(p.depth0 >= 0 && p.depth0 <= 10)) problems.push("initial depth must be 0–10 m");
  if (!(p.inflow >= 0 && p.inflow <= 2000)) problems.push("inflow must be 0–2000 m²/min");
  if (p.inflow === 0 && p.rainMmPerHour === 0 && p.depth0 === 0) {
    problems.push("give the river some water: inflow, rain, or initial depth");
  }
  return problems;
}

function buildProfile(p) {
  const stations = new Array(p.cells);
  const dx = p.length / (p.cells - 1);
  for (let i = 0; i < p.cells; i += 1) stations[i] = i * dx;
  const widths = new Array(p.cells).fill(dx);
  // Match RiverProfile's edge handling: end cells get half-width + half-extension,
  // but with uniform spacing every cell width is simply dx.
  return {
    station_m: stations,
    dx_m: widths,
    slope: new Array(p.cells).fill(p.slope),
    manning_n: new Array(p.cells).fill(p.manningSI / 60.0),
    initial_depth_m: new Array(p.cells).fill(Math.max(p.depth0, 1e-6)),
  };
}

/** Grid, bed, and normal-flow start state for a 2-D playground run. */
function build2dSetup(p) {
  const manningN = p.manningSI / 60.0;
  const ax = uniformAxis(p.length, p.cells);
  const ay = uniformAxis(p.width, p.cellsY);
  const bed = compoundChannelBed({
    x: ax.centres,
    y: ay.centres,
    slopeX: p.slope,
    bankHeightM: p.bankHeight,
    halfWidthM: p.halfWidth,
    // Bank slope: the benches rise over a third of the remaining half-width.
    transitionM: Math.max((p.width / 2 - p.halfWidth) / 1.5, 1.0),
  });
  const start = normalFlowState({
    x: ax.centres,
    y: ay.centres,
    bed,
    slopeX: p.slope,
    channelDepthM: p.channelDepth,
    manningN,
  });
  return { ax, ay, bed, start, manningN };
}

/**
 * Estimated time steps for a 2-D run: the CFL step from the initial state,
 * corrected for the fact that a larger inflow deepens the flow and speeds the
 * waves up (depth ~ q^0.6, celerity ~ sqrt(depth), so dt ~ q^-0.3).
 */
function estimate2dSteps(p, setup) {
  const { ax, ay, start } = setup;
  const ny = p.cellsY;
  let maxRate = 0;
  for (let i = 0; i < p.cells; i += 1) {
    for (let j = 0; j < ny; j += 1) {
      const k = i * ny + j;
      const h = start.h[k];
      const u = h > 1e-10 ? start.hu[k] / h : 0;
      const celerity = Math.sqrt(G_M_PER_MIN2 * Math.max(h, 0));
      const rate = (Math.abs(u) + celerity) / ax.widths[i] + celerity / ay.widths[j];
      if (rate > maxRate) maxRate = rate;
    }
  }
  if (!(maxRate > 0)) return 1;
  const dt = CFL_2D / maxRate;
  const growth = Math.pow(Math.max(p.inflowMultiple, 1.0), 0.3);
  return Math.ceil((p.minutes / dt) * growth * 1.25);
}

function run(p, onStatus) {
  const recordInterval = Math.max(p.minutes / 120, 0.25);
  const rainRate = p.rainMmPerHour * MM_PER_HOUR_TO_M_PER_MIN;

  if (p.model === "saint_venant_2d") {
    const setup = build2dSetup(p);
    const estimated = estimate2dSteps(p, setup);
    const work = estimated * p.cells * p.cellsY;
    if (work > WORK_BUDGET_CELL_UPDATES) {
      throw new Error(
        `this 2-D run needs roughly ${(work / 1e6).toFixed(0)} million cell updates ` +
        `(~${(work / 4.5e6).toFixed(0)} s in the browser). Reduce the cell counts or the ` +
        `simulation length — halving either roughly halves the work.`,
      );
    }
    onStatus(`Running Saint-Venant 2-D (~${estimated} time steps)…`);
    const result = saintVenant2D({
      x_m: setup.ax.centres,
      y_m: setup.ay.centres,
      dx_m: setup.ax.widths,
      dy_m: setup.ay.widths,
      bedElevationM: setup.bed,
      manningN: setup.manningN,
      hInit: setup.start.h,
      huInit: setup.start.hu,
      tFinalMin: p.minutes,
      recordIntervalMin: Math.max(p.minutes / 40, 0.25),
      leftInflow: Float64Array.from(setup.start.inletDischarge, (v) => v * p.inflowMultiple),
      rainfall: (t) => (t < p.rainMinutes ? rainRate : 0.0),
      // Backstop in case the estimate is optimistic: fail loudly rather than
      // leaving the tab wedged.
      maxSteps: Math.ceil(estimated * 3),
    });
    return result;
  }

  const profile = buildProfile(p);
  if (p.model === "kinematic_wave") {
    const result = kinematicWave(profile, {
      tFinalMin: p.minutes,
      leftInflowFlux: p.inflow,
      recordIntervalMin: recordInterval,
      rainfallRateMPerMin: rainRate,
      rainfallStartMin: 0.0,
      rainfallEndMin: p.rainMinutes > 0 ? p.rainMinutes : 0.0,
    });
    const nLast = profile.manning_n[p.cells - 1];
    const sLast = profile.slope[p.cells - 1];
    result.downstream_q = result.depth_history.map((row) => manningFlux(row[p.cells - 1], sLast, nLast));
    return result;
  }

  onStatus("Running Saint-Venant (dynamic wave)…");
  const result = saintVenant1D({
    x_m: profile.station_m,
    dx_m: profile.dx_m,
    slope: profile.slope,
    manning_n: profile.manning_n,
    hInit: profile.initial_depth_m,
    tFinalMin: p.minutes,
    recordIntervalMin: recordInterval,
    leftInflow: p.inflow,
    rainfall: (x, t) => (t < p.rainMinutes ? rainRate : 0.0),
  });
  result.downstream_q = result.discharge_history.map((row) => row[p.cells - 1]);
  return result;
}

function show(id, visible) {
  document.getElementById(id).hidden = !visible;
}

function renderResults1d(result) {
  let peak = -Infinity;
  let peakT = 0;
  let peakS = 0;
  for (let t = 0; t < result.depth_history.length; t += 1) {
    const row = result.depth_history[t];
    for (let i = 0; i < row.length; i += 1) {
      if (row[i] > peak) {
        peak = row[i];
        peakT = result.times[t];
        peakS = result.station_m[i];
      }
    }
  }
  const storage = (depths) => depths.reduce((acc, d, i) => acc + d * result.dx_m[i], 0);
  const balanceError = storage(result.depth_final) - storage(result.depth_initial)
    - (result.mass_inflow + result.mass_source - result.mass_outflow);

  metricTiles(document.getElementById("playground-metrics"), [
    { label: "Peak depth", value: `${formatNumber(peak, 3)} m`,
      sub: `at ${formatNumber(peakS, 0)} m, ${formatMinutes(peakT)}` },
    { label: "Final downstream depth", value: `${formatNumber(result.depth_final[result.depth_final.length - 1], 3)} m` },
    { label: "Rain volume", value: `${formatNumber(result.mass_source, 2)} m²`,
      sub: "per m of channel width" },
    { label: "Mass-balance error", value: formatNumber(balanceError, 6), sub: "m² (numerical quality)" },
  ]);

  const depth = depthProfileChart(document.getElementById("pg-depth-chart"), {
    stations: result.station_m,
    depthHistory: result.depth_history,
  });
  document.getElementById("pg-hydrograph-title").textContent = "Downstream discharge";
  const hydro = hydrographChart(document.getElementById("pg-hydrograph-chart"), {
    times: result.times,
    values: result.downstream_q,
    unit: "discharge (m²/min)",
  });
  return (i) => {
    depth.setFrame(i);
    hydro.setCursor(result.times[i]);
  };
}

function renderResults2d(result, p) {
  const { nx, ny } = result;
  const area = (i, j) => result.dx_m[i] * result.dy_m[j];

  let peak = -Infinity;
  let peakT = 0;
  let peakI = 0;
  let peakJ = 0;
  for (let t = 0; t < result.depth_history.length; t += 1) {
    const frame = result.depth_history[t];
    for (let k = 0; k < frame.length; k += 1) {
      if (frame[k] > peak) {
        peak = frame[k];
        peakT = result.times[t];
        peakI = Math.floor(k / ny);
        peakJ = k % ny;
      }
    }
  }

  // Ground that starts dry and ends up under water.
  const startFrame = result.depth_history[0];
  let dryArea = 0;
  for (let i = 0; i < nx; i += 1) {
    for (let j = 0; j < ny; j += 1) if (startFrame[i * ny + j] <= 0) dryArea += area(i, j);
  }
  const floodedHistory = result.depth_history.map((frame) => {
    let flooded = 0;
    for (let i = 0; i < nx; i += 1) {
      for (let j = 0; j < ny; j += 1) {
        const k = i * ny + j;
        if (startFrame[k] <= 0 && frame[k] > FLOOD_DEPTH_M) flooded += area(i, j);
      }
    }
    return flooded;
  });
  const peakFlooded = Math.max(...floodedHistory);

  const volume = (frame) => {
    let total = 0;
    for (let i = 0; i < nx; i += 1) {
      for (let j = 0; j < ny; j += 1) total += frame[i * ny + j] * area(i, j);
    }
    return total;
  };
  const balanceError = volume(result.depth_final) - volume(result.depth_initial)
    - (result.mass_inflow + result.mass_source - result.mass_outflow + result.mass_floor_correction);

  const outletDischarge = result.discharge_x_history.map((frame) => {
    let total = 0;
    for (let j = 0; j < ny; j += 1) total += frame[(nx - 1) * ny + j] * result.dy_m[j];
    return total;
  });

  metricTiles(document.getElementById("playground-metrics"), [
    { label: "Peak depth", value: `${formatNumber(peak, 3)} m`,
      sub: `${formatNumber(result.x_m[peakI], 0)} m downstream, ${formatNumber(result.y_m[peakJ], 0)} m across, ${formatMinutes(peakT)}` },
    { label: "Floodplain under water", value: `${formatNumber(peakFlooded / 10000, 2)} ha`,
      sub: dryArea > 0 ? `${formatNumber((peakFlooded / dryArea) * 100, 0)}% of the dry ground` : "no dry ground in this section" },
    { label: "Time steps", value: `${result.steps}`, sub: `${nx}×${ny} cells` },
    { label: "Mass-balance error", value: formatNumber(balanceError, 4), sub: "m³ (numerical quality)" },
  ]);

  const grid = {
    nx,
    ny,
    x_m: result.x_m,
    y_m: result.y_m,
    bedElevation: result.bed_elevation_m,
  };
  const map = inundationMap(document.getElementById("pg-map-chart"), {
    grid,
    depthHistory: result.depth_history,
    wetTolM: FLOOD_DEPTH_M,
  });
  document.getElementById("pg-section-title").textContent =
    `Cross-section ${formatNumber(result.x_m[peakI], 0)} m downstream (through the deepest point)`;
  const section = crossSectionChart(document.getElementById("pg-section-chart"), {
    grid,
    depthHistory: result.depth_history,
    stationIndex: peakI,
    maxDepth: map.peakDepth,
  });
  document.getElementById("pg-hydrograph-title").textContent = "Discharge past the downstream end";
  const hydro = hydrographChart(document.getElementById("pg-hydrograph-chart"), {
    times: result.times,
    values: outletDischarge,
    unit: "discharge (m³/min)",
  });
  void p;
  return (i) => {
    map.setFrame(i);
    section.setFrame(i);
    hydro.setCursor(result.times[i]);
  };
}

function renderResults(result, p) {
  document.getElementById("playground-results").hidden = false;
  const isTwoD = p.model === "saint_venant_2d";
  show("pg-block-1d", !isTwoD);
  show("pg-block-map", isTwoD);
  show("pg-block-section", isTwoD);

  if (animation) animation.stop();
  const onFrame = isTwoD ? renderResults2d(result, p) : renderResults1d(result);
  animation = animator({
    frameCount: result.times.length,
    times: result.times,
    onFrame,
    playButton: document.getElementById("pg-play-toggle"),
    slider: document.getElementById("pg-frame-slider"),
    label: document.getElementById("pg-frame-label"),
  });
  animation.play();
}

// A 2-D run costs cells-along x cells-across x time steps, so the 1-D presets
// would blow the work budget on the very first click. Switching models moves
// the shared fields to that model's preset -- but only when they still hold the
// other preset's value, so a deliberate edit is never clobbered.
const PRESETS = {
  oneD: { length: 2000, cells: 80, minutes: 180 },
  twoD: { length: 1200, cells: 48, minutes: 90 },
};

function applyPreset(form, to, from) {
  for (const [name, value] of Object.entries(to)) {
    // Not form.elements[name]: a field named "length" resolves to the
    // collection's own length property instead of the input.
    const input = form.querySelector(`[name="${name}"]`);
    if (input && Number(input.value) === from[name]) input.value = String(value);
  }
}

function syncModelFields(form, { movePreset = false } = {}) {
  const model = new FormData(form).get("model");
  const isTwoD = model === "saint_venant_2d";
  form.querySelectorAll(".only-2d").forEach((node) => { node.hidden = !isTwoD; });
  form.querySelectorAll(".only-1d").forEach((node) => { node.hidden = isTwoD; });
  if (movePreset) {
    if (isTwoD) applyPreset(form, PRESETS.twoD, PRESETS.oneD);
    else applyPreset(form, PRESETS.oneD, PRESETS.twoD);
  }
}

export function initPlayground() {
  const form = document.getElementById("playground-form");
  const status = document.getElementById("playground-status");
  const button = document.getElementById("run-button");

  syncModelFields(form);

  // A field that fails native validation blocks submit without firing it, which
  // otherwise looks like the Run button doing nothing at all. `invalid` does not
  // bubble, so listen in the capture phase.
  form.addEventListener("invalid", (event) => {
    const field = event.target;
    status.className = "status error";
    status.textContent = `${field.name || "a field"}: ${field.validationMessage}`;
    field.focus();
  }, true);

  form.querySelectorAll('input[name="model"]').forEach((radio) => {
    radio.addEventListener("change", () => syncModelFields(form, { movePreset: true }));
  });

  form.onsubmit = (event) => {
    event.preventDefault();
    const params = readForm(form);
    const problems = validate(params);
    if (problems.length > 0) {
      status.className = "status error";
      status.textContent = problems.join("; ");
      return;
    }
    status.className = "status";
    status.textContent = "Running…";
    button.disabled = true;

    // Yield to the event loop so the busy state paints before the synchronous
    // solve. Deliberately NOT requestAnimationFrame: rAF never fires while the
    // tab is hidden, which would leave the run stuck at "Running…".
    setTimeout(() => {
      try {
        const started = performance.now();
        const result = run(params, (msg) => { status.textContent = msg; });
        const elapsed = performance.now() - started;
        renderResults(result, params);
        const size = params.model === "saint_venant_2d"
          ? `${params.cells}×${params.cellsY} cells`
          : `${params.cells} cells`;
        status.className = "status";
        status.textContent =
          `Done in ${formatNumber(elapsed / 1000, 2)} s — ${result.times.length} frames, ${size} (${params.model}).`;
      } catch (error) {
        status.className = "status error";
        status.textContent = `Run failed: ${error.message}`;
      } finally {
        button.disabled = false;
      }
    }, 30);
  };
}
