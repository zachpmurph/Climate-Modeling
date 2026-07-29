/* Playground tab: run the parity-tested JS solver ports on user-defined
 * conditions, in the browser. */

import { kinematicWave, saintVenant1D, manningFlux } from "./solvers.js";
import {
  animator,
  depthProfileChart,
  formatMinutes,
  formatNumber,
  hydrographChart,
  metricTiles,
} from "./charts.js";

const MM_PER_HOUR_TO_M_PER_MIN = 1 / 1000 / 60;
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
  };
}

function validate(p) {
  const problems = [];
  if (!(p.length >= 100 && p.length <= 100000)) problems.push("reach length must be 100–100000 m");
  if (!(p.cells >= 10 && p.cells <= 400)) problems.push("cells must be 10–400");
  if (!(p.slope >= 5e-5 && p.slope <= 0.1)) problems.push("slope must be 5e-5–0.1");
  if (!(p.manningSI >= 0.01 && p.manningSI <= 0.15)) problems.push("Manning n must be 0.01–0.15 (SI)");
  if (!(p.depth0 >= 0 && p.depth0 <= 10)) problems.push("initial depth must be 0–10 m");
  if (!(p.inflow >= 0 && p.inflow <= 2000)) problems.push("inflow must be 0–2000 m²/min");
  if (!(p.rainMmPerHour >= 0 && p.rainMmPerHour <= 150)) problems.push("rain must be 0–150 mm/h");
  if (!(p.minutes >= 5 && p.minutes <= 1440)) problems.push("duration must be 5–1440 min");
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

function run(p, onStatus) {
  const profile = buildProfile(p);
  const rainRate = p.rainMmPerHour * MM_PER_HOUR_TO_M_PER_MIN;
  const recordInterval = Math.max(p.minutes / 120, 0.25);

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

function renderResults(result, p) {
  const holder = document.getElementById("playground-results");
  holder.hidden = false;

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

  if (animation) animation.stop();
  const depth = depthProfileChart(document.getElementById("pg-depth-chart"), {
    stations: result.station_m,
    depthHistory: result.depth_history,
  });
  const hydro = hydrographChart(document.getElementById("pg-hydrograph-chart"), {
    times: result.times,
    values: result.downstream_q,
    unit: "discharge (m²/min)",
  });
  animation = animator({
    frameCount: result.times.length,
    times: result.times,
    onFrame: (i) => {
      depth.setFrame(i);
      hydro.setCursor(result.times[i]);
    },
    playButton: document.getElementById("pg-play-toggle"),
    slider: document.getElementById("pg-frame-slider"),
    label: document.getElementById("pg-frame-label"),
  });
  animation.play();
}

export function initPlayground() {
  const form = document.getElementById("playground-form");
  const status = document.getElementById("playground-status");
  const button = document.getElementById("run-button");

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
        status.textContent =
          `Done in ${formatNumber(elapsed / 1000, 2)} s — ${result.times.length} frames, ${params.cells} cells (${params.model}).`;
      } catch (error) {
        status.className = "status error";
        status.textContent = `Run failed: ${error.message}`;
      } finally {
        button.disabled = false;
      }
    }, 30);
  };
}
