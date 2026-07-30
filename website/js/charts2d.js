/* 2-D views: a plan-view inundation map and an animated cross-section.
 * Canvas for the map (one fill per grid cell per frame), inline SVG for the
 * cross-section, matching the rest of the site. No dependencies. */

import { formatNumber } from "./charts.js";

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(name, attrs, parent) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (parent) parent.appendChild(node);
  return node;
}

function lerp(a, b, f) {
  return a + (b - a) * f;
}

function mixHex(c1, c2, f) {
  const r = Math.round(lerp(c1[0], c2[0], f));
  const g = Math.round(lerp(c1[1], c2[1], f));
  const b = Math.round(lerp(c1[2], c2[2], f));
  return `rgb(${r},${g},${b})`;
}

// Shallow -> deep. sqrt scaling so shallow sheet flow stays visible.
const WATER_STOPS = [[198, 231, 246], [77, 160, 209], [16, 76, 124], [7, 32, 61]];
const LAND_LOW = [214, 205, 178];
const LAND_HIGH = [156, 146, 116];

export function waterColour(depth, maxDepth) {
  const f = Math.min(Math.max(Math.sqrt(depth / maxDepth), 0), 1);
  const span = WATER_STOPS.length - 1;
  const scaled = f * span;
  const i = Math.min(Math.floor(scaled), span - 1);
  return mixHex(WATER_STOPS[i], WATER_STOPS[i + 1], scaled - i);
}

/**
 * Plan-view inundation map.
 *
 * container: element to fill.
 * grid: { nx, ny, x_m, y_m, dx_m, dy_m, bedElevation (flat nx*ny) }
 * depthHistory: array of flat nx*ny depth fields (index i*ny + j)
 * wetTolM: depth at or below which a cell is drawn as dry ground.
 *
 * Returns { setFrame(i) }.
 */
export function inundationMap(container, { grid, depthHistory, wetTolM = 0.02, maxDepth = null }) {
  container.innerHTML = "";
  const { nx, ny } = grid;
  const bed = grid.bedElevation;

  let peak = maxDepth;
  if (peak === null) {
    peak = 0;
    for (const frame of depthHistory) for (let k = 0; k < frame.length; k += 1) {
      if (frame[k] > peak) peak = frame[k];
    }
  }
  peak = Math.max(peak, 1e-6);

  // Land shading uses each column's height ABOVE THAT COLUMN'S LOW POINT, so
  // the picture shows the cross-section shape rather than the downstream slope.
  const relief = new Float64Array(nx * ny);
  for (let i = 0; i < nx; i += 1) {
    let low = Infinity;
    let high = -Infinity;
    for (let j = 0; j < ny; j += 1) {
      const v = bed[i * ny + j];
      if (v < low) low = v;
      if (v > high) high = v;
    }
    const span = Math.max(high - low, 1e-9);
    for (let j = 0; j < ny; j += 1) relief[i * ny + j] = (bed[i * ny + j] - low) / span;
  }

  const wrap = document.createElement("div");
  wrap.className = "map2d";
  container.appendChild(wrap);

  const cellPx = 14;
  const width = nx * cellPx;
  const height = ny * cellPx;
  const canvas = document.createElement("canvas");
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.aspectRatio = `${width} / ${height}`;
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", "Plan view of water depth over the reach");
  wrap.appendChild(canvas);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);

  // The frame-0 waterline: the edge of the channel before the event. Drawn as
  // a reference outline so you can see what is NEW water.
  const baseline = depthHistory[0];
  const segments = [];
  for (let i = 0; i < nx; i += 1) {
    for (let j = 0; j < ny; j += 1) {
      const wet = baseline[i * ny + j] > wetTolM;
      if (!wet) continue;
      const x = i * cellPx;
      const y = j * cellPx;
      if (j === 0 || baseline[i * ny + j - 1] <= wetTolM) segments.push([x, y, x + cellPx, y]);
      if (j === ny - 1 || baseline[i * ny + j + 1] <= wetTolM) {
        segments.push([x, y + cellPx, x + cellPx, y + cellPx]);
      }
      if (i === 0 || baseline[(i - 1) * ny + j] <= wetTolM) segments.push([x, y, x, y + cellPx]);
      if (i === nx - 1 || baseline[(i + 1) * ny + j] <= wetTolM) {
        segments.push([x + cellPx, y, x + cellPx, y + cellPx]);
      }
    }
  }

  let current = 0;

  function draw(index) {
    current = index;
    const frame = depthHistory[index];
    for (let i = 0; i < nx; i += 1) {
      for (let j = 0; j < ny; j += 1) {
        const depth = frame[i * ny + j];
        ctx.fillStyle = depth > wetTolM
          ? waterColour(depth, peak)
          : mixHex(LAND_LOW, LAND_HIGH, relief[i * ny + j]);
        ctx.fillRect(i * cellPx, j * cellPx, cellPx, cellPx);
      }
    }
    ctx.strokeStyle = "rgba(180, 60, 40, 0.75)";
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    for (const [x1, y1, x2, y2] of segments) {
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
    ctx.stroke();
  }

  // Hover readout: which cell, and how deep.
  const readout = document.createElement("div");
  readout.className = "map2d-readout";
  readout.textContent = "Hover the map to read a depth";
  wrap.appendChild(readout);

  canvas.addEventListener("mousemove", (event) => {
    const rect = canvas.getBoundingClientRect();
    const i = Math.min(nx - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * nx)));
    const j = Math.min(ny - 1, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * ny)));
    const depth = depthHistory[current][i * ny + j];
    readout.textContent = depth > wetTolM
      ? `${formatNumber(grid.x_m[i], 0)} m downstream, ${formatNumber(grid.y_m[j], 0)} m across — ${formatNumber(depth, 2)} m deep`
      : `${formatNumber(grid.x_m[i], 0)} m downstream, ${formatNumber(grid.y_m[j], 0)} m across — dry`;
  });
  canvas.addEventListener("mouseleave", () => {
    readout.textContent = "Hover the map to read a depth";
  });

  // Legend: depth ramp plus the two non-depth things on the map.
  const legend = document.createElement("div");
  legend.className = "map2d-legend";
  const ramp = [];
  for (let s = 0; s <= 24; s += 1) {
    ramp.push(`${waterColour((s / 24) * peak, peak)} ${(s / 24) * 100}%`);
  }
  legend.innerHTML = `
    <span class="legend-scale">
      <span class="legend-bar" style="background:linear-gradient(to right, ${ramp.join(",")})"></span>
      <span class="legend-ends"><span>0 m</span><span>${formatNumber(peak, 2)} m</span></span>
    </span>
    <span class="legend-key"><span class="swatch land"></span>dry ground</span>
    <span class="legend-key"><span class="swatch outline"></span>channel edge before the event</span>`;
  wrap.appendChild(legend);

  draw(0);
  return { setFrame: draw, peakDepth: peak };
}

/**
 * Animated cross-section at one station: ground profile plus water surface.
 *
 * Returns { setFrame(i) }.
 */
export function crossSectionChart(container, { grid, depthHistory, stationIndex, maxDepth }) {
  container.innerHTML = "";
  const { nx, ny } = grid;
  const i = Math.min(Math.max(stationIndex, 0), nx - 1);
  const bed = grid.bedElevation;

  const W = 960;
  const H = 260;
  const M = { top: 16, right: 18, bottom: 40, left: 66 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  const bedRow = new Float64Array(ny);
  for (let j = 0; j < ny; j += 1) bedRow[j] = bed[i * ny + j];
  let low = Infinity;
  let high = -Infinity;
  for (let j = 0; j < ny; j += 1) {
    if (bedRow[j] < low) low = bedRow[j];
    if (bedRow[j] > high) high = bedRow[j];
  }
  const top = high + Math.max(maxDepth, 0.2) * 1.1;
  const span = Math.max(top - low, 1e-6);

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
  const yValues = grid.y_m;
  const across0 = yValues[0];
  const acrossSpan = Math.max(yValues[ny - 1] - across0, 1e-9);
  const toX = (yv) => M.left + ((yv - across0) / acrossSpan) * plotW;
  const toY = (elev) => M.top + plotH - ((elev - low) / span) * plotH;

  for (let step = 0; step <= 4; step += 1) {
    const elev = low + (span * step) / 4;
    const y = toY(elev);
    svgEl("line", { x1: M.left, x2: M.left + plotW, y1: y, y2: y, stroke: "#e3e9ec" }, svg);
    const label = svgEl("text", { x: M.left - 8, y: y + 4, "text-anchor": "end",
      "font-size": 12, fill: "#51636d" }, svg);
    label.textContent = `${formatNumber(elev - low, 1)} m`;
  }
  const yTitle = svgEl("text", { x: 16, y: M.top + plotH / 2, "font-size": 12, fill: "#51636d",
    transform: `rotate(-90 16 ${M.top + plotH / 2})`, "text-anchor": "middle" }, svg);
  yTitle.textContent = "elevation above channel bed (m)";
  const xTitle = svgEl("text", { x: M.left + plotW / 2, y: H - 6, "text-anchor": "middle",
    "font-size": 12, fill: "#51636d" }, svg);
  xTitle.textContent = "distance across the valley (m)";

  const water = svgEl("polygon", { fill: "rgba(46, 132, 190, 0.55)", stroke: "#0d4f77",
    "stroke-width": 1.6 }, svg);
  const ground = svgEl("polygon", { fill: "#c8bd9b", stroke: "#8d8261", "stroke-width": 1.2 }, svg);
  const bedPoints = [];
  for (let j = 0; j < ny; j += 1) bedPoints.push(`${toX(yValues[j]).toFixed(1)},${toY(bedRow[j]).toFixed(1)}`);
  const baseY = (M.top + plotH).toFixed(1);
  ground.setAttribute(
    "points",
    `${toX(yValues[0]).toFixed(1)},${baseY} ${bedPoints.join(" ")} ${toX(yValues[ny - 1]).toFixed(1)},${baseY}`,
  );

  function setFrame(index) {
    const frame = depthHistory[index];
    // Water polygon: surface across the top, bed back along the bottom, so dry
    // cells pinch the polygon shut instead of drawing water over land.
    const surface = [];
    const under = [];
    for (let j = 0; j < ny; j += 1) {
      const depth = frame[i * ny + j];
      surface.push(`${toX(yValues[j]).toFixed(1)},${toY(bedRow[j] + depth).toFixed(1)}`);
      under.push(`${toX(yValues[ny - 1 - j]).toFixed(1)},${toY(bedRow[ny - 1 - j]).toFixed(1)}`);
    }
    water.setAttribute("points", `${surface.join(" ")} ${under.join(" ")}`);
  }

  setFrame(0);
  return { setFrame, stationIndex: i };
}
