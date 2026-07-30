/* Shared SVG chart components: animated depth profile, hydrograph, metric tiles,
 * and a frame animator. No dependencies. */

const SVG_NS = "http://www.w3.org/2000/svg";

function el(name, attrs, parent) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (parent) parent.appendChild(node);
  return node;
}

function niceTicks(maxValue, count = 5) {
  if (!(maxValue > 0)) return [0];
  const rough = maxValue / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => maxValue / s <= count) || magnitude * 10;
  const ticks = [];
  for (let v = 0; v <= maxValue + 1e-12; v += step) ticks.push(v);
  return ticks;
}

export function formatNumber(value, digits = 2) {
  if (!Number.isFinite(value)) return "–";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 0.01 || abs >= 100000)) return value.toExponential(1);
  return value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function formatMinutes(min) {
  if (min < 120) return `${formatNumber(min, 1)} min`;
  return `${formatNumber(min / 60, 1)} h`;
}

export function metricTiles(container, items) {
  container.innerHTML = "";
  for (const item of items) {
    const tile = document.createElement("div");
    tile.className = "metric";
    tile.innerHTML = `<span class="label"></span><span class="value"></span> <span class="sub"></span>`;
    tile.querySelector(".label").textContent = item.label;
    tile.querySelector(".value").textContent = item.value;
    tile.querySelector(".sub").textContent = item.sub || "";
    container.appendChild(tile);
  }
}

/** Animated depth-along-reach chart. Returns { setFrame(i) }. */
export function depthProfileChart(container, { stations, depthHistory }) {
  container.innerHTML = "";
  const W = 960;
  const H = 320;
  const M = { top: 16, right: 18, bottom: 40, left: 62 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  let maxDepth = 0;
  for (const row of depthHistory) for (const v of row) if (v > maxDepth) maxDepth = v;
  maxDepth = Math.max(maxDepth * 1.08, 1e-6);
  const x0 = stations[0];
  const xSpan = Math.max(stations[stations.length - 1] - x0, 1e-9);
  const useKm = xSpan > 3000;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
  const toX = (s) => M.left + ((s - x0) / xSpan) * plotW;
  const toY = (d) => M.top + plotH - (d / maxDepth) * plotH;

  for (const tick of niceTicks(maxDepth, 4)) {
    const y = toY(tick);
    el("line", { x1: M.left, x2: M.left + plotW, y1: y, y2: y, stroke: "#e3e9ec", "stroke-width": 1 }, svg);
    const label = el("text", { x: M.left - 8, y: y + 4, "text-anchor": "end", "font-size": 12, fill: "#51636d" }, svg);
    label.textContent = formatNumber(tick, 2);
  }
  for (const tick of niceTicks(xSpan, 6)) {
    const x = toX(x0 + tick);
    const label = el("text", { x, y: H - 14, "text-anchor": "middle", "font-size": 12, fill: "#51636d" }, svg);
    label.textContent = useKm ? `${formatNumber(tick / 1000, 0)} km` : `${formatNumber(tick, 0)} m`;
  }
  const yTitle = el("text", { x: 16, y: M.top + plotH / 2, "font-size": 12, fill: "#51636d",
    transform: `rotate(-90 16 ${M.top + plotH / 2})`, "text-anchor": "middle" }, svg);
  yTitle.textContent = "depth (m)";
  const xTitle = el("text", { x: M.left + plotW / 2, y: H - 1, "text-anchor": "middle", "font-size": 12, fill: "#51636d" }, svg);
  xTitle.textContent = "distance downstream";

  const initialPath = el("polyline", { fill: "none", stroke: "#9db8c4", "stroke-width": 1.4, "stroke-dasharray": "5 4" }, svg);
  const area = el("polygon", { fill: "rgba(30, 136, 184, 0.28)", stroke: "none" }, svg);
  const line = el("polyline", { fill: "none", stroke: "#0d4f77", "stroke-width": 2.2 }, svg);

  const pointString = (row) => stations.map((s, i) => `${toX(s).toFixed(1)},${toY(row[i]).toFixed(1)}`).join(" ");
  initialPath.setAttribute("points", pointString(depthHistory[0]));
  const baseY = (M.top + plotH).toFixed(1);

  function setFrame(index) {
    const row = depthHistory[index];
    const points = pointString(row);
    line.setAttribute("points", points);
    area.setAttribute(
      "points",
      `${toX(stations[0]).toFixed(1)},${baseY} ${points} ${toX(stations[stations.length - 1]).toFixed(1)},${baseY}`,
    );
  }
  setFrame(0);
  return { setFrame };
}

/** Static line chart with a movable time cursor. Returns { setCursor(t) }. */
export function hydrographChart(container, { times, values, unit, minAxisMax = 1e-9 }) {
  container.innerHTML = "";
  const W = 960;
  const H = 220;
  const M = { top: 14, right: 18, bottom: 36, left: 70 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  // minAxisMax keeps an all-zero series (e.g. "nothing flooded") on a readable
  // axis instead of one labelled in billionths.
  const maxValue = Math.max(...values, minAxisMax) * 1.08;
  const tSpan = Math.max(times[times.length - 1] - times[0], 1e-9);
  const useHours = tSpan > 180;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
  const toX = (t) => M.left + ((t - times[0]) / tSpan) * plotW;
  const toY = (v) => M.top + plotH - (v / maxValue) * plotH;

  for (const tick of niceTicks(maxValue, 4)) {
    const y = toY(tick);
    el("line", { x1: M.left, x2: M.left + plotW, y1: y, y2: y, stroke: "#e3e9ec" }, svg);
    const label = el("text", { x: M.left - 8, y: y + 4, "text-anchor": "end", "font-size": 12, fill: "#51636d" }, svg);
    label.textContent = formatNumber(tick, 1);
  }
  for (const tick of niceTicks(tSpan, 6)) {
    const x = toX(times[0] + tick);
    const label = el("text", { x, y: H - 12, "text-anchor": "middle", "font-size": 12, fill: "#51636d" }, svg);
    label.textContent = useHours ? `${formatNumber(tick / 60, 0)} h` : `${formatNumber(tick, 0)} min`;
  }
  const yTitle = el("text", { x: 16, y: M.top + plotH / 2, "font-size": 12, fill: "#51636d",
    transform: `rotate(-90 16 ${M.top + plotH / 2})`, "text-anchor": "middle" }, svg);
  yTitle.textContent = unit;

  const line = el("polyline", {
    fill: "none", stroke: "#0b7285", "stroke-width": 2,
    points: times.map((t, i) => `${toX(t).toFixed(1)},${toY(values[i]).toFixed(1)}`).join(" "),
  }, svg);
  void line;
  const cursor = el("line", { x1: toX(times[0]), x2: toX(times[0]), y1: M.top, y2: M.top + plotH,
    stroke: "#c0392b", "stroke-width": 1.4, "stroke-dasharray": "4 3" }, svg);

  function setCursor(t) {
    const x = toX(t).toFixed(1);
    cursor.setAttribute("x1", x);
    cursor.setAttribute("x2", x);
  }
  return { setCursor };
}

/** Two or more named series on shared axes, with a legend. Static (no cursor). */
export function multiSeriesChart(container, { times, series, unit }) {
  container.innerHTML = "";
  const W = 960;
  const H = 260;
  const M = { top: 30, right: 18, bottom: 36, left: 78 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  let maxValue = 0;
  for (const s of series) for (const v of s.values) if (v > maxValue) maxValue = v;
  maxValue = Math.max(maxValue * 1.08, 1e-9);
  const tSpan = Math.max(times[times.length - 1] - times[0], 1e-9);
  const useHours = tSpan > 180;

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
  const toX = (t) => M.left + ((t - times[0]) / tSpan) * plotW;
  const toY = (v) => M.top + plotH - (v / maxValue) * plotH;

  for (const tick of niceTicks(maxValue, 4)) {
    const y = toY(tick);
    el("line", { x1: M.left, x2: M.left + plotW, y1: y, y2: y, stroke: "#e3e9ec" }, svg);
    const label = el("text", { x: M.left - 8, y: y + 4, "text-anchor": "end", "font-size": 12,
      fill: "#51636d" }, svg);
    label.textContent = formatNumber(tick, 0);
  }
  for (const tick of niceTicks(tSpan, 6)) {
    const label = el("text", { x: toX(times[0] + tick), y: H - 12, "text-anchor": "middle",
      "font-size": 12, fill: "#51636d" }, svg);
    label.textContent = useHours ? `${formatNumber(tick / 60, 0)} h` : `${formatNumber(tick, 0)} min`;
  }
  const yTitle = el("text", { x: 16, y: M.top + plotH / 2, "font-size": 12, fill: "#51636d",
    transform: `rotate(-90 16 ${M.top + plotH / 2})`, "text-anchor": "middle" }, svg);
  yTitle.textContent = unit;

  let legendX = M.left;
  for (const s of series) {
    el("polyline", {
      fill: "none", stroke: s.color, "stroke-width": s.width || 2,
      "stroke-dasharray": s.dash || "none",
      points: times.map((t, i) => `${toX(t).toFixed(1)},${toY(s.values[i]).toFixed(1)}`).join(" "),
    }, svg);
    el("line", { x1: legendX, x2: legendX + 22, y1: 14, y2: 14, stroke: s.color,
      "stroke-width": s.width || 2, "stroke-dasharray": s.dash || "none" }, svg);
    const label = el("text", { x: legendX + 28, y: 18, "font-size": 12, fill: "#51636d" }, svg);
    label.textContent = s.label;
    legendX += 40 + s.label.length * 7.1;
  }
}

/**
 * Observed against modelled on a shared square scale, with the 1:1 line.
 * Points below the diagonal are under-predictions -- the pattern a pair of
 * hydrograph lines makes you infer but never shows directly.
 */
export function scatterChart(container, { observed, predicted, unit }) {
  container.innerHTML = "";
  const W = 520;
  const H = 460;
  const M = { top: 16, right: 16, bottom: 46, left: 74 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;

  let lo = Infinity;
  let hi = -Infinity;
  for (const v of observed.concat(predicted)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const pad = Math.max((hi - lo) * 0.06, 1e-9);
  lo = Math.max(0, lo - pad);
  hi += pad;
  const span = Math.max(hi - lo, 1e-9);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
  const toX = (v) => M.left + ((v - lo) / span) * plotW;
  const toY = (v) => M.top + plotH - ((v - lo) / span) * plotH;

  for (let s = 0; s <= 4; s += 1) {
    const value = lo + (span * s) / 4;
    el("line", { x1: M.left, x2: M.left + plotW, y1: toY(value), y2: toY(value), stroke: "#e3e9ec" }, svg);
    el("line", { x1: toX(value), x2: toX(value), y1: M.top, y2: M.top + plotH, stroke: "#e3e9ec" }, svg);
    const yl = el("text", { x: M.left - 8, y: toY(value) + 4, "text-anchor": "end", "font-size": 11,
      fill: "#51636d" }, svg);
    yl.textContent = formatNumber(value, 0);
    const xl = el("text", { x: toX(value), y: M.top + plotH + 18, "text-anchor": "middle",
      "font-size": 11, fill: "#51636d" }, svg);
    xl.textContent = formatNumber(value, 0);
  }
  el("line", { x1: toX(lo), y1: toY(lo), x2: toX(hi), y2: toY(hi), stroke: "#51636d",
    "stroke-width": 1.4, "stroke-dasharray": "6 4" }, svg);
  const oneToOne = el("text", { x: toX(hi) - 6, y: toY(hi) + 16, "text-anchor": "end",
    "font-size": 11, fill: "#51636d" }, svg);
  oneToOne.textContent = "1:1";

  for (let i = 0; i < observed.length; i += 1) {
    el("circle", { cx: toX(observed[i]).toFixed(1), cy: toY(predicted[i]).toFixed(1), r: 3.1,
      fill: "rgba(13, 79, 119, 0.55)", stroke: "#0d4f77", "stroke-width": 0.7 }, svg);
  }
  const xTitle = el("text", { x: M.left + plotW / 2, y: H - 8, "text-anchor": "middle",
    "font-size": 12, fill: "#51636d" }, svg);
  xTitle.textContent = `observed ${unit}`;
  const yTitle = el("text", { x: 16, y: M.top + plotH / 2, "font-size": 12, fill: "#51636d",
    transform: `rotate(-90 16 ${M.top + plotH / 2})`, "text-anchor": "middle" }, svg);
  yTitle.textContent = `modelled ${unit}`;
}

/**
 * Diverging bars about a zero baseline. compareBars assumes positive values;
 * Nash-Sutcliffe can go negative, and that case is the one worth seeing.
 */
export function divergingBars(container, rows, currentId) {
  container.innerHTML = "";
  const limit = Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
  for (const row of rows) {
    const div = document.createElement("div");
    div.className = "diverge-row" + (row.id === currentId ? " current" : "");
    const half = (Math.abs(row.value) / limit) * 50;
    const negative = row.value < 0;
    div.innerHTML = `<span class="diverge-name"></span>
      <span class="diverge-track"><span class="diverge-zero"></span>
        <span class="diverge-fill ${negative ? "neg" : "pos"}"
              style="width:${half}%;${negative ? `right:50%` : `left:50%`}"></span></span>
      <span class="mono"></span>`;
    div.querySelector(".diverge-name").textContent = row.name;
    div.querySelector(".mono").textContent = row.display;
    container.appendChild(div);
  }
}

/** Horizontal comparison bars (peak depth per event). */
export function compareBars(container, rows, currentId) {
  container.innerHTML = "";
  const maxValue = Math.max(...rows.map((r) => r.value), 1e-9);
  for (const row of rows) {
    const div = document.createElement("div");
    div.className = "compare-row" + (row.id === currentId ? " current" : "");
    const width = Math.max((row.value / maxValue) * 100, 1);
    div.innerHTML = `<span class="compare-name"></span>
      <span class="compare-track"><span class="compare-fill" style="width:${width}%"></span></span>
      <span class="mono"></span>`;
    div.querySelector(".compare-name").textContent = row.name;
    div.querySelector(".mono").textContent = `${formatNumber(row.value, 2)} m`;
    container.appendChild(div);
  }
}

/** Frame animator wired to a play button, range slider, and label. */
export function animator({ frameCount, times, onFrame, playButton, slider, label, fps = 14 }) {
  let playing = false;
  let frame = 0;
  let lastTick = 0;
  slider.max = String(frameCount - 1);
  slider.value = "0";

  function apply(index) {
    frame = Math.max(0, Math.min(frameCount - 1, index));
    slider.value = String(frame);
    label.textContent = `t = ${formatMinutes(times[frame])}`;
    onFrame(frame);
  }

  function loop(timestamp) {
    if (!playing) return;
    if (timestamp - lastTick >= 1000 / fps) {
      lastTick = timestamp;
      apply(frame + 1 >= frameCount ? 0 : frame + 1);
    }
    requestAnimationFrame(loop);
  }

  function setPlaying(next) {
    playing = next;
    playButton.textContent = playing ? "❚❚" : "▶";
    if (playing) requestAnimationFrame(loop);
  }

  playButton.onclick = () => setPlaying(!playing);
  slider.oninput = () => {
    setPlaying(false);
    apply(Number(slider.value));
  };

  apply(0);
  return {
    stop: () => setPlaying(false),
    play: () => setPlaying(true),
  };
}
