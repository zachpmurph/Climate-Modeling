"""Render a self-contained report from a saved 2-D uncertainty ensemble."""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EnsembleFields:
    x_m: np.ndarray
    y_m: np.ndarray
    quantiles: np.ndarray
    peak_depth_quantiles_m: np.ndarray
    wet_probability: np.ndarray
    maximum_wet_area_quantiles_m2: np.ndarray
    parameter_names: tuple[str, ...]
    parameter_scales: np.ndarray
    member_mass_balance_error_m3: np.ndarray


def load_ensemble_fields(path):
    required = {
        "x_m",
        "y_m",
        "quantiles",
        "peak_depth_quantiles_m",
        "wet_probability",
        "maximum_wet_area_quantiles_m2",
        "parameter_names",
        "parameter_scales",
        "member_mass_balance_error_m3",
    }
    with np.load(path) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(
                f"Ensemble artifact is missing: {sorted(missing)}"
            )
        fields = EnsembleFields(
            x_m=np.asarray(data["x_m"], dtype=float),
            y_m=np.asarray(data["y_m"], dtype=float),
            quantiles=np.asarray(data["quantiles"], dtype=float),
            peak_depth_quantiles_m=np.asarray(
                data["peak_depth_quantiles_m"], dtype=float
            ),
            wet_probability=np.asarray(data["wet_probability"], dtype=float),
            maximum_wet_area_quantiles_m2=np.asarray(
                data["maximum_wet_area_quantiles_m2"], dtype=float
            ),
            parameter_names=tuple(str(value) for value in data["parameter_names"]),
            parameter_scales=np.asarray(data["parameter_scales"], dtype=float),
            member_mass_balance_error_m3=np.asarray(
                data["member_mass_balance_error_m3"], dtype=float
            ),
        )
    expected_field = (len(fields.x_m), len(fields.y_m))
    if fields.wet_probability.shape != expected_field:
        raise ValueError(f"wet_probability must have shape {expected_field}")
    expected_quantiles = (len(fields.quantiles), *expected_field)
    if fields.peak_depth_quantiles_m.shape != expected_quantiles:
        raise ValueError(
            f"peak_depth_quantiles_m must have shape {expected_quantiles}"
        )
    if fields.maximum_wet_area_quantiles_m2.shape != fields.quantiles.shape:
        raise ValueError("Wet-area quantiles do not match quantile probabilities")
    if fields.parameter_scales.shape != (
        len(fields.member_mass_balance_error_m3),
        len(fields.parameter_names),
    ):
        raise ValueError("Member parameter matrix shape is inconsistent")
    numeric = (
        fields.x_m,
        fields.y_m,
        fields.quantiles,
        fields.peak_depth_quantiles_m,
        fields.wet_probability,
        fields.maximum_wet_area_quantiles_m2,
        fields.parameter_scales,
        fields.member_mass_balance_error_m3,
    )
    if any(np.any(~np.isfinite(values)) for values in numeric):
        raise ValueError("Ensemble artifact contains non-finite values")
    if (
        np.any(np.diff(fields.x_m) <= 0.0)
        or np.any(np.diff(fields.y_m) <= 0.0)
        or np.any(np.diff(fields.quantiles) <= 0.0)
        or np.any(fields.wet_probability < 0.0)
        or np.any(fields.wet_probability > 1.0)
        or np.any(fields.peak_depth_quantiles_m < 0.0)
    ):
        raise ValueError("Ensemble coordinates, probabilities, or depths are invalid")
    return fields


def _json_for_script(value):
    return json.dumps(
        value, separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")


def render_uncertainty_report(fields, summary, title):
    sample_count = len(fields.parameter_scales)
    median_index = int(np.argmin(np.abs(fields.quantiles - 0.5)))
    median_area = fields.maximum_wet_area_quantiles_m2[median_index]
    max_probability = float(np.max(fields.wet_probability))
    wet_threshold = summary.get("wet_depth_threshold_m")
    threshold_label = (
        "the configured wet threshold"
        if wet_threshold is None
        else f"{float(wet_threshold):g} m"
    )
    parameter_rows = "".join(
        "<tr><th>{}</th><td>{:.3f}–{:.3f}</td></tr>".format(
            html.escape(name.replace("_", " ").title()),
            float(np.min(fields.parameter_scales[:, index])),
            float(np.max(fields.parameter_scales[:, index])),
        )
        for index, name in enumerate(fields.parameter_names)
    )
    payload = {
        "x": fields.x_m.tolist(),
        "y": fields.y_m.tolist(),
        "quantiles": fields.quantiles.tolist(),
        "peakDepth": fields.peak_depth_quantiles_m.tolist(),
        "wetProbability": fields.wet_probability.tolist(),
    }
    interpretation = summary.get(
        "interpretation",
        "Probabilities are conditional on the sampled ranges and model structure.",
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--bg:#f3f6f7;--panel:#fff;--text:#14232b;--muted:#5c6d76;--border:#d6e0e4}}
@media(prefers-color-scheme:dark){{:root{{--bg:#10191e;--panel:#17242b;--text:#eef5f7;--muted:#a9bbc4;--border:#34464f}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,sans-serif}}
main{{max-width:1050px;margin:auto;padding:28px 20px 48px}}h1{{margin-bottom:4px}}.muted{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}}
.card,.panel{{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:17px}}
.value{{font-size:1.45rem;font-weight:650;margin-top:4px}}.panel{{margin-top:14px}}
canvas{{width:100%;height:auto;image-rendering:pixelated}}select{{font:inherit;padding:7px;margin-bottom:10px}}
table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:8px;border-bottom:1px solid var(--border)}}
th{{color:var(--muted);font-weight:500}}.warning{{border-left:5px solid #d99b20}}
</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="muted">Conditional 2-D uncertainty screening · not a deterministic flood boundary</p>
<div class="cards">
<div class="card"><div class="muted">Ensemble members</div><div class="value">{sample_count}</div></div>
<div class="card"><div class="muted">Median maximum wet area</div><div class="value">{median_area:,.1f} m²</div></div>
<div class="card"><div class="muted">Highest probability above {html.escape(threshold_label)}</div><div class="value">{100*max_probability:.1f}%</div></div>
<div class="card"><div class="muted">Worst absolute mass residual</div><div class="value">{np.max(np.abs(fields.member_mass_balance_error_m3)):.3e} m³</div></div>
</div>
<section class="panel">
<h2>Spatial outcome band</h2>
<label for="metric">Displayed field</label>
<select id="metric"><option value="wet">Probability depth exceeded {html.escape(threshold_label)}</option>
{"".join(f'<option value="q{index}">Peak-depth quantile {probability:.0%}</option>' for index, probability in enumerate(fields.quantiles))}
</select>
<canvas id="map" width="960" height="470" role="img" aria-label="Plan-view uncertainty map"></canvas>
</section>
<section class="panel"><h2>Sampled parameter values</h2><table><tbody>{parameter_rows}</tbody></table></section>
<section class="panel warning"><h2>Interpretation</h2><p>{html.escape(str(interpretation))}</p>
<p class="muted">Changing the parameter ranges changes these probabilities. Structural model error is not represented unless explicitly sampled.</p></section>
</main><script>
const report={_json_for_script(payload)};
const canvas=document.getElementById("map"),ctx=canvas.getContext("2d"),select=document.getElementById("metric");
function palette(z,probability){{z=Math.max(0,Math.min(1,z));if(probability)return `rgb(${{Math.round(245-225*z)}},${{Math.round(248-118*z)}},${{Math.round(250-48*z)}})`;
return `rgb(${{Math.round(240-220*z)}},${{Math.round(247-100*z)}},${{Math.round(250-35*z)}})`}}
function draw(){{const choice=select.value,field=choice==="wet"?report.wetProbability:report.peakDepth[Number(choice.slice(1))];
let max=choice==="wet"?1:Math.max(...field.flat(),1e-12),nx=report.x.length,ny=report.y.length,left=70,top=20,w=canvas.width-left-25,h=canvas.height-top-55;
ctx.clearRect(0,0,canvas.width,canvas.height);for(let i=0;i<nx;i++)for(let j=0;j<ny;j++){{ctx.fillStyle=palette(field[i][j]/max,choice==="wet");
ctx.fillRect(left+i*w/nx,top+(ny-1-j)*h/ny,Math.ceil(w/nx),Math.ceil(h/ny));}}
ctx.strokeStyle=getComputedStyle(document.body).color;ctx.strokeRect(left,top,w,h);ctx.fillStyle=getComputedStyle(document.body).color;ctx.font="14px system-ui";
ctx.fillText("x = "+report.x[0].toFixed(0)+" m",left,canvas.height-14);ctx.textAlign="right";ctx.fillText("x = "+report.x[nx-1].toFixed(0)+" m",left+w,canvas.height-14);ctx.textAlign="start";
ctx.fillText(choice==="wet"?"Scale: 0–100%":"Scale: 0–"+max.toFixed(3)+" m",left+10,top+22);}}
select.addEventListener("change",draw);draw();
</script></body></html>"""


def generate_report(fields_path, summary_path=None, output_path=None, title=None):
    fields_path = Path(fields_path)
    fields = load_ensemble_fields(fields_path)
    if summary_path is None:
        candidate = fields_path.with_name(
            fields_path.name.replace("_ensemble.npz", "_ensemble_summary.json")
        )
        summary_path = candidate if candidate.is_file() else None
    summary = (
        {}
        if summary_path is None
        else json.loads(Path(summary_path).read_text(encoding="utf-8"))
    )
    if not isinstance(summary, dict):
        raise ValueError("Ensemble summary must contain a JSON object")
    destination = (
        Path(output_path)
        if output_path is not None
        else fields_path.with_name(
            fields_path.name.replace("_ensemble.npz", "_ensemble_report.html")
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_uncertainty_report(
            fields,
            summary,
            title or "2-D Flood Uncertainty Report",
        ),
        encoding="utf-8",
    )
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate an HTML report from a 2-D ensemble artifact."
    )
    parser.add_argument("ensemble", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title")
    args = parser.parse_args(argv)
    try:
        destination = generate_report(
            args.ensemble,
            summary_path=args.summary,
            output_path=args.output,
            title=args.title,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"Done. Report: {destination}")


if __name__ == "__main__":
    main()
