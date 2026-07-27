"""Quantitative Tier 3 verification cases for ``saint_venant_2d``.

The analytic benchmark is a periodic shear/contact wave:

    h = H,  u = U,  v = A sin(k (x - U t)).

It is an exact solution of the inviscid 2-D shallow-water equations on a flat
bed. It exercises transport of transverse momentum without linearizing the
equations. The first-order Rusanov discretization should converge at order one.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from general.solvers import saint_venant_2d as sv2
from general.solvers import saint_venant_1d as sv1


def _uniform_grid(length, count):
    step = length / count
    return (np.arange(count) + 0.5) * step, np.full(count, step)


def _zero_rain(shape):
    return lambda x, y, time: np.zeros(shape)


def analytic_shear_wave(resolutions=(20, 40, 80), final_time=0.2):
    length = 10.0
    width = 10.0
    base_depth = 0.01
    velocity_x = 4.0
    velocity_amplitude_y = 1.0
    errors = []

    for nx in resolutions:
        x, dx = _uniform_grid(length, nx)
        y, dy = _uniform_grid(width, 1)
        shape = (nx, 1)
        phase = 2.0 * np.pi * x / length
        h = np.full(shape, base_depth)
        hu = np.full(shape, base_depth * velocity_x)
        hv = (base_depth * velocity_amplitude_y * np.sin(phase))[:, None]
        zero = np.zeros(shape)
        result = sv2.run_model(
            T_final=final_time,
            record_interval=final_time,
            h_init=h,
            hu_init=hu,
            hv_init=hv,
            x_m=x,
            y_m=y,
            dx_m=dx,
            dy_m=dy,
            slope_x=zero,
            slope_y=zero,
            manning_n=zero,
            bed_elevation_m=zero,
            rainfall=_zero_rain(shape),
            boundary_x="periodic",
            boundary_y="periodic",
            cfl=0.45,
        )
        velocity_y = result["hv_final"][:, 0] / result["h_final"][:, 0]
        exact = velocity_amplitude_y * np.sin(
            2.0 * np.pi * (x - velocity_x * final_time) / length
        )
        error = float(np.sqrt(np.sum((velocity_y - exact) ** 2 * dx) / length))
        errors.append(error)

    rates = [
        float(np.log(errors[index] / errors[index + 1])
              / np.log(resolutions[index + 1] / resolutions[index]))
        for index in range(len(errors) - 1)
    ]
    fitted_order = float(
        -np.polyfit(np.log(np.asarray(resolutions, dtype=float)), np.log(errors), 1)[0]
    )
    return {
        "case": "periodic_analytic_shear_wave",
        "resolutions": list(resolutions),
        "l2_velocity_y_errors": errors,
        "pairwise_orders": rates,
        "fitted_order": fitted_order,
        "acceptance": {
            "errors_strictly_decrease": all(a > b for a, b in zip(errors, errors[1:])),
            "finest_l2_error_below_0_08": errors[-1] < 0.08,
            "fitted_order_between_0_7_and_1_3": 0.7 <= fitted_order <= 1.3,
        },
    }


def analytic_diagonal_vortex_wave(resolutions=(24, 48, 96), final_time=0.2):
    """Convergence for a genuinely two-dimensional exact vortical contact wave."""
    length = 10.0
    base_depth = 0.01
    base_velocity = 2.0
    amplitude = 0.5
    transverse_component = 1.0 / np.sqrt(2.0)
    errors = []

    for count in resolutions:
        x, dx = _uniform_grid(length, count)
        y, dy = _uniform_grid(length, count)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        phase = 2.0 * np.pi * (xx + yy) / length
        depth = np.full((count, count), base_depth)
        velocity_x = base_velocity + amplitude * transverse_component * np.sin(phase)
        velocity_y = base_velocity - amplitude * transverse_component * np.sin(phase)
        zero = np.zeros_like(depth)
        result = sv2.run_model(
            T_final=final_time,
            record_interval=final_time,
            h_init=depth,
            hu_init=base_depth * velocity_x,
            hv_init=base_depth * velocity_y,
            x_m=x,
            y_m=y,
            dx_m=dx,
            dy_m=dy,
            slope_x=zero,
            slope_y=zero,
            manning_n=zero,
            bed_elevation_m=zero,
            rainfall=_zero_rain(depth.shape),
            boundary_x="periodic",
            boundary_y="periodic",
            cfl=0.45,
        )
        exact_phase = 2.0 * np.pi * (
            xx + yy - 2.0 * base_velocity * final_time
        ) / length
        exact_u = base_velocity + amplitude * transverse_component * np.sin(exact_phase)
        exact_v = base_velocity - amplitude * transverse_component * np.sin(exact_phase)
        numerical_u = result["hu_final"] / result["h_final"]
        numerical_v = result["hv_final"] / result["h_final"]
        errors.append(float(np.sqrt(np.mean(
            (numerical_u - exact_u) ** 2 + (numerical_v - exact_v) ** 2
        ))))

    rates = [
        float(np.log(errors[index] / errors[index + 1])
              / np.log(resolutions[index + 1] / resolutions[index]))
        for index in range(len(errors) - 1)
    ]
    fitted_order = float(
        -np.polyfit(np.log(np.asarray(resolutions, dtype=float)), np.log(errors), 1)[0]
    )
    return {
        "case": "periodic_analytic_diagonal_vortex_wave",
        "resolutions": list(resolutions),
        "l2_vector_velocity_errors": errors,
        "pairwise_orders": rates,
        "fitted_order": fitted_order,
        "acceptance": {
            "errors_strictly_decrease": all(a > b for a, b in zip(errors, errors[1:])),
            "finest_l2_error_below_0_06": errors[-1] < 0.06,
            "fitted_order_between_0_7_and_1_3": 0.7 <= fitted_order <= 1.3,
        },
    }


def manufactured_pressure_wave(resolutions=(20, 40, 80), final_time=0.02):
    """Manufactured solution exercising variable depth and pressure flux."""
    length = 10.0
    width = 10.0
    base_depth = 0.05
    depth_amplitude = 0.01
    velocity_x = 4.0
    wave_number = 2.0 * np.pi / length
    errors = []

    for nx in resolutions:
        x, dx = _uniform_grid(length, nx)
        y, dy = _uniform_grid(width, 1)
        phase = wave_number * x
        depth = (base_depth + depth_amplitude * np.sin(phase))[:, None]
        hu = velocity_x * depth
        zero = np.zeros_like(depth)

        def momentum_source(stations, cross_stations, time):
            del cross_stations
            exact_phase = wave_number * (stations - velocity_x * time)
            exact_depth = base_depth + depth_amplitude * np.sin(exact_phase)
            depth_gradient = depth_amplitude * wave_number * np.cos(exact_phase)
            return (sv2.g * exact_depth * depth_gradient)[:, None], zero

        result = sv2.run_model(
            T_final=final_time,
            record_interval=final_time,
            h_init=depth,
            hu_init=hu,
            hv_init=zero,
            x_m=x,
            y_m=y,
            dx_m=dx,
            dy_m=dy,
            slope_x=zero,
            slope_y=zero,
            manning_n=zero,
            bed_elevation_m=zero,
            rainfall=_zero_rain(depth.shape),
            momentum_source=momentum_source,
            boundary_x="periodic",
            boundary_y="periodic",
            cfl=0.45,
        )
        exact_phase = wave_number * (x - velocity_x * final_time)
        exact_depth = base_depth + depth_amplitude * np.sin(exact_phase)
        exact_hu = velocity_x * exact_depth
        depth_error = result["h_final"][:, 0] - exact_depth
        momentum_error = result["hu_final"][:, 0] - exact_hu
        errors.append(float(np.sqrt(np.mean(
            depth_error**2 + (momentum_error / velocity_x) ** 2
        ))))

    rates = [
        float(np.log(errors[index] / errors[index + 1])
              / np.log(resolutions[index + 1] / resolutions[index]))
        for index in range(len(errors) - 1)
    ]
    fitted_order = float(
        -np.polyfit(np.log(np.asarray(resolutions, dtype=float)), np.log(errors), 1)[0]
    )
    return {
        "case": "manufactured_variable_depth_pressure_wave",
        "resolutions": list(resolutions),
        "combined_l2_errors": errors,
        "pairwise_orders": rates,
        "fitted_order": fitted_order,
        "acceptance": {
            "errors_strictly_decrease": all(a > b for a, b in zip(errors, errors[1:])),
            "finest_l2_error_below_0_001": errors[-1] < 0.001,
            "fitted_order_between_0_7_and_1_3": 0.7 <= fitted_order <= 1.3,
        },
    }


def nonflat_lake_at_rest():
    nx, ny = 24, 16
    x, dx = _uniform_grid(4.0, nx)
    y, dy = _uniform_grid(3.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    bed = 0.15 * np.exp(-((xx - 2.0) ** 2 + (yy - 1.5) ** 2) / 0.4)
    depth = 0.5 - bed
    zero = np.zeros_like(depth)
    result = sv2.run_model(
        T_final=0.1,
        record_interval=0.1,
        h_init=depth,
        hu_init=zero,
        hv_init=zero,
        x_m=x,
        y_m=y,
        dx_m=dx,
        dy_m=dy,
        slope_x=zero,
        slope_y=zero,
        manning_n=zero,
        bed_elevation_m=bed,
        rainfall=_zero_rain(depth.shape),
        boundary_x="periodic",
        boundary_y="periodic",
    )
    return {
        "case": "nonflat_lake_at_rest",
        "max_depth_error_m": float(np.max(np.abs(result["h_final"] - depth))),
        "max_abs_hu_m2_per_min": float(np.max(np.abs(result["hu_final"]))),
        "max_abs_hv_m2_per_min": float(np.max(np.abs(result["hv_final"]))),
        "acceptance": {
            "depth_error_below_1e_12": bool(
                np.max(np.abs(result["h_final"] - depth)) < 1e-12
            ),
            "momentum_below_1e_11": bool(max(
                np.max(np.abs(result["hu_final"])),
                np.max(np.abs(result["hv_final"])),
            ) < 1e-11),
        },
    }


def one_dimensional_reduction():
    nx = 80
    length = 4.0
    final_time = 0.002
    x, dx = _uniform_grid(length, nx)
    depth = 0.05 + 0.01 * np.exp(-((x - 2.0) / 0.35) ** 2)
    discharge = np.zeros(nx)
    slope = np.zeros(nx)
    roughness = np.full(nx, 1e-12)
    result_1d = sv1.run_model(
        length,
        final_time,
        record_interval=final_time,
        h_init=depth,
        q_init=discharge,
        left_inflow=0.0,
        rainfall=lambda stations, time: np.zeros_like(stations),
        x_m=x,
        dx_m=dx,
        slope=slope,
        manning_n=roughness,
        cfl=0.45,
    )

    y = np.array([500.0])
    dy = np.array([1000.0])
    zero_2d = np.zeros((nx, 1))
    result_2d = sv2.run_model(
        T_final=final_time,
        record_interval=final_time,
        h_init=depth[:, None],
        hu_init=zero_2d,
        hv_init=zero_2d,
        left_inflow=0.0,
        x_m=x,
        y_m=y,
        dx_m=dx,
        dy_m=dy,
        slope_x=zero_2d,
        slope_y=zero_2d,
        manning_n=np.full((nx, 1), 1e-12),
        bed_elevation_m=zero_2d,
        rainfall=_zero_rain((nx, 1)),
        cfl=0.45,
    )
    depth_l2 = float(np.sqrt(np.mean(
        (result_2d["h_final"][:, 0] - result_1d["h_final"]) ** 2
    )))
    discharge_l2 = float(np.sqrt(np.mean(
        (result_2d["hu_final"][:, 0] - result_1d["q_final"]) ** 2
    )))
    transverse_max = float(np.max(np.abs(result_2d["hv_final"])))
    return {
        "case": "one_dimensional_reduction",
        "depth_l2_difference_m": depth_l2,
        "discharge_l2_difference_m2_per_min": discharge_l2,
        "max_transverse_discharge_m2_per_min": transverse_max,
        "acceptance": {
            "depth_l2_below_1e_8": depth_l2 < 1e-8,
            "discharge_l2_below_1e_6": discharge_l2 < 1e-6,
            "transverse_discharge_below_1e_12": transverse_max < 1e-12,
        },
    }


def radial_dam_break_symmetry(grid_size=64):
    length = 4.0
    x, dx = _uniform_grid(length, grid_size)
    y, dy = _uniform_grid(length, grid_size)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    radius = np.hypot(xx - length / 2, yy - length / 2)
    depth = np.where(radius < 0.5, 0.2, 0.05)
    zero = np.zeros_like(depth)
    result = sv2.run_model(
        T_final=0.01,
        record_interval=0.01,
        h_init=depth,
        hu_init=zero,
        hv_init=zero,
        x_m=x,
        y_m=y,
        dx_m=dx,
        dy_m=dy,
        slope_x=zero,
        slope_y=zero,
        manning_n=zero,
        bed_elevation_m=zero,
        rainfall=_zero_rain(depth.shape),
        boundary_x="periodic",
        boundary_y="periodic",
    )
    final_depth = result["h_final"]
    angles = np.linspace(0.0, 2.0 * np.pi, 144, endpoint=False)
    angular_ranges = []
    for sample_radius in np.linspace(0.2, 1.0, 81):
        px = length / 2 + sample_radius * np.cos(angles)
        py = length / 2 + sample_radius * np.sin(angles)
        fx = px / dx[0] - 0.5
        fy = py / dy[0] - 0.5
        i = np.floor(fx).astype(int)
        j = np.floor(fy).astype(int)
        tx = fx - i
        ty = fy - j
        samples = (
            (1 - tx) * (1 - ty) * final_depth[i, j]
            + tx * (1 - ty) * final_depth[i + 1, j]
            + (1 - tx) * ty * final_depth[i, j + 1]
            + tx * ty * final_depth[i + 1, j + 1]
        )
        angular_ranges.append(float(np.max(samples) - np.min(samples)))
    normalized_deviation = max(angular_ranges) / 0.15
    rotational_error = float(np.max(np.abs(final_depth - np.rot90(final_depth))))
    return {
        "case": "wet_background_radial_dam_break",
        "grid_size": grid_size,
        "max_angular_depth_range_m": max(angular_ranges),
        "normalized_angular_deviation": normalized_deviation,
        "quarter_turn_symmetry_error_m": rotational_error,
        "acceptance": {
            "normalized_angular_deviation_below_0_02": bool(
                normalized_deviation < 0.02
            ),
            "quarter_turn_error_below_1e_12": bool(rotational_error < 1e-12),
        },
    }


def strict_periodic_mass_conservation():
    nx, ny = 40, 24
    x, dx = _uniform_grid(5.0, nx)
    y, dy = _uniform_grid(3.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    depth = 0.08 + 0.01 * np.sin(2 * np.pi * xx / 5.0) * np.cos(2 * np.pi * yy / 3.0)
    hu = 0.02 * depth
    hv = -0.01 * depth
    zero = np.zeros_like(depth)
    result = sv2.run_model(
        T_final=0.02,
        record_interval=0.02,
        h_init=depth,
        hu_init=hu,
        hv_init=hv,
        x_m=x,
        y_m=y,
        dx_m=dx,
        dy_m=dy,
        slope_x=zero,
        slope_y=zero,
        manning_n=zero,
        bed_elevation_m=zero,
        rainfall=_zero_rain(depth.shape),
        boundary_x="periodic",
        boundary_y="periodic",
    )
    area = dx[:, None] * dy[None, :]
    initial_volume = float(np.sum(depth * area))
    final_volume = float(np.sum(result["h_final"] * area))
    residual = final_volume - initial_volume
    relative_residual = abs(residual) / initial_volume
    return {
        "case": "strict_periodic_mass_conservation",
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "residual_m3": residual,
        "relative_residual": relative_residual,
        "mass_floor_correction_m3": result["mass_floor_correction"],
        "acceptance": {
            "relative_residual_below_1e_12": relative_residual < 1e-12,
            "floor_correction_below_1e_14": result["mass_floor_correction"] < 1e-14,
        },
    }


def wet_dry_dam_break():
    grid_size = 48
    length = 4.0
    x, dx = _uniform_grid(length, grid_size)
    y, dy = _uniform_grid(length, grid_size)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    depth = np.where(np.hypot(xx - 2.0, yy - 2.0) < 0.5, 0.2, 0.0)
    zero = np.zeros_like(depth)
    result = sv2.run_model(
        T_final=0.01,
        record_interval=0.002,
        h_init=depth,
        hu_init=zero,
        hv_init=zero,
        x_m=x,
        y_m=y,
        dx_m=dx,
        dy_m=dy,
        slope_x=zero,
        slope_y=zero,
        manning_n=zero,
        bed_elevation_m=zero,
        rainfall=_zero_rain(depth.shape),
        boundary_x="periodic",
        boundary_y="periodic",
    )
    area = dx[:, None] * dy[None, :]
    initial_volume = float(np.sum(depth * area))
    final_volume = float(np.sum(result["h_final"] * area))
    return {
        "case": "wet_dry_radial_dam_break",
        "minimum_recorded_depth_m": float(np.min(result["h_history"])),
        "relative_mass_residual": abs(final_volume - initial_volume) / initial_volume,
        "mass_floor_correction_m3": result["mass_floor_correction"],
        "acceptance": {
            "all_depths_nonnegative": bool(np.min(result["h_history"]) >= 0),
            "relative_mass_residual_below_1e_12": (
                abs(final_volume - initial_volume) / initial_volume < 1e-12
            ),
            "floor_correction_below_1e_14": result["mass_floor_correction"] < 1e-14,
        },
    }


def run_verification_suite():
    cases = {
        "analytic_shear_wave": analytic_shear_wave(),
        "analytic_diagonal_vortex_wave": analytic_diagonal_vortex_wave(),
        "manufactured_pressure_wave": manufactured_pressure_wave(),
        "nonflat_lake_at_rest": nonflat_lake_at_rest(),
        "one_dimensional_reduction": one_dimensional_reduction(),
        "radial_dam_break": radial_dam_break_symmetry(),
        "mass_conservation": strict_periodic_mass_conservation(),
        "wet_dry_dam_break": wet_dry_dam_break(),
    }
    passed = all(
        all(bool(value) for value in case["acceptance"].values())
        for case in cases.values()
    )
    return {
        "schema_version": 1,
        "solver": "saint_venant_2d",
        "passed": passed,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "cases": cases,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    args = parser.parse_args(argv)
    results = run_verification_suite()
    text = json.dumps(results, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
