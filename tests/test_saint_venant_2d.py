"""Tests for the 2D shallow-water (Saint-Venant) solver in
general/solvers/saint_venant_2d.py.

The scheme is a conservative finite-volume method with Rusanov (local
Lax-Friedrichs) fluxes in x and y, an inflow/open-outflow x-boundary and
reflecting solid walls in y. Covers:

1. Stability: the run completes with all-finite output (the old central-difference
   version silently overflowed to NaN while still exiting 0).
2. Mass conservation: change in stored volume == inflow - outflow + rainfall
   source + dry-floor correction.
3. Well-balanced "lake at rest": flat water, zero slope, no forcing stays exactly
   at rest.
4. y-invariance: a problem uniform in y (and with solid y-walls) stays uniform in y.
"""
import numpy as np
import pytest

from general.solvers import saint_venant_2d as sv2


def _no_rain(t):
    return 0.0


def test_runs_stably_with_all_finite_output():
    result = sv2.run_model(L=2.0, W=1.0, T_final=1.0, record_interval=0.25, left_inflow=0.02)

    for key in ("h_history", "h_initial", "h_final", "hu_final", "hv_final", "times"):
        assert np.all(np.isfinite(result[key])), f"{key} contains non-finite values"
    assert np.all(result["h_final"] >= 0.0)
    # 2D snapshots: (n_times, nx, ny)
    assert result["h_history"].ndim == 3
    assert result["h_history"].shape[0] == len(result["times"])


def test_mass_conservation(monkeypatch):
    monkeypatch.setattr(sv2, "r", lambda t: 0.0003)  # uniform rainfall rate m/min

    result = sv2.run_model(L=2.0, W=1.0, T_final=1.5, record_interval=0.5, left_inflow=0.02)

    dx = result["x"][1] - result["x"][0]
    dy = result["y"][1] - result["y"][0]
    storage_initial = float(np.sum(result["h_initial"]) * dx * dy)
    storage_final = float(np.sum(result["h_final"]) * dx * dy)
    delta = storage_final - storage_initial
    expected = (
        result["mass_inflow"]
        - result["mass_outflow"]
        + result["mass_source"]
        + result["mass_floor_correction"]
    )
    scale = max(abs(storage_initial), abs(storage_final), abs(expected), 1.0)
    assert abs(delta - expected) / scale < 1e-12
    assert result["mass_floor_correction"] < 1e-14
    assert result["mass_inflow"] > 0.0
    assert result["mass_source"] > 0.0


def test_lake_at_rest_stays_at_rest(monkeypatch):
    # Zero bed slope in both directions and no rainfall: a flat lake must not move.
    monkeypatch.setattr(sv2, "S0x", 0.0)
    monkeypatch.setattr(sv2, "S0y", 0.0)
    monkeypatch.setattr(sv2, "r", _no_rain)

    nx, ny = 20, 10
    h0 = np.full((nx, ny), 0.5)
    result = sv2.run_model(
        L=2.0, W=1.0, T_final=1.0, record_interval=0.5,
        h_init=h0, hu_init=np.zeros((nx, ny)), hv_init=np.zeros((nx, ny)),
        left_inflow=0.0,
    )

    assert np.allclose(result["h_final"], 0.5, rtol=0.0, atol=1e-12)
    assert np.max(np.abs(result["hu_final"])) == pytest.approx(0.0, abs=1e-12)
    assert np.max(np.abs(result["hv_final"])) == pytest.approx(0.0, abs=1e-12)


def test_second_order_reconstruction_preserves_nonflat_lake():
    count = 24
    length = 2.4
    spacing = np.full(count, length / count)
    x = (np.arange(count) + 0.5) * spacing[0]
    y = x.copy()
    xx, yy = np.meshgrid(x, y, indexing="ij")
    bed = (
        0.05
        * np.sin(2.0 * np.pi * xx / length)
        * np.sin(2.0 * np.pi * yy / length)
    )
    depth = 0.2 - bed
    zero = np.zeros_like(depth)
    result = sv2.run_model(
        T_final=0.02,
        record_interval=0.02,
        h_init=depth,
        hu_init=zero,
        hv_init=zero,
        x_m=x,
        y_m=y,
        dx_m=spacing,
        dy_m=spacing,
        slope_x=zero,
        slope_y=zero,
        manning_n=zero,
        bed_elevation_m=bed,
        rainfall=lambda x, y, time: zero,
        boundary_x="periodic",
        boundary_y="periodic",
        spatial_order=2,
        cfl=0.3,
    )

    assert np.max(np.abs(result["h_final"] - depth)) < 1e-12
    assert np.max(np.abs(result["hu_final"])) < 1e-12
    assert np.max(np.abs(result["hv_final"])) < 1e-12


def test_second_order_reconstruction_reduces_shear_wave_error():
    def shear_error(order):
        cells = 40
        length = 10.0
        dx = np.full(cells, length / cells)
        x = (np.arange(cells) + 0.5) * dx[0]
        y = np.array([5.0])
        dy = np.array([10.0])
        depth = np.full((cells, 1), 0.01)
        velocity_x = 4.0
        zero = np.zeros_like(depth)
        result = sv2.run_model(
            T_final=0.2,
            record_interval=0.2,
            h_init=depth,
            hu_init=np.full_like(depth, depth[0, 0] * velocity_x),
            hv_init=(depth[0, 0] * np.sin(2.0 * np.pi * x / length))[:, None],
            x_m=x,
            y_m=y,
            dx_m=dx,
            dy_m=dy,
            slope_x=zero,
            slope_y=zero,
            manning_n=zero,
            bed_elevation_m=zero,
            rainfall=lambda x, y, time: zero,
            boundary_x="periodic",
            boundary_y="periodic",
            spatial_order=order,
            cfl=0.3,
        )
        exact = np.sin(
            2.0 * np.pi * (x - velocity_x * 0.2) / length
        )
        numerical = result["hv_final"][:, 0] / result["h_final"][:, 0]
        return float(np.sqrt(np.mean((numerical - exact) ** 2)))

    assert shear_error(2) < 0.25 * shear_error(1)


def test_problem_uniform_in_y_stays_uniform_in_y(monkeypatch):
    # Uniform in y, solid y-walls, uniform inflow -> the solution must not develop
    # any y-structure (each x-column stays constant across y).
    monkeypatch.setattr(sv2, "r", lambda t: 0.0002)

    nx, ny = 20, 10
    result = sv2.run_model(
        L=2.0, W=1.0, T_final=1.0, record_interval=0.5,
        h_init=np.full((nx, ny), 0.05),
        hu_init=np.zeros((nx, ny)), hv_init=np.zeros((nx, ny)),
        left_inflow=0.02,
    )

    h_final = result["h_final"]
    spread = np.max(np.ptp(h_final, axis=1))  # max variation across y over all x
    assert spread == pytest.approx(0.0, abs=1e-10)
    assert np.max(np.abs(result["hv_final"])) == pytest.approx(0.0, abs=1e-10)


def test_rainfall_on_partially_wet_domain_is_positive_and_conservative():
    nx, ny = 20, 10
    shape = (nx, ny)
    depth = np.zeros(shape)
    depth[: nx // 2, :] = 0.02
    zero = np.zeros(shape)
    rate = 0.0004
    result = sv2.run_model(
        L=2.0,
        W=1.0,
        T_final=0.1,
        record_interval=0.05,
        h_init=depth,
        hu_init=zero,
        hv_init=zero,
        rainfall=lambda x, y, time: np.full(shape, rate),
        slope_x=zero,
        slope_y=zero,
        manning_n=zero,
        bed_elevation_m=zero,
        boundary_x="periodic",
        boundary_y="periodic",
    )

    dx = result["dx_m"][:, None]
    dy = result["dy_m"][None, :]
    initial_volume = float(np.sum(depth * dx * dy))
    final_volume = float(np.sum(result["h_final"] * dx * dy))
    expected_rain = rate * 2.0 * 1.0 * 0.1
    assert np.min(result["h_history"]) >= 0.0
    assert final_volume - initial_volume == pytest.approx(expected_rain, rel=1e-12, abs=1e-14)
    assert result["mass_source"] == pytest.approx(expected_rain, rel=1e-12, abs=1e-14)
    assert result["mass_floor_correction"] < 1e-14


def test_nonfinite_dynamics_fail_with_diagnostic():
    shape = (10, 10)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="Invalid time step|non-finite"):
            sv2.run_model(
                L=1.0,
                W=1.0,
                T_final=0.01,
                h_init=np.ones(shape),
                hu_init=np.full(shape, 1e308),
                hv_init=np.full(shape, 1e308),
                rainfall=lambda x, y, time: np.zeros(shape),
                slope_x=np.zeros(shape),
                slope_y=np.zeros(shape),
                manning_n=np.zeros(shape),
                bed_elevation_m=np.zeros(shape),
            )


def test_periodic_wet_dry_front_crosses_domain_edge_conservatively():
    nx, ny = 24, 12
    length, width = 2.4, 1.2
    dx = np.full(nx, length / nx)
    dy = np.full(ny, width / ny)
    x = (np.arange(nx) + 0.5) * dx[0]
    y = (np.arange(ny) + 0.5) * dy[0]
    depth = np.zeros((nx, ny))
    depth[[0, 1, -2, -1], :] = 0.1
    zero = np.zeros_like(depth)
    result = sv2.run_model(
        T_final=0.005,
        record_interval=0.005,
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
        rainfall=lambda x, y, time: zero,
        boundary_x="periodic",
        boundary_y="periodic",
    )
    cell_area = dx[:, None] * dy[None, :]
    initial_volume = float(np.sum(depth * cell_area))
    final_volume = float(np.sum(result["h_final"] * cell_area))
    assert np.min(result["h_history"]) >= 0.0
    assert abs(final_volume - initial_volume) / initial_volume < 1e-12
    assert result["mass_floor_correction"] < 1e-14


def test_wet_dry_front_over_nonflat_bed_is_positive_and_conservative():
    # The draining limiter's hardest, previously-untested corner: a wet/dry front
    # advancing over NON-FLAT bed topography (every other wet/dry case uses a flat
    # bed). A raised water dome sits on a Gaussian bed bump and drains, frictionless,
    # down the bump's sloping flanks into dry cells. With periodic boundaries the
    # domain is watertight, so volume must be conserved and depth must stay >= 0.
    nx, ny = 24, 24
    length = width = 2.4
    dx = np.full(nx, length / nx)
    dy = np.full(ny, width / ny)
    x = (np.arange(nx) + 0.5) * dx[0]
    y = (np.arange(ny) + 0.5) * dy[0]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    bed = 0.1 * np.exp(-(((xx - 1.2) ** 2 + (yy - 1.2) ** 2) / 0.4 ** 2))  # decays ~0 at edges
    depth = np.zeros((nx, ny))
    core = (np.abs(xx - 1.2) < 0.3) & (np.abs(yy - 1.2) < 0.3)
    depth[core] = 0.08  # raised dome -> not in hydrostatic balance -> drains over the slope
    zero = np.zeros((nx, ny))

    result = sv2.run_model(
        T_final=0.02,
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
        manning_n=zero,  # frictionless: most stressful for positivity at the front
        bed_elevation_m=bed,
        rainfall=lambda x, y, time: zero,
        boundary_x="periodic",
        boundary_y="periodic",
    )
    cell_area = dx[:, None] * dy[None, :]
    initial_volume = float(np.sum(depth * cell_area))
    final_volume = float(np.sum(result["h_final"] * cell_area))
    assert np.min(result["h_history"]) >= 0.0
    assert abs(final_volume - initial_volume) / initial_volume < 1e-12
    assert result["mass_floor_correction"] < 1e-14


def test_stage_outlet_preserves_nonflat_lake_at_rest():
    nx, ny = 21, 7
    x = np.linspace(0.0, 10.0, nx)
    y = np.linspace(0.0, 3.0, ny)
    dx = np.full(nx, 10.0 / nx)
    dy = np.full(ny, 3.0 / ny)
    bed = -0.01 * x[:, None] + np.zeros((nx, ny))
    surface = 1.0
    depth = surface - bed
    zero = np.zeros_like(depth)

    result = sv2.run_model(
        T_final=0.05,
        record_interval=0.05,
        h_init=depth,
        hu_init=zero,
        hv_init=zero,
        x_m=x,
        y_m=y,
        dx_m=dx,
        dy_m=dy,
        slope_x=np.full_like(depth, 0.01),
        slope_y=zero,
        manning_n=zero,
        bed_elevation_m=bed,
        rainfall=lambda x, y, time: zero,
        left_inflow=0.0,
        boundary_x="inflow_stage",
        downstream_stage_m=surface,
    )

    assert np.max(np.abs(result["h_final"] - depth)) < 1e-12
    assert np.max(np.abs(result["hu_final"])) < 1e-10
    assert np.max(np.abs(result["hv_final"])) < 1e-10


def test_high_downstream_stage_drives_conservative_2d_backflow():
    nx, ny = 12, 6
    shape = (nx, ny)
    depth = np.full(shape, 0.1)
    zero = np.zeros(shape)
    result = sv2.run_model(
        L=1.2,
        W=0.6,
        T_final=0.001,
        record_interval=0.001,
        h_init=depth,
        hu_init=zero,
        hv_init=zero,
        slope_x=zero,
        slope_y=zero,
        manning_n=zero,
        bed_elevation_m=zero,
        rainfall=lambda x, y, time: zero,
        left_inflow=0.0,
        boundary_x="inflow_stage",
        downstream_stage_m=0.5,
    )
    area = result["dx_m"][:, None] * result["dy_m"][None, :]
    storage_change = float(
        np.sum((result["h_final"] - result["h_initial"]) * area)
    )

    assert result["mass_inflow"] > 0.0
    assert result["mass_outflow"] == pytest.approx(0.0)
    assert result["h_final"][-1].mean() > depth[-1].mean()
    assert storage_change == pytest.approx(
        result["mass_inflow"]
        - result["mass_outflow"]
        + result["mass_floor_correction"],
        abs=1e-12,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"boundary_x": "inflow_stage"},
        {
            "boundary_x": "inflow_outflow",
            "downstream_stage_m": 1.0,
        },
    ],
)
def test_invalid_2d_stage_boundary_configuration_raises(kwargs):
    with pytest.raises(ValueError, match="downstream_stage"):
        sv2.run_model(T_final=0.0, **kwargs)
