"""Regression tests for source-created waves from an initially dry domain."""

import numpy as np
import pytest

from general.solvers import saint_venant_1d, saint_venant_2d


def _constant_rain_1d(rate, breakpoints=()):
    def rainfall(x, time):
        del time
        return np.full_like(x, rate, dtype=float)

    rainfall.breakpoints_min = tuple(breakpoints)
    return rainfall


def _constant_rain_2d(shape, rate, breakpoints=()):
    def rainfall(x, y, time):
        del x, y, time
        return np.full(shape, rate, dtype=float)

    rainfall.breakpoints_min = tuple(breakpoints)
    return rainfall


@pytest.mark.parametrize("spatial_order", [1, 2])
def test_1d_dry_start_routes_constant_rain_without_artificial_breakpoints(
    spatial_order,
):
    cells = 20
    common = dict(
        L=20.0,
        T_final=10.0,
        record_interval=10.0,
        h_init=np.zeros(cells),
        q_init=np.zeros(cells),
        left_inflow=0.0,
        x_m=np.arange(cells, dtype=float) + 0.5,
        dx_m=np.ones(cells),
        slope=np.full(cells, 0.01),
        manning_n=np.full(cells, 0.04 / 60.0),
        downstream_boundary="outflow",
        spatial_order=spatial_order,
    )
    continuous = saint_venant_1d.run_model(
        **common, rainfall=_constant_rain_1d(1e-4)
    )
    segmented = saint_venant_1d.run_model(
        **common,
        rainfall=_constant_rain_1d(1e-4, range(1, 10)),
    )

    assert continuous["mass_outflow"] > 0.0
    assert np.ptp(continuous["h_final"]) > 0.0
    assert continuous["mass_outflow"] == pytest.approx(
        segmented["mass_outflow"], rel=5e-3
    )
    assert continuous["h_final"] == pytest.approx(
        segmented["h_final"], abs=2e-6
    )


def test_2d_dry_start_routes_rain_independently_of_forcing_knots():
    shape = (12, 3)
    common = dict(
        T_final=5.0,
        record_interval=5.0,
        h_init=np.zeros(shape),
        hu_init=np.zeros(shape),
        hv_init=np.zeros(shape),
        left_inflow=0.0,
        x_m=np.arange(shape[0], dtype=float) + 0.5,
        y_m=np.arange(shape[1], dtype=float) + 0.5,
        dx_m=np.ones(shape[0]),
        dy_m=np.ones(shape[1]),
        slope_x=np.full(shape, 0.01),
        slope_y=np.zeros(shape),
        manning_n=np.full(shape, 0.04 / 60.0),
        boundary_x="inflow_outflow",
        boundary_y="wall",
    )
    continuous = saint_venant_2d.run_model(
        **common, rainfall=_constant_rain_2d(shape, 1e-4)
    )
    segmented = saint_venant_2d.run_model(
        **common,
        rainfall=_constant_rain_2d(shape, 1e-4, range(1, 5)),
    )

    assert continuous["mass_outflow"] > 0.0
    assert np.ptp(continuous["h_final"]) > 0.0
    assert continuous["mass_outflow"] == pytest.approx(
        segmented["mass_outflow"], rel=2e-3
    )
    assert continuous["h_final"] == pytest.approx(
        segmented["h_final"], abs=2e-7
    )


def test_source_aware_wetting_preserves_soil_mass_and_knot_independence():
    cells = 10
    common = dict(
        L=10.0,
        T_final=5.0,
        record_interval=5.0,
        h_init=np.zeros(cells),
        q_init=np.zeros(cells),
        left_inflow=0.0,
        x_m=np.arange(cells, dtype=float) + 0.5,
        dx_m=np.ones(cells),
        slope=np.full(cells, 0.005),
        manning_n=np.full(cells, 0.04 / 60.0),
        soil_ksat_m_per_min=np.full(cells, 1e-6),
        soil_suction_head_m=np.full(cells, 0.01),
        soil_moisture_deficit=np.full(cells, 0.05),
    )
    continuous = saint_venant_1d.run_model(
        **common, rainfall=_constant_rain_1d(1e-4)
    )
    segmented = saint_venant_1d.run_model(
        **common,
        rainfall=_constant_rain_1d(1e-4, range(1, 5)),
    )

    assert continuous["mass_infiltration"] > 0.0
    assert continuous["mass_outflow"] > 0.0
    assert continuous["mass_infiltration"] == pytest.approx(
        segmented["mass_infiltration"], rel=1e-12
    )
    storage_change = float(
        np.sum(
            (continuous["h_final"] - continuous["h_initial"])
            * continuous["dx_m"]
        )
    )
    assert storage_change == pytest.approx(
        continuous["mass_inflow"]
        + continuous["mass_source"]
        + continuous["mass_floor_correction"]
        - continuous["mass_outflow"],
        abs=1e-12,
    )
