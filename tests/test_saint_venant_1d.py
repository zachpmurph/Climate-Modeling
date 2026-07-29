import math

import numpy as np
import pytest

from general.solvers import saint_venant_1d as sv


def disable_forcing(monkeypatch):
    monkeypatch.setattr(sv, "S0", 0.0)
    monkeypatch.setattr(sv, "r", lambda x, t: np.zeros_like(x))


def test_still_water_at_rest(monkeypatch):
    disable_forcing(monkeypatch)
    h0 = np.full(int(sv.L * 10), 0.5)
    q0 = np.zeros_like(h0)

    result = sv.run_model(sv.L, 0.01, h_init=h0, q_init=q0)

    assert np.array_equal(result["h_final"], h0)
    assert np.array_equal(result["q_final"], q0)


def test_mass_conservation():
    result = sv.run_model(sv.L, 40.0)
    dx = result["x"][1] - result["x"][0]
    storage_delta = np.sum(result["h_final"] - result["h_initial"]) * dx
    expected_delta = (
        result["mass_inflow"]
        + result["mass_source"]
        - result["mass_outflow"]
        + result["mass_floor_correction"]
    )

    assert storage_delta == pytest.approx(expected_delta, rel=1e-10, abs=1e-12)
    assert result["mass_inflow"] == pytest.approx(0.0)
    assert result["mass_floor_correction"] == pytest.approx(0.0)


def test_record_interval_does_not_change_solution(monkeypatch):
    disable_forcing(monkeypatch)

    sparse = sv.run_model(sv.L, 0.2, record_interval=0.2)
    frequent = sv.run_model(sv.L, 0.2, record_interval=0.001)

    assert np.array_equal(sparse["h_final"], frequent["h_final"])
    assert np.array_equal(sparse["q_final"], frequent["q_final"])
    assert frequent["times"][-1] == pytest.approx(0.2)


def test_uniform_manning_flow_is_steady(monkeypatch):
    monkeypatch.setattr(sv, "r", lambda x, t: np.zeros_like(x))
    h0 = np.full(int(sv.L * 10), 0.5)
    equilibrium_q = h0[0] ** (5.0 / 3.0) * math.sqrt(sv.S0) / sv.n0
    q0 = np.full_like(h0, equilibrium_q)

    result = sv.run_model(
        sv.L,
        0.01,
        h_init=h0,
        q_init=q0,
        left_inflow=equilibrium_q,
    )

    # A first-order hydrostatic scheme is exactly well-balanced for still water,
    # not for moving frictional equilibria. The normal-flow state should remain
    # close over this short step and converge under refinement.
    assert np.max(np.abs(result["h_final"] - h0)) < 0.01
    assert np.max(np.abs(result["q_final"] - q0)) / equilibrium_q < 0.02


def test_prescribed_upstream_inflow_is_accounted(monkeypatch):
    disable_forcing(monkeypatch)
    h0 = np.full(int(sv.L * 10), 0.2)
    q0 = np.zeros_like(h0)
    inflow = 0.01
    final_time = 0.05

    result = sv.run_model(
        sv.L,
        final_time,
        h_init=h0,
        q_init=q0,
        left_inflow=lambda t: inflow,
    )
    dx = result["x"][1] - result["x"][0]
    storage_delta = np.sum(result["h_final"] - result["h_initial"]) * dx

    assert result["mass_inflow"] == pytest.approx(inflow * final_time)
    assert storage_delta == pytest.approx(
        result["mass_inflow"] - result["mass_outflow"] + result["mass_floor_correction"],
        abs=1e-12,
    )


def test_exactly_dry_domain_has_no_warning_or_mass_gain(monkeypatch):
    disable_forcing(monkeypatch)
    dry = np.zeros(int(sv.L * 10))

    result = sv.run_model(sv.L, 0.01, h_init=dry, q_init=dry)

    assert np.all(result["h_final"] == 0.0)
    assert np.all(result["q_final"] == 0.0)
    assert result["mass_floor_correction"] == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"record_interval": 0}, "record_interval"),
        ({"h_init": np.ones(3)}, "h_init"),
        ({"h_init": -np.ones(int(sv.L * 10))}, "negative"),
        ({"left_inflow": -1.0}, "left_inflow"),
    ],
)
def test_invalid_inputs_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        sv.run_model(sv.L, 0.01, **kwargs)


def test_profile_grid_uses_per_cell_slope_and_roughness(monkeypatch):
    monkeypatch.setattr(sv, "r", lambda x, t: np.zeros_like(x))
    x_m = np.array([0.0, 80.0, 210.0])
    dx_m = np.array([50.0, 100.0, 140.0])
    slope = np.array([0.0005, 0.001, 0.002])
    manning_n = np.array([0.0005, 0.001, 0.002])
    h0 = np.full(3, 0.5)
    q0 = np.full(3, 0.1)

    slope_result = sv.run_model(
        float(np.sum(dx_m)),
        0.01,
        h_init=h0,
        q_init=q0,
        left_inflow=0.1,
        rainfall=lambda x, t: np.zeros_like(x),
        x_m=x_m,
        dx_m=dx_m,
        slope=slope,
        manning_n=np.full(3, 0.001),
    )

    roughness_result = sv.run_model(
        float(np.sum(dx_m)),
        0.01,
        h_init=h0,
        q_init=q0,
        left_inflow=0.1,
        rainfall=lambda x, t: np.zeros_like(x),
        x_m=x_m,
        dx_m=dx_m,
        slope=np.full(3, 0.001),
        manning_n=manning_n,
    )

    assert np.array_equal(slope_result["x"], x_m)
    assert np.array_equal(slope_result["dx_m"], dx_m)
    assert np.array_equal(slope_result["slope"], slope)
    assert np.array_equal(roughness_result["manning_n"], manning_n)
    assert np.all(np.isfinite(slope_result["q_final"]))
    assert not np.array_equal(slope_result["q_final"], roughness_result["q_final"])


def test_nonflat_lake_at_rest_is_well_balanced():
    x_m = np.linspace(0.0, 1000.0, 101)
    dx_m = np.full_like(x_m, 1000.0 / len(x_m))
    slope = np.full_like(x_m, 0.001)
    bed = -slope * x_m
    depth = 2.0 - bed

    result = sv.run_model(
        1000.0,
        0.1,
        h_init=depth,
        q_init=np.zeros_like(depth),
        left_inflow=0.0,
        rainfall=lambda x, t: np.zeros_like(x),
        x_m=x_m,
        dx_m=dx_m,
        slope=slope,
        manning_n=np.full_like(x_m, 1e-12),
        bed_elevation_m=bed,
        cfl=0.4,
    )

    assert np.max(np.abs(result["h_final"] - depth)) < 1e-12
    assert np.max(np.abs(result["q_final"])) < 1e-10
    assert result["mass_floor_correction"] == 0.0


def test_varying_rectangular_width_preserves_nonflat_lake_at_rest():
    x_m = np.linspace(0.0, 1000.0, 101)
    dx_m = np.full_like(x_m, 1000.0 / len(x_m))
    slope = np.full_like(x_m, 0.001)
    bed = -slope * x_m
    depth = 2.0 - bed
    width = 20.0 + 10.0 * np.sin(np.linspace(0.0, np.pi, len(x_m)))

    result = sv.run_model(
        1000.0,
        0.1,
        h_init=depth,
        q_init=np.zeros_like(depth),
        left_inflow=0.0,
        rainfall=lambda x, t: np.zeros_like(x),
        x_m=x_m,
        dx_m=dx_m,
        slope=slope,
        manning_n=np.full_like(x_m, 0.001),
        bed_elevation_m=bed,
        channel_width_m=width,
        cfl=0.4,
    )

    assert np.max(np.abs(result["h_final"] - depth)) < 1e-12
    assert np.max(np.abs(result["q_final"])) < 1e-10
    assert result["uses_cross_section"] is True


def test_rectangular_width_gives_volumetric_rainfall_mass_balance():
    x_m = np.array([0.0, 100.0, 250.0])
    dx_m = np.array([50.0, 125.0, 100.0])
    width = np.array([10.0, 20.0, 30.0])
    rainfall_rate = np.array([0.0, 0.00001, 0.00002])
    duration = 0.1

    result = sv.run_model(
        float(np.sum(dx_m)),
        duration,
        h_init=np.full(3, 0.2),
        q_init=np.zeros(3),
        rainfall=lambda x, t: rainfall_rate,
        x_m=x_m,
        dx_m=dx_m,
        slope=np.zeros(3),
        manning_n=np.full(3, 0.001),
        channel_width_m=width,
    )

    expected_source = float(np.sum(rainfall_rate * width * dx_m) * duration)
    storage_delta = float(
        np.sum((result["h_final"] - result["h_initial"]) * width * dx_m)
    )
    assert result["mass_source"] == pytest.approx(expected_source)
    assert storage_delta == pytest.approx(
        result["mass_source"]
        - result["mass_outflow"]
        + result["mass_inflow"]
        + result["mass_floor_correction"],
        abs=1e-12,
    )


def test_exposed_rainfall_function_is_spatial_and_mass_conservative():
    x_m = np.array([0.0, 100.0, 250.0])
    dx_m = np.array([50.0, 125.0, 100.0])
    rainfall_rate = np.array([0.0, 0.00001, 0.00002])
    duration = 0.1

    result = sv.run_model(
        float(np.sum(dx_m)),
        duration,
        h_init=np.full(3, 0.2),
        q_init=np.zeros(3),
        rainfall=lambda x, t: rainfall_rate,
        x_m=x_m,
        dx_m=dx_m,
        slope=np.zeros(3),
        manning_n=np.full(3, 0.001),
    )

    expected_source = float(np.sum(rainfall_rate * dx_m) * duration)
    storage_delta = float(np.sum((result["h_final"] - result["h_initial"]) * dx_m))
    assert result["mass_source"] == pytest.approx(expected_source)
    assert storage_delta == pytest.approx(
        result["mass_source"]
        - result["mass_outflow"]
        + result["mass_inflow"]
        + result["mass_floor_correction"],
        abs=1e-12,
    )


def test_rainfall_function_rejects_negative_rates():
    with pytest.raises(ValueError, match="non-negative"):
        sv.run_model(
            sv.L,
            0.01,
            rainfall=lambda x, t: np.full_like(x, -0.001),
        )


def test_rainfall_function_is_evaluated_during_time_stepping():
    evaluation_times = []

    def rainfall(x, t):
        evaluation_times.append(t)
        return np.full_like(x, 0.00001 if t < 0.01 else 0.00002)

    sv.run_model(sv.L, 0.02, rainfall=rainfall)

    assert len(evaluation_times) > 1
    assert min(evaluation_times) == 0.0
    assert max(evaluation_times) > 0.0
