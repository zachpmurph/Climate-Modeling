import math

import numpy as np
import pytest

from floods import saint_venant_1d as sv


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

    assert np.allclose(result["h_final"], h0, rtol=0, atol=1e-14)
    assert np.allclose(result["q_final"], q0, rtol=0, atol=2e-13)


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

    assert np.all(result["h_final"] == sv.H_FLOOR)
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
