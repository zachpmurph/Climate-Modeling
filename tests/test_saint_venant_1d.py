import numpy as np
import pytest

from floods import saint_venant_1d as sv


def test_still_water_at_rest(monkeypatch):
    # Flat h=0.5, q=0, no rain, no slope — cells far from the left BC
    # should remain exactly undisturbed.
    #
    # The left BC pins h[0] ~ 0, which creates a step gradient that
    # propagates rightward. Each LxF step spreads this disturbance at
    # most 1 cell (±1 stencil), so after n steps only cells 1..n are
    # affected. With h=0.5 the CFL dt ≈ 3.76e-4 min; T_final=1e-3 min
    # executes ~3 steps, touching at most cells 1..3. Cells 5..Nx-2 are
    # exactly unaffected and should equal the initial condition to machine
    # precision.
    monkeypatch.setattr(sv, "S0", 0.0)

    def no_rain(x, t):
        return np.zeros_like(x)

    monkeypatch.setattr(sv, "r", no_rain)

    Nx = int(sv.L * 10)
    h0 = 0.5 * np.ones(Nx)
    q0 = np.zeros(Nx)

    result = sv.run_model(sv.L, 1e-3, h_init=h0, q_init=q0)

    h_final = result["h_final"]
    q_final = result["q_final"]

    assert np.allclose(h_final[5:-1], 0.5, rtol=1e-10), \
        f"Still water disturbed: max deviation = {np.max(np.abs(h_final[5:-1] - 0.5))}"
    assert np.allclose(q_final[5:-1], 0.0, atol=1e-15), \
        f"Discharge non-zero: max = {np.max(np.abs(q_final[5:-1]))}"
