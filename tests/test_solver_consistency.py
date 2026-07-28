"""Cross-solver consistency guardrail.

The kinematic-wave (linear_advection) and Saint-Venant (saint_venant_1d) solvers
are different physics, but for steady uniform flow they must both converge to the
SAME Manning normal depth for the same profile:

    q0 = (1/n) * h_n**(5/3) * sqrt(S)   ->   h_n = (q0 * n / sqrt(S))**(3/5)

This is the check that catches a Manning-n unit-convention mismatch between the
two solver families: if one interpreted n under a different (e.g. seconds vs
minutes) convention, its normal depth would be off by a large factor. Both
solvers must speak the SAME convention as the profile files (meters-and-minutes,
i.e. Manning n = SI n / 60).
"""
import numpy as np
import pytest

from general.solvers import linear_advection as la
from general.solvers import saint_venant_1d as sv1


def _normal_depth(q0, manning_n, slope):
    return (q0 * manning_n / np.sqrt(slope)) ** (3.0 / 5.0)


# Meters-and-minutes Manning n (SI 0.036 / 60), a realistic converted value.
SLOPE = 0.01
MANNING_N = 0.0006
Q0 = 0.05  # upstream inflow, m^2/min
T_FINAL = 90.0
L = 10.0
NX = int(L * 10)


def test_kinematic_wave_converges_to_manning_normal_depth():
    h_n = _normal_depth(Q0, MANNING_N, SLOPE)
    profile = la.make_profile(
        station_m=np.linspace(0.0, L, NX),
        slope=np.full(NX, SLOPE),
        manning_n=np.full(NX, MANNING_N),
    )
    result = la.run_model(profile, t_final_min=T_FINAL, left_inflow_flux=Q0, base_depth_m=h_n)
    interior = result["depth_final"][NX // 2]
    assert interior == pytest.approx(h_n, rel=0.1)


def test_saint_venant_converges_to_manning_normal_depth():
    h_n = _normal_depth(Q0, MANNING_N, SLOPE)
    result = sv1.run_model(
        L, T_FINAL,
        h_init=np.full(NX, h_n),
        left_inflow=Q0,
        slope=np.full(NX, SLOPE),
        manning_n=np.full(NX, MANNING_N),
    )
    interior = result["h_final"][NX // 2]
    assert interior == pytest.approx(h_n, rel=0.1)


def test_two_solvers_agree_on_steady_depth_for_same_profile():
    """The decisive cross-solver check: same profile -> same steady depth."""
    profile = la.make_profile(
        station_m=np.linspace(0.0, L, NX),
        slope=np.full(NX, SLOPE),
        manning_n=np.full(NX, MANNING_N),
    )
    h_n = _normal_depth(Q0, MANNING_N, SLOPE)
    kw = la.run_model(profile, t_final_min=T_FINAL, left_inflow_flux=Q0, base_depth_m=h_n)
    sv = sv1.run_model(
        L, T_FINAL, h_init=np.full(NX, h_n), left_inflow=Q0,
        slope=np.full(NX, SLOPE), manning_n=np.full(NX, MANNING_N),
    )
    kw_depth = kw["depth_final"][NX // 2]
    sv_depth = sv["h_final"][NX // 2]
    assert kw_depth == pytest.approx(sv_depth, rel=0.1)
