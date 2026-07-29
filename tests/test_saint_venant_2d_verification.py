"""Quantitative verification gates for the 2-D Saint-Venant solver."""

from general.verification.verify_saint_venant_2d import (
    analytic_diagonal_vortex_wave,
    analytic_shear_wave,
    manufactured_pressure_wave,
    nonflat_lake_at_rest,
    partially_dry_nonflat_lake_at_rest,
    one_dimensional_reduction,
    radial_dam_break_symmetry,
    strict_periodic_mass_conservation,
    wet_dry_dam_break,
)


def _assert_acceptance(case):
    failed = [name for name, passed in case["acceptance"].items() if not passed]
    assert not failed, f"{case['case']} failed verification gates: {failed}; {case}"


def test_analytic_shear_wave_converges_at_first_order():
    case = analytic_shear_wave()
    _assert_acceptance(case)
    assert len(case["resolutions"]) >= 3
    assert all(
        left > right
        for left, right in zip(
            case["l2_velocity_y_errors"],
            case["l2_velocity_y_errors"][1:],
        )
    )


def test_analytic_diagonal_wave_converges_at_first_order():
    _assert_acceptance(analytic_diagonal_vortex_wave())


def test_manufactured_pressure_wave_converges_at_first_order():
    _assert_acceptance(manufactured_pressure_wave())


def test_nonflat_lake_at_rest_is_well_balanced():
    _assert_acceptance(nonflat_lake_at_rest())


def test_partially_dry_nonflat_lake_at_rest_preserves_shoreline():
    _assert_acceptance(partially_dry_nonflat_lake_at_rest())


def test_two_dimensional_solver_reduces_to_one_dimensional_solver():
    _assert_acceptance(one_dimensional_reduction())


def test_radial_dam_break_remains_radially_symmetric():
    _assert_acceptance(radial_dam_break_symmetry())


def test_periodic_mass_conservation_is_at_machine_precision():
    _assert_acceptance(strict_periodic_mass_conservation())


def test_wet_dry_dam_break_is_positive_and_conservative():
    _assert_acceptance(wet_dry_dam_break())
