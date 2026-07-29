import json

import numpy as np
import pytest

from rivers.simulations import run_2d_ensemble as ensemble
from rivers.reporting.generate_uncertainty_report import (
    generate_report,
    load_ensemble_fields,
)


PROFILE_PATH = "real_world_rivers/tools/example_river_profile.csv"
GEOMETRY_PATH = "real_world_rivers/tools/example_geometry.csv"


def _config(**overrides):
    config = {
        "schema_version": 1,
        "sample_count": 4,
        "seed": 17,
        "bounds": {
            name: (0.8, 1.2) for name in ensemble.PARAMETER_NAMES
        },
        "quantiles": np.array([0.05, 0.5, 0.95]),
        "wet_depth_threshold_m": 0.01,
        "retain_member_depth": False,
    }
    config.update(overrides)
    return config


def test_load_ensemble_config_rejects_unknown_or_invalid_ranges(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_count": 1,
                "seed": 0,
                "parameter_scales": {"manning_scale": [0.8, 1.2]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sample_count"):
        ensemble.load_ensemble_config(path)

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_count": 2,
                "seed": 0,
                "parameter_scales": {"unknown": [0.8, 1.2]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown"):
        ensemble.load_ensemble_config(path)


def test_latin_hypercube_is_seeded_bounded_and_stratified():
    config = _config(sample_count=8)
    first = ensemble.sample_parameter_scales(config)
    second = ensemble.sample_parameter_scales(config)

    assert np.array_equal(first, second)
    assert np.all(first >= 0.8)
    assert np.all(first <= 1.2)
    normalized = (first - 0.8) / 0.4
    for column in range(normalized.shape[1]):
        strata = np.floor(normalized[:, column] * 8).astype(int)
        assert sorted(strata) == list(range(8))


def test_explicit_joint_samples_preserve_parameter_dependence(tmp_path):
    path = tmp_path / "joint.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "quantiles": [0.1, 0.9],
                "parameter_samples": [
                    {"manning_scale": 0.8, "channel_width_scale": 1.2},
                    {"manning_scale": 1.2, "channel_width_scale": 0.8},
                ],
            }
        ),
        encoding="utf-8",
    )
    config = ensemble.load_ensemble_config(path)
    samples = ensemble.sample_parameter_scales(config)

    assert config["method"] == "explicit_joint_samples"
    assert config["sample_count"] == 2
    assert config["seed"] is None
    manning_index = ensemble.PARAMETER_NAMES.index("manning_scale")
    width_index = ensemble.PARAMETER_NAMES.index("channel_width_scale")
    assert np.allclose(
        samples[:, [manning_index, width_index]],
        [[0.8, 1.2], [1.2, 0.8]],
    )
    fixed_columns = [
        index
        for index in range(len(ensemble.PARAMETER_NAMES))
        if index not in {manning_index, width_index}
    ]
    for index in fixed_columns:
        expected = (
            0.0
            if ensemble.PARAMETER_NAMES[index]
            == "downstream_stage_offset_m"
            else 1.0
        )
        assert np.all(samples[:, index] == expected)


def test_field_summary_reports_quantiles_probability_and_wet_area():
    member_depth = np.array(
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.2], [0.0, 0.0]],
            ],
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.3], [0.4, 0.0]],
            ],
        ]
    )
    result = ensemble.summarize_member_fields(
        member_depth,
        dx_m=np.array([10.0, 20.0]),
        dy_m=np.array([2.0, 2.0]),
        quantiles=np.array([0.0, 0.5, 1.0]),
        wet_depth_threshold_m=0.1,
    )

    assert np.allclose(
        result["wet_probability"],
        [[0.0, 1.0], [0.5, 0.0]],
    )
    assert result["member_maximum_wet_area_m2"] == pytest.approx(
        [20.0, 60.0]
    )
    assert result["maximum_wet_area_quantiles_m2"] == pytest.approx(
        [20.0, 40.0, 60.0]
    )
    assert np.all(
        np.diff(result["depth_quantiles_m"], axis=0) >= 0.0
    )


def test_cli_writes_reproducible_2d_uncertainty_artifacts(tmp_path):
    config_path = tmp_path / "ensemble.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_count": 3,
                "seed": 42,
                "quantiles": [0.1, 0.5, 0.9],
                "wet_depth_threshold_m": 0.001,
                "parameter_scales": {
                    "manning_scale": [0.9, 1.1],
                    "longitudinal_slope_scale": [0.9, 1.1],
                    "channel_width_scale": [0.9, 1.1],
                    "bankfull_depth_scale": [0.9, 1.1],
                    "floodplain_slope_scale": [0.9, 1.1],
                    "inflow_scale": [1.0, 1.0],
                    "rainfall_scale": [1.0, 1.0],
                    "downstream_stage_offset_m": [-0.01, 0.01],
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "runs"
    args = [
        PROFILE_PATH,
        "--hydraulic-geometry",
        GEOMETRY_PATH,
        "--ensemble-config",
        str(config_path),
        "--width",
        "100",
        "--cross-cells",
        "8",
        "--t-final",
        "0.01",
        "--record-interval",
        "0.01",
        "--downstream-stage",
        "0.04",
        "--output-dir",
        str(output_dir),
        "--run-name",
        "screening",
    ]
    ensemble.main(args)

    fields_path = output_dir / "screening_ensemble.npz"
    summary_path = output_dir / "screening_ensemble_summary.json"
    first = np.load(fields_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert first["parameter_scales"].shape == (
        3,
        len(ensemble.PARAMETER_NAMES),
    )
    stage_index = ensemble.PARAMETER_NAMES.index(
        "downstream_stage_offset_m"
    )
    assert np.min(first["parameter_scales"][:, stage_index]) < 0.0
    assert np.max(first["parameter_scales"][:, stage_index]) > 0.0
    assert first["depth_quantiles_m"].shape == (3, 2, 5, 8)
    assert first["peak_depth_quantiles_m"].shape == (3, 5, 8)
    assert first["bed_elevation_quantiles_m"].shape == (3, 5, 8)
    assert first["water_surface_elevation_quantiles_m"].shape == (
        3,
        2,
        5,
        8,
    )
    assert first["peak_water_surface_elevation_quantiles_m"].shape == (
        3,
        5,
        8,
    )
    assert first["wet_probability"].shape == (5, 8)
    assert np.all((first["wet_probability"] >= 0.0))
    assert np.all((first["wet_probability"] <= 1.0))
    assert np.all(np.diff(first["depth_quantiles_m"], axis=0) >= 0.0)
    assert summary["sample_count"] == 3
    assert summary["method"] == "latin_hypercube"
    assert summary["context"]["hydraulic_geometry"] == GEOMETRY_PATH
    assert summary["context"]["downstream_stage_m"] == pytest.approx(0.04)
    assert "not calibrated forecast probabilities" in summary["interpretation"]
    assert summary["maximum_absolute_mass_balance_error_m3"] == pytest.approx(
        0.0
    )

    saved_scales = first["parameter_scales"].copy()
    first.close()
    report_path = generate_report(fields_path)
    report = report_path.read_text(encoding="utf-8")
    loaded = load_ensemble_fields(fields_path)
    assert loaded.parameter_scales.shape == (3, len(ensemble.PARAMETER_NAMES))
    assert "Spatial outcome band" in report
    assert "Probability depth exceeded 0.001 m" in report
    assert "not calibrated forecast probabilities" in report

    ensemble.main(args)
    with np.load(fields_path) as repeated:
        assert np.array_equal(repeated["parameter_scales"], saved_scales)
