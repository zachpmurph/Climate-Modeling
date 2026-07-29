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
            in ensemble.OFFSET_PARAMETER_NAMES
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


def test_cli_runs_uncertainty_on_reviewed_terrain(tmp_path):
    terrain_path = tmp_path / "terrain.csv"
    rows = [
        "x_m,y_m,dx_m,dy_m,bed_elevation_m,manning_n",
    ]
    for x, base in ((0, 0.0), (2000, -2.0), (4000, -3.4)):
        for y, relief, roughness in (
            (0, 1.0, 0.0012),
            (10, 0.0, 0.0007),
            (20, 1.2, 0.0015),
        ):
            rows.append(
                f"{x},{y},1000,10,{base + relief},{roughness}"
            )
    terrain_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    config_path = tmp_path / "terrain_ensemble.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_count": 3,
                "seed": 9,
                "quantiles": [0.1, 0.5, 0.9],
                "parameter_scales": {
                    "manning_scale": [0.9, 1.1],
                    "terrain_elevation_offset_m": [-0.05, 0.05],
                    "terrain_relief_scale": [0.8, 1.2],
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "runs"

    ensemble.main(
        [
            PROFILE_PATH,
            "--terrain-grid",
            str(terrain_path),
            "--ensemble-config",
            str(config_path),
            "--t-final",
            "0.01",
            "--record-interval",
            "0.01",
            "--output-dir",
            str(output_dir),
            "--run-name",
            "terrain",
        ]
    )

    with np.load(output_dir / "terrain_ensemble.npz") as fields:
        assert fields["depth_quantiles_m"].shape == (3, 2, 3, 3)
        assert fields["bed_elevation_quantiles_m"].shape == (3, 3, 3)
        assert fields["manning_n_quantiles"].shape == (3, 3, 3)
        assert np.any(
            fields["bed_elevation_quantiles_m"][0]
            != fields["bed_elevation_quantiles_m"][-1]
        )
        assert np.max(
            np.abs(fields["member_mass_balance_error_m3"])
        ) < 1e-10

    summary = json.loads(
        (
            output_dir / "terrain_ensemble_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["context"]["terrain_grid"] == str(terrain_path)
    assert summary["context"]["terrain_source"] == "reviewed_grid"
    assert summary["context"]["roughness_source"] == "terrain_grid"


def test_reviewed_terrain_rejects_synthetic_geometry_uncertainty(tmp_path):
    from general.solvers.profile import load_profile, load_reviewed_terrain

    terrain_path = tmp_path / "terrain.csv"
    terrain_path.write_text(
        "x_m,y_m,dx_m,dy_m,bed_elevation_m\n"
        "0,0,1,1,0\n0,1,1,1,1\n"
        "1,0,1,1,-1\n1,1,1,1,0\n",
        encoding="utf-8",
    )
    profile = load_profile(PROFILE_PATH)
    domain, _ = load_reviewed_terrain(terrain_path, profile)
    config = _config()
    width_index = ensemble.PARAMETER_NAMES.index("channel_width_scale")
    samples = np.ones((2, len(ensemble.PARAMETER_NAMES)))
    for name in ensemble.OFFSET_PARAMETER_NAMES:
        samples[:, ensemble.PARAMETER_NAMES.index(name)] = 0.0
    samples[:, width_index] = [0.9, 1.1]
    config["samples"] = samples
    config["sample_count"] = 2

    with pytest.raises(ValueError, match="synthetic-terrain uncertainty"):
        ensemble.run_ensemble(
            profile,
            None,
            None,
            config,
            base_domain=domain,
            t_final_min=0.0,
        )
