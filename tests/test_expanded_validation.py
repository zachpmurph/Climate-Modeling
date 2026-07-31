import csv
import json
from pathlib import Path

from rivers.validation.analyze_expanded_events import analyze_expanded_events


ROOT = Path(__file__).resolve().parents[1] / "real_world_rivers" / "validation"
MANIFEST = ROOT / "expanded_river_error_assessment.json"


def test_expanded_events_are_distinct_approved_2d_rivers():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rivers = set()
    for relative_path in manifest["cases"]:
        config_path = ROOT / relative_path
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result = json.loads(
            config_path.with_suffix(".results.json").read_text(encoding="utf-8")
        )
        with (ROOT / config["observations"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        rivers.add(config["case"]["river"])
        assert config["validation_policy"]["calibration"] == "none"
        assert config["validation_2d"]["representation"] == "ribbon"
        assert result["solver"] == "saint_venant_2d"
        assert {row["approval_status"] for row in rows} == {"Approved"}
        assert {row["role"] for row in rows} == {"upstream", "downstream"}
    assert len(rivers) == len(manifest["cases"]) == 5


def test_expanded_event_analysis_is_reproducible(tmp_path):
    actual = analyze_expanded_events(
        MANIFEST, output_path=tmp_path / "assessment.json"
    )
    tracked = json.loads(
        MANIFEST.with_suffix(".results.json").read_text(encoding="utf-8")
    )
    assert actual == tracked
    assert actual["case_count"] == 5
    associations = actual["descriptive_associations"]
    assert associations[
        "drainage_growth_vs_modeled_volume_deficit_pearson_r"
    ] > 0.9
    assert associations["modeled_routing_too_fast_case_count"] >= 4
    russian = next(
        item
        for item in actual["structural_sensitivity"]
        if item["river"] == "Russian River"
    )
    assert abs(russian["variant_routing_lag_error_min"]) < abs(
        russian["baseline_routing_lag_error_min"]
    )
    assert russian["variant_pearson_r"] > russian["baseline_pearson_r"]
    assert abs(
        russian["variant_volume_ratio"] - russian["baseline_volume_ratio"]
    ) < 0.02
    assert [item["rank"] for item in actual["ranked_hypotheses"]] == [
        1,
        2,
        3,
        4,
        5,
    ]
