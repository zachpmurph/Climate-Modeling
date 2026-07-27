"""Unit tests for the profile validation layer.

Validation classifies findings as error / warning / info. Errors must block
export; warnings and infos are retained in metadata.
"""

import math

from rivers.ingest.validation import (
    ValidationReport,
    validate_profile,
    validate_temporal_coverage,
)


def _row(station_m, slope=0.001, manning_n=0.035, **extra):
    row = {
        "station_m": station_m,
        "slope": slope,
        "manning_n": manning_n,
        "classification": {"station_m": "observed", "slope": "derived", "manning_n": "estimated"},
    }
    row.update(extra)
    return row


def _good_rows():
    return [_row(0.0), _row(1000.0), _row(2000.0)]


def test_valid_profile_reports_no_errors():
    report = validate_profile(_good_rows())
    assert isinstance(report, ValidationReport)
    assert not report.has_errors
    assert report.errors == []


def test_non_increasing_station_is_error():
    rows = [_row(0.0), _row(1000.0), _row(900.0)]
    report = validate_profile(rows)
    assert report.has_errors
    assert any(f.code == "station_order" for f in report.errors)


def test_duplicate_station_is_error():
    rows = [_row(0.0), _row(1000.0), _row(1000.0)]
    report = validate_profile(rows)
    assert any(f.code == "station_duplicate" for f in report.errors)


def test_non_positive_slope_is_error():
    rows = [_row(0.0, slope=0.0), _row(1000.0), _row(2000.0)]
    report = validate_profile(rows)
    assert any(f.code == "slope_non_positive" for f in report.errors)


def test_non_finite_value_is_error():
    rows = [_row(0.0, manning_n=math.nan), _row(1000.0), _row(2000.0)]
    report = validate_profile(rows)
    assert any(f.code == "non_finite" for f in report.errors)


def test_suspicious_roughness_is_warning_not_error():
    rows = [_row(0.0, manning_n=0.9), _row(1000.0), _row(2000.0)]
    report = validate_profile(rows)
    assert not report.has_errors
    assert any(f.code == "roughness_range" for f in report.warnings)


def test_suspicious_slope_is_warning_not_error():
    rows = [_row(0.0, slope=0.9), _row(1000.0), _row(2000.0)]
    report = validate_profile(rows)
    assert not report.has_errors
    assert any(f.code == "slope_range" for f in report.warnings)


def test_negative_rainfall_is_error():
    rows = [_row(0.0, rainfall_rate_m_per_min=-1e-6), _row(1000.0), _row(2000.0)]
    report = validate_profile(rows)
    assert any(f.code == "rainfall_negative" for f in report.errors)


def test_invalid_coordinate_is_error():
    report = validate_profile(_good_rows(), coordinates=[(40.0, -120.0), (91.0, -120.0), (39.0, -120.0)])
    assert any(f.code == "coordinate_range" for f in report.errors)


def test_missing_provenance_classification_is_warning():
    rows = _good_rows()
    rows[1]["classification"] = {"station_m": "observed", "slope": "derived"}  # manning_n missing
    report = validate_profile(rows)
    assert any(f.code == "provenance_incomplete" for f in report.warnings)


def test_report_summarizes_classification_coverage_as_info():
    report = validate_profile(_good_rows())
    infos = [f for f in report.infos if f.code == "classification_summary"]
    assert infos, "expected a classification coverage summary info finding"
    assert infos[0].context["estimated"] >= 1


def test_report_to_metadata_is_json_friendly():
    rows = [_row(0.0), _row(1000.0), _row(900.0)]
    report = validate_profile(rows)
    payload = report.to_metadata()
    assert isinstance(payload, list)
    assert all({"severity", "code", "message"} <= set(item) for item in payload)


def test_temporal_coverage_flags_gap_and_short_window():
    # Hourly data with a large gap and coverage shorter than the requested window.
    times = ["2020-01-01T00:00", "2020-01-01T01:00", "2020-01-03T00:00"]
    findings = validate_temporal_coverage(times, "2020-01-01T00:00", "2020-01-05T00:00", label="rainfall")
    codes = {f.code for f in findings}
    assert "temporal_gap" in codes
    assert "temporal_coverage" in codes


def test_temporal_coverage_handles_mixed_naive_and_utc_timestamps():
    # Flow timestamps carry a trailing Z (aware); the window may be naive.
    # Subtracting aware and naive datetimes would raise TypeError if not normalized.
    times = ["2020-01-01T00:00:00Z", "2020-01-01T01:00:00Z"]
    findings = validate_temporal_coverage(times, "2020-01-01T00:00", "2020-01-01T02:00", label="flow")
    # Must not raise; findings is a list (possibly with a coverage warning).
    assert isinstance(findings, list)
