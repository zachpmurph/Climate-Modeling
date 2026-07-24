"""Structured validation for model-ready river profiles.

Findings are graded by severity:

* ``error``   -- blocks profile export; the data is unusable or self-inconsistent.
* ``warning`` -- retained in metadata and batch summaries; the value is
  suspicious or the provenance is incomplete, but export can proceed.
* ``info``    -- neutral context (e.g. classification coverage counts).

The validators operate on plain Python structures (lists of profile-row dicts,
coordinate tuples, timestamp strings) rather than the database, so they are
cheap to unit-test in isolation. ``export_profile`` assembles these structures
and calls :func:`validate_profile` before writing anything to disk.
"""

import math
from dataclasses import dataclass, field

ERROR = "error"
WARNING = "warning"
INFO = "info"

# Plausible ranges for lowland-to-mountain natural channels. Values outside
# these ranges are flagged as warnings (suspicious), never hard errors, because
# extreme-but-real reaches exist and the modeller should decide.
SLOPE_PLAUSIBLE_MIN = 1e-6  # slope is dimensionless, so unit conventions don't shift it
SLOPE_PLAUSIBLE_MAX = 0.1
# Manning's n range must span BOTH conventions present in this repo pending a
# ruling (see docs/ingestion_integration_requests.md): standard SI n (~0.01-0.2)
# and the meters-and-minutes converted n (SI ÷ 60, ~1.5e-4 to 3.3e-3, as used by
# the curated Columbia data). The lower bound covers the converted convention so
# legitimately-converted values are not flagged as suspicious.
ROUGHNESS_PLAUSIBLE_MIN = 1e-4
ROUGHNESS_PLAUSIBLE_MAX = 0.2

CLASSIFICATION_LEVELS = ("observed", "derived", "estimated", "fallback")

# Fields whose provenance classification is expected on every exported row.
PROVENANCE_FIELDS = ("station_m", "slope", "manning_n")


class ProfileValidationError(ValueError):
    """Raised when a profile has validation errors and must not be exported.

    The :class:`ValidationReport` is attached as ``.report`` so callers can
    surface the individual error findings.
    """

    def __init__(self, report):
        self.report = report
        error_messages = "; ".join(f.message for f in report.errors)
        super().__init__(f"Profile validation failed with {len(report.errors)} error(s): {error_messages}")


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str
    context: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class ValidationReport:
    findings: list = field(default_factory=list)

    def add(self, severity, code, message, **context):
        self.findings.append(Finding(severity, code, message, context))

    def extend(self, findings):
        self.findings.extend(findings)

    def _by(self, severity):
        return [f for f in self.findings if f.severity == severity]

    @property
    def errors(self):
        return self._by(ERROR)

    @property
    def warnings(self):
        return self._by(WARNING)

    @property
    def infos(self):
        return self._by(INFO)

    @property
    def has_errors(self):
        return any(f.severity == ERROR for f in self.findings)

    def to_metadata(self):
        return [f.to_dict() for f in self.findings]

    def counts(self):
        return {
            "error": len(self.errors),
            "warning": len(self.warnings),
            "info": len(self.infos),
        }


def _is_finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _check_finite(report, rows):
    for index, row in enumerate(rows):
        for key in ("station_m", "slope", "manning_n"):
            if not _is_finite_number(row.get(key)):
                report.add(
                    ERROR, "non_finite",
                    f"Row {index}: {key} is missing or not a finite number ({row.get(key)!r}).",
                    row=index, field=key, value=repr(row.get(key)),
                )
        rain = row.get("rainfall_rate_m_per_min")
        if rain is not None and not _is_finite_number(rain):
            report.add(
                ERROR, "non_finite",
                f"Row {index}: rainfall_rate_m_per_min is not finite ({rain!r}).",
                row=index, field="rainfall_rate_m_per_min",
            )


def _check_station_ordering(report, rows):
    stations = [row.get("station_m") for row in rows]
    if len(stations) < 2:
        report.add(ERROR, "station_count", "A profile needs at least two stations.")
        return
    numeric = [s for s in stations if _is_finite_number(s)]
    if len(numeric) < 2:
        return  # finite check already reported the offending rows
    for index in range(1, len(stations)):
        prev, curr = stations[index - 1], stations[index]
        if not (_is_finite_number(prev) and _is_finite_number(curr)):
            continue
        if curr == prev:
            report.add(
                ERROR, "station_duplicate",
                f"Rows {index - 1} and {index} share station {curr:g} m (duplicate).",
                rows=[index - 1, index], station_m=curr,
            )
        elif curr < prev:
            report.add(
                ERROR, "station_order",
                f"Row {index}: station {curr:g} m is not downstream of the previous "
                f"station {prev:g} m (stations must strictly increase upstream to downstream).",
                row=index, previous_m=prev, station_m=curr,
            )
    # Gap check: flag a spacing far larger than the median as a possible
    # coverage hole (warning only).
    diffs = [b - a for a, b in zip(numeric, numeric[1:]) if b > a]
    if diffs:
        ordered = sorted(diffs)
        median = ordered[len(ordered) // 2]
        for index, gap in enumerate(diffs, start=1):
            if median > 0 and gap > 5 * median:
                report.add(
                    WARNING, "station_gap",
                    f"Row {index}: spacing {gap:g} m is more than 5x the median "
                    f"spacing {median:g} m; the reach may be under-sampled here.",
                    row=index, gap_m=gap, median_gap_m=median,
                )


def _check_slope(report, rows):
    for index, row in enumerate(rows):
        slope = row.get("slope")
        if not _is_finite_number(slope):
            continue
        if slope <= 0:
            report.add(
                ERROR, "slope_non_positive",
                f"Row {index}: slope {slope:g} is not positive; the model requires positive slope.",
                row=index, slope=slope,
            )
        elif not (SLOPE_PLAUSIBLE_MIN <= slope <= SLOPE_PLAUSIBLE_MAX):
            report.add(
                WARNING, "slope_range",
                f"Row {index}: slope {slope:g} is outside the plausible range "
                f"[{SLOPE_PLAUSIBLE_MIN:g}, {SLOPE_PLAUSIBLE_MAX:g}]; verify the elevation data.",
                row=index, slope=slope,
            )


def _check_roughness(report, rows):
    for index, row in enumerate(rows):
        n = row.get("manning_n")
        if not _is_finite_number(n):
            continue
        if n <= 0:
            report.add(
                ERROR, "roughness_non_positive",
                f"Row {index}: manning_n {n:g} is not positive.",
                row=index, manning_n=n,
            )
        elif not (ROUGHNESS_PLAUSIBLE_MIN <= n <= ROUGHNESS_PLAUSIBLE_MAX):
            report.add(
                WARNING, "roughness_range",
                f"Row {index}: manning_n {n:g} is outside the typical natural-channel "
                f"range [{ROUGHNESS_PLAUSIBLE_MIN:g}, {ROUGHNESS_PLAUSIBLE_MAX:g}]; verify the source.",
                row=index, manning_n=n,
            )


def _check_rainfall(report, rows):
    for index, row in enumerate(rows):
        rain = row.get("rainfall_rate_m_per_min")
        if rain is None or not _is_finite_number(rain):
            continue
        if rain < 0:
            report.add(
                ERROR, "rainfall_negative",
                f"Row {index}: rainfall_rate_m_per_min {rain:g} is negative.",
                row=index, rainfall_rate_m_per_min=rain,
            )
        elif rain > 0.01:  # >10 mm/min sustained is physically extreme
            report.add(
                WARNING, "rainfall_range",
                f"Row {index}: rainfall_rate_m_per_min {rain:g} (~{rain * 60000:g} mm/hr) "
                "is implausibly high; check the aggregation window.",
                row=index, rainfall_rate_m_per_min=rain,
            )


def _check_coordinates(report, coordinates):
    if not coordinates:
        return
    for index, pair in enumerate(coordinates):
        try:
            lat, lon = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError):
            report.add(ERROR, "coordinate_range", f"Row {index}: coordinate {pair!r} is not a (lat, lon) pair.",
                       row=index)
            continue
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            report.add(
                ERROR, "coordinate_range",
                f"Row {index}: coordinate ({lat}, {lon}) is outside valid WGS84 bounds.",
                row=index, lat=lat, lon=lon,
            )


def _check_provenance(report, rows):
    counts = {level: 0 for level in CLASSIFICATION_LEVELS}
    unknown = 0
    for index, row in enumerate(rows):
        classification = row.get("classification") or {}
        for prov_field in PROVENANCE_FIELDS:
            if row.get(prov_field) is None:
                continue
            label = classification.get(prov_field)
            if label is None:
                report.add(
                    WARNING, "provenance_incomplete",
                    f"Row {index}: {prov_field} has no provenance classification.",
                    row=index, field=prov_field,
                )
            elif label in counts:
                counts[label] += 1
            else:
                unknown += 1
                report.add(
                    WARNING, "provenance_unknown",
                    f"Row {index}: {prov_field} has unrecognized classification {label!r}.",
                    row=index, field=prov_field, classification=label,
                )
    total_classified = sum(counts.values())
    report.add(
        INFO, "classification_summary",
        "Exported-value provenance: "
        + ", ".join(f"{level}={counts[level]}" for level in CLASSIFICATION_LEVELS)
        + (f", unknown={unknown}" if unknown else ""),
        **counts, unknown=unknown,
    )
    if total_classified and counts["observed"] == 0:
        report.add(
            WARNING, "observed_coverage",
            "No exported value is classified 'observed'; the profile rests entirely on "
            "derived, estimated, or fallback inputs and is not calibrated.",
            **counts,
        )


def validate_temporal_coverage(times, start, end, *, label="observation", max_gap_hours=24.0):
    """Validate that ``times`` (ISO-8601 strings) cover the [start, end] window.

    Returns a list of :class:`Finding`. Flags: coverage that falls short of the
    requested window (``temporal_coverage``) and internal gaps larger than
    ``max_gap_hours`` (``temporal_gap``). Timezone handling is intentionally
    simple: naive ISO timestamps are compared lexically for coverage and parsed
    for gap sizing; unparseable timestamps produce an info finding.
    """
    from datetime import datetime, timezone

    def _parse_utc(stamp):
        # Normalize every timestamp to aware-UTC so naive provider timestamps
        # (Open-Meteo rainfall) and 'Z'-suffixed ones (USGS flow) can be
        # compared without raising TypeError on naive/aware subtraction.
        value = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value

    findings = []
    parsed = []
    for stamp in times:
        try:
            parsed.append(_parse_utc(stamp))
        except (ValueError, AttributeError):
            findings.append(Finding(INFO, "temporal_unparsed",
                                     f"{label}: could not parse timestamp {stamp!r}.", {"timestamp": stamp}))
    if not parsed:
        findings.append(Finding(WARNING, "temporal_coverage", f"{label}: no usable timestamps.", {}))
        return findings
    parsed.sort()
    try:
        window_start = _parse_utc(start)
        window_end = _parse_utc(end)
    except (ValueError, AttributeError):
        window_start = window_end = None

    if window_start is not None:
        # Coverage shortfall at either end of the requested window.
        lead = (parsed[0] - window_start).total_seconds() / 3600.0
        trail = (window_end - parsed[-1]).total_seconds() / 3600.0
        if lead > max_gap_hours or trail > max_gap_hours:
            findings.append(Finding(
                WARNING, "temporal_coverage",
                f"{label}: observations cover {parsed[0].isoformat()}..{parsed[-1].isoformat()}, "
                f"short of the requested window {start}..{end}.",
                {"first": parsed[0].isoformat(), "last": parsed[-1].isoformat(),
                 "lead_gap_hours": lead, "trail_gap_hours": trail},
            ))

    for earlier, later in zip(parsed, parsed[1:]):
        gap_hours = (later - earlier).total_seconds() / 3600.0
        if gap_hours > max_gap_hours:
            findings.append(Finding(
                WARNING, "temporal_gap",
                f"{label}: {gap_hours:g} h gap between {earlier.isoformat()} and {later.isoformat()}.",
                {"gap_hours": gap_hours, "from": earlier.isoformat(), "to": later.isoformat()},
            ))
    return findings


def validate_profile(rows, *, coordinates=None, adjustments=None):
    """Validate assembled profile rows and return a :class:`ValidationReport`.

    ``rows`` is a list of dicts with at least ``station_m``, ``slope`` and
    ``manning_n``, optionally ``rainfall_rate_m_per_min``/``initial_depth_m`` and
    a per-field ``classification`` dict. ``coordinates`` (optional) is a list of
    ``(lat, lon)`` aligned to rows. ``adjustments`` (optional) is a list of
    adjustment records that are echoed into the report as info findings.
    """
    report = ValidationReport()
    if not rows:
        report.add(ERROR, "empty_profile", "The profile has no rows.")
        return report
    _check_finite(report, rows)
    _check_station_ordering(report, rows)
    _check_slope(report, rows)
    _check_roughness(report, rows)
    _check_rainfall(report, rows)
    _check_coordinates(report, coordinates)
    _check_provenance(report, rows)
    for adjustment in adjustments or []:
        report.add(
            INFO, "adjustment",
            f"{adjustment.get('rule', 'adjustment')} applied to "
            f"{adjustment.get('count', '?')} value(s).",
            **adjustment,
        )
    return report
