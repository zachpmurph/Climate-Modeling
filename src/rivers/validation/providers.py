"""Small provider client used only to refresh committed validation fixtures.

This module intentionally has no database, reach-building, or generic ingestion
capabilities.  Validation cases remain offline and reproducible; these helpers
only rebuild their committed USGS evidence when explicitly requested.
"""

from __future__ import annotations

import json
import time
import warnings
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


USER_AGENT = "Climate-Modeling validation-fixture client/1.0"
CONTINUOUS_URL = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items"
)
DISCHARGE_PARAMETER = "00060"
CFS_TO_M3_PER_MIN = 0.028316846592 * 60.0
SENSITIVE_QUERY_KEYS = frozenset(
    {"api_key", "apikey", "key", "token", "access_token", "auth", "password", "secret"}
)


class ProviderRequestError(RuntimeError):
    """A validation-provider request failed with credentials redacted."""


def redact_url(url):
    if not url:
        return url
    parts = urlsplit(url)
    query = [
        (key, "REDACTED" if key.lower() in SENSITIVE_QUERY_KEYS else value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
    ]
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def request_json(base_url, params=None, timeout=30, max_retries=3, retry_delay=1.0):
    url = f"{base_url}?{urlencode(params)}" if params else base_url
    safe_url = redact_url(url)
    request = Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response), redact_url(response.geturl())
        except HTTPError as exc:
            if exc.code < 500:
                raise ProviderRequestError(
                    f"HTTP {exc.code} from {safe_url}"
                ) from None
            last_error = exc
        except (URLError, OSError) as exc:
            last_error = exc
        if attempt < max_retries:
            time.sleep(retry_delay * (2**attempt))
    raise ProviderRequestError(
        f"Provider request to {safe_url} failed after {max_retries + 1} "
        f"attempt(s): {type(last_error).__name__}"
    ) from None


def discharge_to_m3_per_min(value, unit):
    normalized = unit.lower().replace(" ", "").replace("^", "")
    if "?" in normalized:
        warnings.warn(
            f"USGS unit string {unit!r} contains '?'; treating it as a "
            "mangled superscript 3.",
            stacklevel=2,
        )
        normalized = normalized.replace("?", "3")
    if normalized in {"ft3/s", "cfs", "cubicfeetpersecond"}:
        return float(value) * CFS_TO_M3_PER_MIN
    if normalized in {"m3/s", "cubicmeterspersecond"}:
        return float(value) * 60.0
    if normalized in {"m3/min", "cubicmetersperminute"}:
        return float(value)
    raise ValueError(f"Unsupported USGS discharge unit: {unit}")


def _parse_flow_features(payload):
    observations = []
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        if properties.get("parameter_code") != DISCHARGE_PARAMETER:
            continue
        value = properties.get("value")
        if value is None:
            continue
        unit = properties.get("unit_of_measure", "ft3/s")
        observations.append(
            {
                "observed_at": properties["time"],
                "value": float(value),
                "unit": unit,
                "discharge_m3_per_min": discharge_to_m3_per_min(value, unit),
                "approval_status": properties.get("approval_status"),
            }
        )
    return observations


def fetch_usgs_flow(site_id, start, end, *, api_key=None, requester=request_json):
    gauge_id = site_id if "-" in site_id else f"USGS-{site_id}"
    params = {
        "f": "json",
        "monitoring_location_id": gauge_id,
        "parameter_code": DISCHARGE_PARAMETER,
        "datetime": f"{start}/{end}",
        "limit": 50000,
    }
    if api_key:
        params["api_key"] = api_key
    payload, url = requester(CONTINUOUS_URL, params)
    observations = _parse_flow_features(payload)
    next_url = next(
        (
            link.get("href")
            for link in payload.get("links", [])
            if link.get("rel") == "next"
        ),
        None,
    )
    while next_url:
        payload, _ = requester(next_url)
        observations.extend(_parse_flow_features(payload))
        next_url = next(
            (
                link.get("href")
                for link in payload.get("links", [])
                if link.get("rel") == "next"
            ),
            None,
        )
    if not observations:
        raise ValueError(f"No discharge observations returned for {gauge_id}")
    return observations, redact_url(url), gauge_id
