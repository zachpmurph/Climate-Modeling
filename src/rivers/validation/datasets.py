"""Validation-only hydraulic dataset hierarchy and policy checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HydraulicDataset:
    dataset_id: str
    rank: int
    label: str
    resolution: str
    role: str
    limitations: str


# Lower rank is preferred.  The last two proxy entries are retained only so the
# current evidence can be described honestly; they are below the target hierarchy.
DATASETS = {
    item.dataset_id: item
    for item in (
        HydraulicDataset(
            "surveyed_topobathymetry",
            1,
            "Surveyed cross-sections or topobathymetric lidar",
            "survey scale",
            "Preferred channel, bank, floodplain, and submerged-bed geometry",
            "Coverage is sparse and survey dates must bracket the event.",
        ),
        HydraulicDataset(
            "usgs_3dep_1m",
            2,
            "USGS 3DEP 1 m lidar DEM",
            "1 m",
            "Preferred U.S. bare-earth floodplain terrain when bathymetry is absent",
            "Does not normally resolve submerged channel bathymetry.",
        ),
        HydraulicDataset(
            "noaa_topobathy",
            2,
            "NOAA/USGS integrated topobathymetric DEM",
            "project specific",
            "Preferred public terrain where an integrated product exists",
            "Geographically limited; vertical datum and survey epoch must be checked.",
        ),
        HydraulicDataset(
            "anadem_30m",
            3,
            "ANADEM regional bare-earth DEM",
            "30 m",
            "South America fallback when higher-resolution reviewed terrain is absent",
            "Too coarse for many channels, levees, culverts, and local bank controls.",
        ),
        HydraulicDataset(
            "gedtm30",
            4,
            "GEDTM30 global bare-earth DEM",
            "30 m",
            "Global screening fallback",
            "Screening-grade only; requires conditioning and independent verification.",
        ),
        HydraulicDataset(
            "field_measurement_cross_sections",
            1,
            "USGS field-measurement cross-sections",
            "two section snapshots",
            "Observed channel geometry at gauge sections",
            "Sparse sections do not map the intervening floodplain or channel bed.",
        ),
        HydraulicDataset(
            "gage_datum_reach_proxy",
            5,
            "Gauge-datum reach-average proxy",
            "reach average",
            "Legacy slope proxy below the preferred hierarchy",
            "Gauge datum is not bed elevation and cannot describe local controls.",
        ),
        HydraulicDataset(
            "assumed_reach_geometry",
            5,
            "Assumed constant reach geometry",
            "reach average",
            "Legacy geometry proxy below the preferred hierarchy",
            "Cannot distinguish terrain error from channel-shape assumptions.",
        ),
    )
}

TARGET_HIERARCHY = (
    ("surveyed_topobathymetry",),
    ("usgs_3dep_1m", "noaa_topobathy"),
    ("anadem_30m",),
    ("gedtm30",),
)


def dataset_record(dataset_id):
    try:
        return asdict(DATASETS[dataset_id])
    except KeyError as exc:
        raise ValueError(f"Unknown hydraulic validation dataset: {dataset_id}") from exc


def validate_case_policy(config):
    """Reject calibration and require explicit hydraulic-data provenance."""
    policy = config.get("validation_policy", {})
    if policy.get("calibration") != "none":
        raise ValueError("Validation cases must set validation_policy.calibration='none'")
    forbidden = {
        "calibration",
        "calibrated_parameters",
        "manning_scale",
        "width_scale",
        "slope_scale",
        "reach_gain",
    }
    present = forbidden.intersection(config)
    if present:
        raise ValueError(
            "Calibration fields are forbidden in validation cases: "
            + ", ".join(sorted(present))
        )
    dataset_id = config.get("hydraulic_dataset")
    if dataset_id is None:
        raise ValueError("Validation case must declare hydraulic_dataset")
    settings = config.get("validation_2d")
    if not isinstance(settings, dict):
        raise ValueError(
            "Validation case must declare validation_2d; 1-D fallback is forbidden"
        )
    if settings.get("representation") not in {"ribbon", "shelf"}:
        raise ValueError("validation_2d.representation must be ribbon or shelf")
    if int(settings.get("x_cells", 0)) < 2 or int(settings.get("y_cells", 0)) < 1:
        raise ValueError("validation_2d requires x_cells >= 2 and y_cells >= 1")
    return dataset_record(dataset_id)


def hierarchy_evidence():
    return [
        [dataset_record(dataset_id) for dataset_id in level]
        for level in TARGET_HIERARCHY
    ]
