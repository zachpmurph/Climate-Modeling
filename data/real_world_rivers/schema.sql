PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS data_sources (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    url TEXT,
    citation TEXT,
    accessed_at TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS rivers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT,
    country TEXT,
    notes TEXT,
    UNIQUE (name, region, country)
);

CREATE TABLE IF NOT EXISTS reaches (
    id INTEGER PRIMARY KEY,
    river_id INTEGER NOT NULL REFERENCES rivers(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    start_lat REAL NOT NULL,
    start_lon REAL NOT NULL,
    end_lat REAL NOT NULL,
    end_lon REAL NOT NULL,
    length_m REAL,
    notes TEXT,
    UNIQUE (river_id, name)
);

CREATE TABLE IF NOT EXISTS reach_markers (
    id INTEGER PRIMARY KEY,
    reach_id INTEGER NOT NULL REFERENCES reaches(id) ON DELETE CASCADE,
    marker_order INTEGER NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    station_m REAL,
    label TEXT,
    source_id INTEGER REFERENCES data_sources(id),
    notes TEXT,
    UNIQUE (reach_id, marker_order)
);

CREATE TABLE IF NOT EXISTS elevation_samples (
    id INTEGER PRIMARY KEY,
    reach_id INTEGER NOT NULL REFERENCES reaches(id) ON DELETE CASCADE,
    marker_id INTEGER REFERENCES reach_markers(id) ON DELETE CASCADE,
    station_m REAL NOT NULL,
    elevation_m REAL NOT NULL,
    method TEXT,
    source_id INTEGER REFERENCES data_sources(id),
    notes TEXT,
    UNIQUE (reach_id, station_m, source_id)
);

CREATE TABLE IF NOT EXISTS slope_samples (
    id INTEGER PRIMARY KEY,
    reach_id INTEGER NOT NULL REFERENCES reaches(id) ON DELETE CASCADE,
    start_station_m REAL,
    end_station_m REAL,
    slope REAL NOT NULL,
    elevation_start_m REAL,
    elevation_end_m REAL,
    method TEXT,
    source_id INTEGER REFERENCES data_sources(id),
    notes TEXT
);

CREATE TABLE IF NOT EXISTS roughness_samples (
    id INTEGER PRIMARY KEY,
    reach_id INTEGER NOT NULL REFERENCES reaches(id) ON DELETE CASCADE,
    start_station_m REAL,
    end_station_m REAL,
    manning_n REAL NOT NULL,
    method TEXT,
    source_id INTEGER REFERENCES data_sources(id),
    notes TEXT
);

CREATE TABLE IF NOT EXISTS channel_geometry_samples (
    id INTEGER PRIMARY KEY,
    reach_id INTEGER NOT NULL REFERENCES reaches(id) ON DELETE CASCADE,
    marker_id INTEGER REFERENCES reach_markers(id) ON DELETE SET NULL,
    station_m REAL,
    width_m REAL,
    bankfull_depth_m REAL,
    method TEXT,
    source_id INTEGER REFERENCES data_sources(id),
    notes TEXT,
    CHECK (width_m IS NULL OR width_m > 0),
    CHECK (bankfull_depth_m IS NULL OR bankfull_depth_m > 0)
);

CREATE TABLE IF NOT EXISTS flow_observations (
    id INTEGER PRIMARY KEY,
    reach_id INTEGER REFERENCES reaches(id) ON DELETE CASCADE,
    marker_id INTEGER REFERENCES reach_markers(id) ON DELETE SET NULL,
    gauge_id TEXT,
    observed_at TEXT,
    discharge_m3_per_min REAL,
    discharge_original_value REAL,
    discharge_original_unit TEXT,
    stage_m REAL,
    source_id INTEGER REFERENCES data_sources(id),
    notes TEXT
);

CREATE TABLE IF NOT EXISTS rainfall_observations (
    id INTEGER PRIMARY KEY,
    reach_id INTEGER NOT NULL REFERENCES reaches(id) ON DELETE CASCADE,
    marker_id INTEGER REFERENCES reach_markers(id) ON DELETE SET NULL,
    observed_at TEXT NOT NULL,
    precipitation_mm REAL NOT NULL,
    interval_min REAL NOT NULL,
    source_id INTEGER REFERENCES data_sources(id),
    notes TEXT,
    CHECK (precipitation_mm >= 0),
    CHECK (interval_min > 0),
    UNIQUE (reach_id, marker_id, observed_at, source_id)
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY,
    reach_id INTEGER NOT NULL REFERENCES reaches(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    run_name TEXT NOT NULL,
    run_at TEXT,
    code_version TEXT,
    conclusion_path TEXT,
    notes TEXT,
    UNIQUE (reach_id, model_name, run_name)
);

CREATE INDEX IF NOT EXISTS idx_reach_markers_reach_order
    ON reach_markers(reach_id, marker_order);

CREATE INDEX IF NOT EXISTS idx_slope_samples_reach
    ON slope_samples(reach_id, start_station_m, end_station_m);

CREATE INDEX IF NOT EXISTS idx_elevation_samples_reach
    ON elevation_samples(reach_id, station_m);

CREATE INDEX IF NOT EXISTS idx_roughness_samples_reach
    ON roughness_samples(reach_id, start_station_m, end_station_m);

CREATE INDEX IF NOT EXISTS idx_flow_observations_reach_time
    ON flow_observations(reach_id, observed_at);

CREATE INDEX IF NOT EXISTS idx_rainfall_observations_reach_time
    ON rainfall_observations(reach_id, observed_at);

CREATE INDEX IF NOT EXISTS idx_channel_geometry_reach_station
    ON channel_geometry_samples(reach_id, station_m);
