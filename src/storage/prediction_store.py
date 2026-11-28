"""SQLite storage for reported CISO observations and live t+2h predictions."""

import argparse
import json
import math
import os
import re
import sqlite3
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from numbers import Real
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = Path("runtime/predictions.sqlite3")
SCHEMA_VERSION = 1
BUSY_TIMEOUT_SECONDS = 5
SOURCES = {"api", "dashboard"}
MODEL_URI_PATTERN = re.compile(r"models:/([A-Za-z0-9_.-]+)(?:@[A-Za-z0-9_.-]+|/[1-9][0-9]*)?")
MODEL_VERSION_PATTERN = re.compile(r"[1-9][0-9]*")
ERROR_MESSAGES = {
    "ingestion_unavailable": "The EIA data source or its network was unavailable.",
    "ingestion_error": "EIA ingestion could not complete.",
    "invalid_observation": "The latest observation or forecast timestamp was invalid.",
    "invalid_features": "The model input did not pass validation.",
    "configuration_error": "The inference API configuration was invalid.",
    "api_timeout": "The prediction API timed out.",
    "api_connection_error": "The prediction API connection failed.",
    "api_request_error": "The prediction API request failed.",
    "api_http_error": "The prediction API returned an HTTP error.",
    "invalid_response": "The prediction API response did not pass validation.",
    "storage_error": "The SQLite write could not complete.",
    "internal_error": "The online inference run failed unexpectedly.",
}
SKIPPED_STATUSES = {"no_observations", "incomplete_context", "stale"}
TERMINAL_STATUSES = SKIPPED_STATUSES | ERROR_MESSAGES.keys() | {"success"}

SCHEMA_V1 = (
    """
    CREATE TABLE observations (
        observation_timestamp TEXT PRIMARY KEY NOT NULL,
        actual_demand REAL NOT NULL CHECK (actual_demand >= 0),
        data_source TEXT NOT NULL CHECK (data_source IN ('api', 'dashboard')),
        ingested_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL,
        revision_count INTEGER NOT NULL DEFAULT 0 CHECK (revision_count >= 0)
    )
    """,
    """
    CREATE TABLE observation_revisions (
        observation_timestamp TEXT NOT NULL REFERENCES observations(observation_timestamp),
        revision_number INTEGER NOT NULL CHECK (revision_number > 0),
        previous_actual_demand REAL NOT NULL,
        previous_data_source TEXT NOT NULL,
        revised_actual_demand REAL NOT NULL,
        revised_data_source TEXT NOT NULL,
        revised_at TEXT NOT NULL,
        PRIMARY KEY (observation_timestamp, revision_number)
    )
    """,
    """
    CREATE TRIGGER save_observation_revision
    AFTER UPDATE OF actual_demand, data_source ON observations
    WHEN OLD.actual_demand != NEW.actual_demand OR OLD.data_source != NEW.data_source
    BEGIN
        INSERT INTO observation_revisions VALUES (
            OLD.observation_timestamp, NEW.revision_count,
            OLD.actual_demand, OLD.data_source,
            NEW.actual_demand, NEW.data_source, NEW.updated_at
        );
    END
    """,
    """
    CREATE TABLE predictions (
        prediction_id INTEGER PRIMARY KEY,
        observation_timestamp TEXT NOT NULL REFERENCES observations(observation_timestamp),
        forecast_target_timestamp TEXT NOT NULL,
        prediction_created_at TEXT NOT NULL,
        predicted_demand REAL NOT NULL,
        actual_input_demand REAL NOT NULL CHECK (actual_input_demand >= 0),
        data_source TEXT NOT NULL CHECK (data_source IN ('api', 'dashboard')),
        forecast_horizon_hours INTEGER NOT NULL CHECK (forecast_horizon_hours = 2),
        model_uri TEXT,
        model_name TEXT,
        model_version TEXT,
        inference_latency_ms REAL NOT NULL CHECK (inference_latency_ms >= 0),
        created_at TEXT NOT NULL,
        CHECK (forecast_target_timestamp =
            strftime('%Y-%m-%dT%H:%M:%S', observation_timestamp, '+2 hours') || '.000000Z'),
        CHECK (observation_timestamp <= prediction_created_at
               AND prediction_created_at < forecast_target_timestamp
               AND prediction_created_at <= created_at)
    )
    """,
    """
    CREATE UNIQUE INDEX prediction_identity ON predictions (
        observation_timestamp, coalesce(model_name, ''), coalesce(model_version, '')
    )
    """,
    """
    CREATE TABLE pipeline_runs (
        execution_id INTEGER PRIMARY KEY,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        status TEXT NOT NULL,
        data_source TEXT CHECK (data_source IN ('api', 'dashboard')),
        latest_observation TEXT,
        error_category TEXT,
        error_message TEXT,
        prediction_id INTEGER REFERENCES predictions(prediction_id),
        CHECK (completed_at IS NULL OR completed_at >= started_at),
        CHECK ((status = 'running') = (completed_at IS NULL)),
        CHECK ((status IN ('success', 'duplicate')) = (prediction_id IS NOT NULL))
    )
    """,
)
MIGRATIONS = {1: SCHEMA_V1}

UPSERT_OBSERVATION = """
INSERT INTO observations (
    observation_timestamp, actual_demand, data_source, ingested_at, updated_at, last_seen_at
) VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(observation_timestamp) DO UPDATE SET
    actual_demand = excluded.actual_demand,
    data_source = excluded.data_source,
    last_seen_at = excluded.last_seen_at,
    updated_at = CASE
        WHEN observations.actual_demand != excluded.actual_demand
          OR observations.data_source != excluded.data_source THEN excluded.updated_at
        ELSE observations.updated_at
    END,
    revision_count = observations.revision_count + (
        observations.actual_demand != excluded.actual_demand
        OR observations.data_source != excluded.data_source
    )
WHERE excluded.last_seen_at >= observations.last_seen_at
"""

READ_QUERIES = {
    "observations": "SELECT * FROM observations ORDER BY observation_timestamp DESC LIMIT ?",
    "predictions": """
        SELECT * FROM predictions ORDER BY prediction_created_at DESC, prediction_id DESC LIMIT ?
    """,
    "runs": "SELECT * FROM pipeline_runs ORDER BY execution_id DESC LIMIT ?",
    "revisions": """
        SELECT * FROM observation_revisions
        ORDER BY revised_at DESC, observation_timestamp DESC, revision_number DESC LIMIT ?
    """,
}


@dataclass(frozen=True)
class Prediction:
    observation_timestamp: datetime
    forecast_target_timestamp: datetime
    prediction_created_at: datetime
    predicted_demand: float
    actual_input_demand: float
    data_source: str
    forecast_horizon_hours: int
    model_uri: str | None
    model_version: str | None
    inference_latency_ms: float


@dataclass(frozen=True)
class CompletedRun:
    status: str
    prediction_id: int | None
    existing_prediction: dict | None = None


class StorageError(RuntimeError):
    """An incompatible database or an invalid run state."""


def prediction_db_path(path=None):
    if path is None:
        load_dotenv(PROJECT_ROOT / ".env")
        path = os.getenv("PREDICTION_DB_PATH") or DEFAULT_DB_PATH
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def _utc(value):
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            raise ValueError("Expected a timezone-aware ISO 8601 timestamp.") from None
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("Expected a timezone-aware timestamp.")
    return value.astimezone(timezone.utc)


def _iso(value):
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _timestamp(value=None):
    return _iso(datetime.now(timezone.utc) if value is None else _utc(value))


def _observation_hour(value):
    hour = _utc(value)
    if hour.minute or hour.second or hour.microsecond or getattr(hour, "nanosecond", 0):
        raise ValueError("Observation and target timestamps must be UTC hour boundaries.")
    return hour


def _finite(value, *, nonnegative=False):
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError("Demand and latency values must be finite numbers.")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError("Demand and latency values must be finite numbers.") from None
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError("Demand and latency values must be finite and within their allowed range.")
    return number


def _source(source):
    if not isinstance(source, str) or source not in SOURCES:
        raise ValueError("Observation source must identify an EIA data route.")
    return source


def _enable_wal(connection):
    # Changing the journal mode can briefly contend with another process opening a new file.
    deadline = time.monotonic() + BUSY_TIMEOUT_SECONDS
    while True:
        try:
            mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if mode != "wal":
                raise StorageError("SQLite WAL mode is unavailable for the prediction database.")
            return
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED} or time.monotonic() >= deadline:
                raise
            time.sleep(0.025)


def _connect(path, *, read_only=False):
    location = path.resolve().as_uri() + "?mode=ro" if read_only else path
    connection = sqlite3.connect(
        location, uri=read_only, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None
    )
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
    except Exception:
        connection.close()
        raise
    return connection


class PredictionStore:
    def __init__(self, path=None):
        self.path = prediction_db_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect(self.path)) as connection:
            if connection.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                _enable_wal(connection)
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version > SCHEMA_VERSION:
                    raise StorageError("The prediction database uses a newer schema version.")
                for migration in range(version + 1, SCHEMA_VERSION + 1):
                    for statement in MIGRATIONS[migration]:
                        connection.execute(statement)
                    connection.execute(f"PRAGMA user_version = {migration}")

    @contextmanager
    def _write(self):
        with closing(_connect(self.path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            yield connection

    def upsert_observations(self, observations, *, source, ingested_at=None):
        """Store a batch of validated reported (timestamp, demand) rows atomically."""
        source = _source(source)
        seen = _timestamp(ingested_at)

        def parameters():
            for timestamp, demand in observations:
                period = _iso(_observation_hour(timestamp))
                if period > seen:
                    raise ValueError("The observation timestamp is after its ingestion time.")
                yield period, _finite(demand, nonnegative=True), source, seen, seen, seen

        with self._write() as connection:
            connection.executemany(UPSERT_OBSERVATION, parameters())

    def start_run(self, *, started_at=None):
        started = _timestamp(started_at)
        with self._write() as connection:
            cursor = connection.execute(
                "INSERT INTO pipeline_runs (started_at, status) VALUES (?, 'running')", (started,)
            )
            return cursor.lastrowid

    def complete_run(
        self, execution_id, *, status, source=None, latest_observation=None, prediction=None,
        completed_at=None, http_status=None,
    ):
        """Commit a terminal status and its optional prediction in one transaction."""
        if status not in TERMINAL_STATUSES:
            raise ValueError("Invalid terminal pipeline status.")
        if (status == "success") != (prediction is not None):
            raise ValueError("A successful run requires one validated prediction.")
        if source is not None:
            _source(source)
        latest = None if latest_observation is None else _iso(_observation_hour(latest_observation))
        completed = _timestamp(completed_at)
        error_category = status if status in ERROR_MESSAGES else None
        error_message = ERROR_MESSAGES.get(status)
        if status == "api_http_error" and type(http_status) is int and 100 <= http_status <= 599:
            error_message = f"The prediction API returned HTTP {http_status}."

        with self._write() as connection:
            pending = connection.execute(
                "SELECT status FROM pipeline_runs WHERE execution_id = ?", (execution_id,)
            ).fetchone()
            if pending is None or pending["status"] != "running":
                raise StorageError("The execution must exist and be running before completion.")

            prediction_id = None
            existing = None
            if prediction is not None:
                if source != prediction.data_source or latest != _iso(_observation_hour(prediction.observation_timestamp)):
                    raise ValueError("The prediction must match the run's latest observation and source.")
                prediction_id, existing = _insert_prediction(connection, prediction, completed)
                if existing is not None:
                    status = "duplicate"

            connection.execute(
                """
                UPDATE pipeline_runs
                SET completed_at = ?, status = ?, data_source = ?, latest_observation = ?,
                    error_category = ?, error_message = ?, prediction_id = ?
                WHERE execution_id = ?
                """,
                (completed, status, source, latest, error_category, error_message, prediction_id, execution_id),
            )
        return CompletedRun(status, prediction_id, existing)


def _insert_prediction(connection, prediction, created_at):
    observation = _observation_hour(prediction.observation_timestamp)
    target = _observation_hour(prediction.forecast_target_timestamp)
    predicted_at = _utc(prediction.prediction_created_at)
    if (type(prediction.forecast_horizon_hours) is not int or prediction.forecast_horizon_hours != 2
            or target != observation + timedelta(hours=2)):
        raise ValueError("The forecast target must be two hours after the observation.")
    if not observation <= predicted_at < target:
        raise ValueError("The prediction must be created after the observation and before its target.")

    source = _source(prediction.data_source)
    model_uri = prediction.model_uri
    model_name = None
    if model_uri is not None:
        match = MODEL_URI_PATTERN.fullmatch(model_uri) if isinstance(model_uri, str) else None
        if match is None:
            raise ValueError("The model URI must be a registered MLflow model identifier.")
        model_name = match.group(1)
    version = prediction.model_version
    if version is not None and (not isinstance(version, str) or not MODEL_VERSION_PATTERN.fullmatch(version)):
        raise ValueError("The loaded model version must be a positive version number.")

    params = {
        "observation": _iso(observation),
        "target": _iso(target),
        "prediction_created_at": _iso(predicted_at),
        "predicted_demand": _finite(prediction.predicted_demand),
        "actual_input_demand": _finite(prediction.actual_input_demand, nonnegative=True),
        "source": source,
        "horizon": prediction.forecast_horizon_hours,
        "model_uri": model_uri,
        "model_name": model_name,
        "model_version": version,
        "latency": _finite(prediction.inference_latency_ms, nonnegative=True),
        "created_at": created_at,
    }
    # Unknown identifiers are ambiguous. Preserve the earliest potentially matching prediction.
    existing = connection.execute(
        """
        SELECT * FROM predictions
        WHERE observation_timestamp = :observation
          AND (model_name IS NULL OR :model_name IS NULL OR model_name = :model_name)
          AND (model_version IS NULL OR :model_version IS NULL OR model_version = :model_version)
        ORDER BY prediction_id LIMIT 1
        """,
        params,
    ).fetchone()
    if existing is not None:
        return existing["prediction_id"], dict(existing)

    cursor = connection.execute(
        """
        INSERT INTO predictions (
            observation_timestamp, forecast_target_timestamp, prediction_created_at, predicted_demand,
            actual_input_demand, data_source, forecast_horizon_hours, model_uri, model_name,
            model_version, inference_latency_ms, created_at
        ) VALUES (
            :observation, :target, :prediction_created_at, :predicted_demand, :actual_input_demand,
            :source, :horizon, :model_uri, :model_name, :model_version, :latency, :created_at
        )
        """,
        params,
    )
    return cursor.lastrowid, None


def recent_records(kind, *, limit=10, path=None):
    """Read recent rows without creating or migrating a database."""
    if kind not in READ_QUERIES or type(limit) is not int or limit < 1:
        raise ValueError("Select a supported record type and a positive row limit.")
    with closing(_connect(prediction_db_path(path), read_only=True)) as connection:
        return [dict(row) for row in connection.execute(READ_QUERIES[kind], (limit,))]


def main():
    parser = argparse.ArgumentParser(description="Inspect recent online inference records (read-only).")
    parser.add_argument("kind", choices=READ_QUERIES)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--db", help="Optional database path. Defaults to PREDICTION_DB_PATH.")
    args = parser.parse_args()
    try:
        rows = recent_records(args.kind, limit=args.limit, path=args.db)
    except (OSError, ValueError, sqlite3.Error):
        parser.exit(1, "Unable to read prediction records. Check the database path and row limit.\n")
    print(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
