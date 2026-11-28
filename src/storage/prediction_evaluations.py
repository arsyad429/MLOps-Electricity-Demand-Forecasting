"""SQLite statements for matching stored forecasts with reported actuals."""

SCHEMA_V2 = (
    """
    CREATE TABLE prediction_evaluations (
        prediction_id INTEGER PRIMARY KEY REFERENCES predictions(prediction_id),
        actual_demand REAL NOT NULL CHECK (actual_demand >= 0),
        actual_data_source TEXT NOT NULL CHECK (actual_data_source IN ('api', 'dashboard')),
        actual_revision_count INTEGER NOT NULL CHECK (actual_revision_count >= 0),
        evaluated_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX prediction_target ON predictions (forecast_target_timestamp)",
)

EVALUATE_READY = """
INSERT INTO prediction_evaluations (
    prediction_id, actual_demand, actual_data_source, actual_revision_count, evaluated_at
)
SELECT p.prediction_id, o.actual_demand, o.data_source, o.revision_count, :as_of
FROM predictions AS p
JOIN observations AS o ON o.observation_timestamp = p.forecast_target_timestamp
WHERE p.forecast_target_timestamp <= :as_of
  AND p.created_at <= :as_of
  AND o.updated_at <= :as_of
ON CONFLICT (prediction_id) DO UPDATE SET
    actual_demand = excluded.actual_demand,
    actual_data_source = excluded.actual_data_source,
    actual_revision_count = excluded.actual_revision_count,
    evaluated_at = excluded.evaluated_at
WHERE excluded.actual_revision_count > prediction_evaluations.actual_revision_count
"""

RECENT_EVALUATIONS = """
SELECT e.*, p.forecast_target_timestamp, p.predicted_demand, p.model_name, p.model_version,
       p.predicted_demand - e.actual_demand AS error_mwh,
       abs(p.predicted_demand - e.actual_demand) AS absolute_error_mwh
FROM prediction_evaluations AS e
JOIN predictions AS p ON p.prediction_id = e.prediction_id
ORDER BY p.forecast_target_timestamp DESC, e.prediction_id DESC
LIMIT ?
"""

EVALUATION_SUMMARY = """
WITH forecasts AS (
    SELECT p.model_name, p.model_version,
           e.actual_demand, p.predicted_demand - e.actual_demand AS error,
           CASE
               WHEN p.forecast_target_timestamp > :as_of THEN 'future'
               WHEN o.observation_timestamp IS NULL OR o.updated_at > :as_of THEN 'pending_actual'
               WHEN e.prediction_id IS NULL OR e.actual_revision_count != o.revision_count
                    OR e.evaluated_at > :as_of THEN 'ready_to_evaluate'
               ELSE 'evaluated'
           END AS evaluation_status
    FROM predictions AS p
    LEFT JOIN observations AS o ON o.observation_timestamp = p.forecast_target_timestamp
    LEFT JOIN prediction_evaluations AS e ON e.prediction_id = p.prediction_id
    WHERE p.created_at <= :as_of
      AND (:since IS NULL OR p.forecast_target_timestamp >= :since)
)
SELECT model_name, model_version,
       count(*) AS predictions,
       sum(evaluation_status = 'future') AS future,
       sum(evaluation_status = 'pending_actual') AS pending_actual,
       sum(evaluation_status = 'ready_to_evaluate') AS ready_to_evaluate,
       sum(evaluation_status = 'evaluated') AS evaluated,
       avg(CASE WHEN evaluation_status = 'evaluated' THEN abs(error) END) AS mae_mwh,
       avg(CASE WHEN evaluation_status = 'evaluated' THEN error * error END) AS mse,
       100.0 * sum(CASE WHEN evaluation_status = 'evaluated' THEN abs(error) END)
           / nullif(sum(CASE WHEN evaluation_status = 'evaluated' THEN actual_demand END), 0) AS wape_pct
FROM forecasts
GROUP BY model_name, model_version
ORDER BY model_name, model_version
"""
