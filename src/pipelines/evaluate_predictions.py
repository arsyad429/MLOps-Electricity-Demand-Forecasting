"""Evaluate stored t+2h CISO predictions and report metrics by model version."""

import argparse
import json
import sqlite3

from src.storage.prediction_store import PredictionStore, StorageError, evaluation_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", help="Override PREDICTION_DB_PATH for this invocation.")
    parser.add_argument("--window-hours", type=int, help="Limit the report to recent UTC target hours.")
    parser.add_argument("--summary-only", action="store_true", help="Open the database read-only.")
    args = parser.parse_args(argv)
    if args.window_hours is not None and args.window_hours < 1:
        parser.error("--window-hours must be positive.")

    try:
        written = None if args.summary_only else PredictionStore(args.db).evaluate_predictions()
        summary = evaluation_summary(path=args.db, window_hours=args.window_hours)
    except (OSError, sqlite3.Error, StorageError, ValueError, OverflowError):
        parser.exit(1, "Could not evaluate predictions. Check the database path, schema, and report window.\n")
    print(json.dumps({"evaluations_updated": written, **summary}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
