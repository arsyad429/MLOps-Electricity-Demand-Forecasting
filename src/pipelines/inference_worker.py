"""Schedule the existing persisted inference cycle at a fixed UTC minute."""

import argparse
import logging
import os
import signal
from datetime import datetime, timedelta, timezone
from threading import Event

from dotenv import load_dotenv

from src.data.ingest_data import PROJECT_ROOT
from src.pipelines.run_online_inference import main as run_inference_once
from src.storage.prediction_store import PredictionStore


LOGGER = logging.getLogger(__name__)
HOUR = timedelta(hours=1)
DEFAULT_MINUTE = 15


def _now_utc():
    return datetime.now(timezone.utc)


def _next_run(now, minute):
    now = now.astimezone(timezone.utc)
    slot = now.replace(minute=minute, second=0, microsecond=0)
    return slot if slot > now else slot + HOUR


def _wait_until(target, stop_event, clock):
    while not stop_event.is_set():
        now = clock()
        remaining = (target - now).total_seconds()
        if remaining <= 0:
            return now
        # Recheck the UTC clock to account for adjustments while the worker is idle.
        stop_event.wait(min(remaining, 60))
    return None


def run_hourly(minute, stop_event, *, clock=_now_utc, run_cycle=run_inference_once):
    """Run cycles serially, skipping older slots when the worker is delayed."""
    last_slot = None
    while not stop_event.is_set():
        now = clock()
        anchor = now if last_slot is None else max(now, last_slot)
        scheduled = _next_run(anchor, minute)
        LOGGER.info("Next inference cycle at %s", scheduled.isoformat())

        now = _wait_until(scheduled, stop_event, clock)
        if now is None or stop_event.is_set():
            return
        last_slot = scheduled + ((now - scheduled) // HOUR) * HOUR
        LOGGER.info("Starting inference cycle for %s", last_slot.isoformat())
        try:
            exit_code = run_cycle()
        except Exception as error:
            LOGGER.error("Inference cycle raised %s. The hourly worker will continue.", type(error).__name__)
        else:
            if exit_code:
                LOGGER.warning("Inference cycle exited with status %d.", exit_code)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one inference cycle immediately.")
    args = parser.parse_args(argv)
    if args.once:
        return run_inference_once()

    load_dotenv(PROJECT_ROOT / ".env")
    try:
        minute = int(os.getenv("INFERENCE_RUN_MINUTE_UTC", DEFAULT_MINUTE))
    except ValueError:
        parser.error("INFERENCE_RUN_MINUTE_UTC must be an integer from 0 to 59.")
    if not 0 <= minute <= 59:
        parser.error("INFERENCE_RUN_MINUTE_UTC must be an integer from 0 to 59.")

    stop_event = Event()

    def request_stop(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with PredictionStore().keep_wal_available():
        run_hourly(minute, stop_event)
    LOGGER.info("Hourly inference worker stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
