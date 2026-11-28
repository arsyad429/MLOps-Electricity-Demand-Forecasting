import logging
import signal
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from src.pipelines import inference_worker as worker
from src.storage.prediction_store import PredictionStore


def utc(hour, minute, second=0):
    return datetime(2026, 11, 28, hour, minute, second, tzinfo=timezone.utc)


class SimulatedTime:
    """An advancing clock and stop event for deterministic scheduling tests."""

    def __init__(self, now):
        self.now = now
        self.stopped = False
        self.waits = []

    def clock(self):
        return self.now

    def is_set(self):
        return self.stopped

    def set(self):
        self.stopped = True

    def wait(self, seconds):
        self.waits.append(seconds)
        self.now += timedelta(seconds=seconds)


@pytest.mark.parametrize(("now", "minute", "expected"), [
    (utc(3, 5), 15, utc(3, 15)),
    (utc(3, 15), 15, utc(4, 15)),
    (utc(3, 59, 59), 0, utc(4, 0)),
    (utc(23, 56), 15, utc(0, 15) + timedelta(days=1)),
    (utc(3, 5).astimezone(timezone(timedelta(hours=7))), 15, utc(3, 15)),
])
def test_next_run_aligns_to_the_requested_utc_minute(now, minute, expected):
    assert worker._next_run(now, minute) == expected


def test_cycles_keep_the_hourly_slot_even_when_inference_takes_time():
    simulated = SimulatedTime(utc(3, 14, 58))
    calls = []

    def cycle():
        calls.append(simulated.clock())
        simulated.now += timedelta(minutes=9)
        if len(calls) == 2:
            simulated.set()
        return 0

    worker.run_hourly(15, simulated, clock=simulated.clock, run_cycle=cycle)

    assert calls == [utc(3, 15), utc(4, 15)]
    assert simulated.waits[0] == 2
    assert max(simulated.waits) <= 60


def test_long_inference_skips_missed_slots_without_replay():
    simulated = SimulatedTime(utc(3, 14, 58))
    calls = []

    def cycle():
        calls.append(simulated.clock())
        if len(calls) == 1:
            simulated.now = utc(5, 25)
        else:
            simulated.set()
        return 0

    worker.run_hourly(15, simulated, clock=simulated.clock, run_cycle=cycle)

    assert calls == [utc(3, 15), utc(6, 15)]


def test_resume_after_clock_jump_runs_only_the_latest_due_slot(monkeypatch, caplog):
    simulated = SimulatedTime(utc(3, 14, 58))
    advance = simulated.wait
    calls = []

    def jump_then_wait(_):
        simulated.now = utc(5, 22)
        simulated.wait = advance

    def cycle():
        calls.append(simulated.clock())
        if len(calls) == 2:
            simulated.set()
        return 0

    monkeypatch.setattr(simulated, "wait", jump_then_wait)
    with caplog.at_level(logging.INFO):
        worker.run_hourly(15, simulated, clock=simulated.clock, run_cycle=cycle)

    assert calls == [utc(5, 22), utc(6, 15)]
    assert "Starting inference cycle for 2026-11-28T05:15:00+00:00" in caplog.text


def test_backward_clock_change_does_not_repeat_the_last_slot():
    simulated = SimulatedTime(utc(3, 14, 58))
    calls = []

    def cycle():
        calls.append(simulated.clock())
        if len(calls) == 1:
            simulated.now = utc(3, 5)
        else:
            simulated.set()
        return 0

    worker.run_hourly(15, simulated, clock=simulated.clock, run_cycle=cycle)

    assert calls == [utc(3, 15), utc(4, 15)]


def test_worker_runs_next_hour_after_skip_or_exception(caplog):
    simulated = SimulatedTime(utc(3, 14, 58))
    calls = []

    def cycle():
        calls.append(simulated.clock())
        if len(calls) == 1:
            return 1
        if len(calls) == 2:
            raise RuntimeError("https://example.invalid/?token=fake-secret")
        simulated.set()
        return 0

    worker.run_hourly(15, simulated, clock=simulated.clock, run_cycle=cycle)

    assert calls == [utc(3, 15), utc(4, 15), utc(5, 15)]
    assert "exited with status 1" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "fake-secret" not in caplog.text


def test_idle_shutdown_does_not_start_an_inference_cycle(monkeypatch):
    simulated = SimulatedTime(utc(3, 5))
    monkeypatch.setattr(simulated, "wait", lambda _: simulated.set())
    cycle = Mock()

    worker.run_hourly(15, simulated, clock=simulated.clock, run_cycle=cycle)

    cycle.assert_not_called()


@pytest.mark.parametrize("value", ["-1", "60", "bad-minute"])
def test_cli_rejects_invalid_minute_before_running(monkeypatch, capsys, value):
    monkeypatch.setenv("INFERENCE_RUN_MINUTE_UTC", value)
    simulate = Mock()
    monkeypatch.setattr(worker, "run_hourly", simulate)

    with pytest.raises(SystemExit) as error:
        worker.main([])

    assert error.value.code == 2
    assert "INFERENCE_RUN_MINUTE_UTC must be an integer from 0 to 59" in capsys.readouterr().err
    simulate.assert_not_called()


@pytest.mark.parametrize(("value", "expected"), [(None, 15), ("42", 42)])
def test_worker_cli_uses_the_configured_minute_and_installs_signal_handlers(monkeypatch, tmp_path, value, expected):
    monkeypatch.setattr(worker, "load_dotenv", Mock())
    monkeypatch.delenv("INFERENCE_RUN_MINUTE_UTC", raising=False)
    path = tmp_path / "predictions.sqlite3"
    monkeypatch.setenv("PREDICTION_DB_PATH", str(path))
    if value is not None:
        monkeypatch.setenv("INFERENCE_RUN_MINUTE_UTC", value)
    handlers = {}
    monkeypatch.setattr(worker.signal, "signal", lambda key, handler: handlers.update({key: handler}))

    def inspect_and_stop(minute, stop_event):
        assert minute == expected
        assert set(handlers) == {signal.SIGINT, signal.SIGTERM}
        assert not stop_event.is_set()
        store = PredictionStore(path)
        execution = store.start_run()
        store.complete_run(execution, status="no_observations")
        assert path.with_suffix(".sqlite3-wal").is_file()
        assert path.with_suffix(".sqlite3-shm").is_file()
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        assert stop_event.is_set()

    monkeypatch.setattr(worker, "run_hourly", inspect_and_stop)
    assert worker.main([]) == 0


def test_once_reuses_the_existing_cli_and_preserves_its_exit_status(monkeypatch):
    cycle = Mock(return_value=1)
    monkeypatch.setattr(worker, "run_inference_once", cycle)
    monkeypatch.setenv("INFERENCE_RUN_MINUTE_UTC", "unused")

    assert worker.main(["--once"]) == 1
    cycle.assert_called_once_with()
