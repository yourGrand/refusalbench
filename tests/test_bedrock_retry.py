"""tests for the bedrock retry taxonomy, backoff, and concurrency behaviour."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from refusalbench.multilingual import bedrock_retry
from refusalbench.multilingual.bedrock_retry import (
    ERROR_PERMANENT,
    ERROR_RETRYABLE_THROTTLE,
    ERROR_RETRYABLE_TRANSIENT,
    AdaptiveConcurrency,
    PermanentBedrockError,
    RetryBudgetExceeded,
    _backoff_seconds,
    acompletion_with_retry,
    classify_error,
)


class _Headers(dict):
    """minimal httpx-like headers mapping."""


class _Response:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = _Headers(headers or {})


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = type("_Msg", (), {"content": content})()


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class TestClassifyError:
    @pytest.mark.parametrize(
        "message",
        [
            "ThrottlingException: Too many requests",
            "Rate exceeded for this model",
            "429 please wait and retry after 5s",
        ],
    )
    def test_throttles(self, message: str) -> None:
        assert classify_error(Exception(message)) == ERROR_RETRYABLE_THROTTLE

    @pytest.mark.parametrize(
        "message",
        ["Read timed out", "connection reset by peer", "503 service unavailable"],
    )
    def test_transients(self, message: str) -> None:
        assert classify_error(Exception(message)) == ERROR_RETRYABLE_TRANSIENT

    @pytest.mark.parametrize(
        "message",
        [
            "AccessDeniedException: not authorized",
            "ValidationException: malformed input",
            "model not found in this region",
            "on-demand throughput is not supported, use an inference profile",
        ],
    )
    def test_permanents(self, message: str) -> None:
        assert classify_error(Exception(message)) == ERROR_PERMANENT

    def test_status_code_is_used_when_text_says_nothing(self) -> None:
        exc = Exception("something odd happened")
        exc.status_code = 429
        assert classify_error(exc) == ERROR_RETRYABLE_THROTTLE
        exc.status_code = 403
        assert classify_error(exc) == ERROR_PERMANENT

    def test_unknown_errors_are_retried_not_dropped(self) -> None:
        assert classify_error(Exception("???")) == ERROR_RETRYABLE_TRANSIENT

    def test_permanent_wins_over_throttle_wording(self) -> None:
        """
        a validation error that happens to mention throttling must not be
        retried for ever.
        """
        exc = Exception("ValidationException: throttling exception in the text")
        assert classify_error(exc) == ERROR_PERMANENT


class TestBackoffSeconds:
    def test_stays_within_floor_and_ceiling(self) -> None:
        for attempt in range(12):
            sleep = _backoff_seconds(attempt, None)
            assert bedrock_retry._BACKOFF_FLOOR_SEC <= sleep
            assert sleep <= bedrock_retry._BACKOFF_CEILING_SEC

    def test_retry_after_hint_is_honoured(self) -> None:
        assert _backoff_seconds(0, 30.0) == pytest.approx(30.0)

    def test_retry_after_is_clamped(self) -> None:
        assert _backoff_seconds(0, 0.1) == pytest.approx(
            bedrock_retry._BACKOFF_FLOOR_SEC
        )
        assert _backoff_seconds(0, 9999.0) == pytest.approx(
            bedrock_retry._BACKOFF_CEILING_SEC
        )


class TestAdaptiveConcurrency:
    def test_throttle_cluster_lowers_the_limit(self) -> None:
        adaptive = AdaptiveConcurrency(default_limit=8)
        for _ in range(3):
            adaptive.note_throttle()
        assert adaptive.current_limit == 7

    def test_isolated_throttles_do_not_lower_the_limit(self) -> None:
        adaptive = AdaptiveConcurrency(default_limit=8)
        adaptive.note_throttle()
        adaptive.note_throttle()
        assert adaptive.current_limit == 8

    def test_limit_never_drops_below_one(self) -> None:
        adaptive = AdaptiveConcurrency(default_limit=2)
        for _ in range(60):
            adaptive.note_throttle()
        assert adaptive.current_limit == 1

    def test_success_streak_restores_the_limit(self) -> None:
        async def scenario() -> int:
            adaptive = AdaptiveConcurrency(default_limit=8)
            for _ in range(3):
                adaptive.note_throttle()
            for _ in range(10):
                await adaptive.note_success()
            return adaptive.current_limit

        assert asyncio.run(scenario()) == 8

    def test_limit_is_not_raised_above_the_default(self) -> None:
        async def scenario() -> int:
            adaptive = AdaptiveConcurrency(default_limit=2)
            for _ in range(100):
                await adaptive.note_success()
            return adaptive.current_limit

        assert asyncio.run(scenario()) == 2

    def test_acquire_blocks_at_the_limit(self) -> None:
        async def scenario() -> bool:
            adaptive = AdaptiveConcurrency(default_limit=1)
            await adaptive.acquire()
            waiter = asyncio.create_task(adaptive.acquire())
            await asyncio.sleep(0)
            blocked = not waiter.done()
            await adaptive.release()
            await waiter
            return blocked

        assert asyncio.run(scenario()) is True


class TestAcompletionWithRetry:
    def test_returns_content_on_first_success(self, monkeypatch) -> None:
        async def fake_acompletion(**kwargs: Any) -> _FakeCompletion:
            return _FakeCompletion("hello")

        monkeypatch.setattr(bedrock_retry.litellm, "acompletion", fake_acompletion)
        result = asyncio.run(
            acompletion_with_retry(
                model="m", messages=[{"role": "user", "content": "x"}]
            )
        )
        assert result == "hello"

    def test_permanent_error_is_not_retried(self, monkeypatch) -> None:
        calls = {"n": 0}

        async def fake_acompletion(**kwargs: Any) -> _FakeCompletion:
            calls["n"] += 1
            raise Exception("AccessDeniedException: nope")

        monkeypatch.setattr(bedrock_retry.litellm, "acompletion", fake_acompletion)
        with pytest.raises(PermanentBedrockError):
            asyncio.run(
                acompletion_with_retry(
                    model="m", messages=[{"role": "user", "content": "x"}]
                )
            )
        assert calls["n"] == 1

    def test_transient_error_is_retried_then_succeeds(self, monkeypatch) -> None:
        calls = {"n": 0}

        async def fake_acompletion(**kwargs: Any) -> _FakeCompletion:
            calls["n"] += 1
            if calls["n"] < 3:
                raise Exception("Read timed out")
            return _FakeCompletion("recovered")

        async def no_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr(bedrock_retry.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(bedrock_retry.asyncio, "sleep", no_sleep)
        result = asyncio.run(
            acompletion_with_retry(
                model="m", messages=[{"role": "user", "content": "x"}]
            )
        )
        assert result == "recovered"
        assert calls["n"] == 3

    def test_concurrency_slot_is_released_before_backing_off(self, monkeypatch) -> None:
        """
        regression test. holding the slot across the backoff sleep collapses
        effective concurrency exactly while the service is throttling, because
        every worker sits on a slot doing nothing for up to two minutes.
        """
        adaptive = AdaptiveConcurrency(default_limit=4)
        in_flight_during_sleep: list[int] = []
        calls = {"n": 0}

        async def fake_acompletion(**kwargs: Any) -> _FakeCompletion:
            calls["n"] += 1
            if calls["n"] == 1:
                raise Exception("ThrottlingException: too many requests")
            return _FakeCompletion("ok")

        async def recording_sleep(_seconds: float) -> None:
            in_flight_during_sleep.append(adaptive._in_flight)

        monkeypatch.setattr(bedrock_retry.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(bedrock_retry.asyncio, "sleep", recording_sleep)

        result = asyncio.run(
            acompletion_with_retry(
                model="m",
                messages=[{"role": "user", "content": "x"}],
                adaptive=adaptive,
            )
        )
        assert result == "ok"
        assert in_flight_during_sleep == [0]

    def test_retry_budget_is_enforced_on_transient_errors(self, monkeypatch) -> None:
        async def fake_acompletion(**kwargs: Any) -> _FakeCompletion:
            raise Exception("Read timed out")

        async def no_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr(bedrock_retry.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(bedrock_retry.asyncio, "sleep", no_sleep)
        with pytest.raises(RetryBudgetExceeded):
            asyncio.run(
                acompletion_with_retry(
                    model="m",
                    messages=[{"role": "user", "content": "x"}],
                    max_retry_minutes=0.0,
                )
            )

    def test_throttling_ignores_the_budget_and_keeps_going(self, monkeypatch) -> None:
        """
        throttling is the one case where giving up wastes work, so the budget
        logs and continues rather than raising.
        """
        calls = {"n": 0}

        async def fake_acompletion(**kwargs: Any) -> _FakeCompletion:
            calls["n"] += 1
            if calls["n"] < 3:
                raise Exception("ThrottlingException: too many requests")
            return _FakeCompletion("eventually")

        async def no_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr(bedrock_retry.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(bedrock_retry.asyncio, "sleep", no_sleep)
        result = asyncio.run(
            acompletion_with_retry(
                model="m",
                messages=[{"role": "user", "content": "x"}],
                max_retry_minutes=0.0,
            )
        )
        assert result == "eventually"

    def test_retry_after_header_drives_the_wait(self, monkeypatch) -> None:
        slept: list[float] = []
        calls = {"n": 0}

        async def fake_acompletion(**kwargs: Any) -> _FakeCompletion:
            calls["n"] += 1
            if calls["n"] == 1:
                exc = Exception("ThrottlingException")
                exc.response = _Response({"Retry-After": "17"})
                raise exc
            return _FakeCompletion("ok")

        async def recording_sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr(bedrock_retry.litellm, "acompletion", fake_acompletion)
        monkeypatch.setattr(bedrock_retry.asyncio, "sleep", recording_sleep)
        asyncio.run(
            acompletion_with_retry(
                model="m", messages=[{"role": "user", "content": "x"}]
            )
        )
        assert slept == [pytest.approx(17.0)]
