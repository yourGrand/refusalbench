"""shared litellm bedrock completion helper with retry and adaptive concurrency."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any

import litellm

from refusalbench.multilingual.env import get_bedrock_bearer_token

logger = logging.getLogger(__name__)

ERROR_RETRYABLE_THROTTLE = "retryable_throttle"
ERROR_RETRYABLE_TRANSIENT = "retryable_transient"
ERROR_PERMANENT = "permanent"

_BACKOFF_FLOOR_SEC = 2.0
_BACKOFF_CEILING_SEC = 120.0

_THROTTLE_PATTERNS = (
    "too many requests",
    "throttlingexception",
    "throttling exception",
    "429",
    "rate exceeded",
    "please wait",
    "retry after",
    "request rate",
    "throughput exceeded",
    "exceeded throughput",
)

_TRANSIENT_PATTERNS = (
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "connection refused",
    "503",
    "502",
    "504",
    "internal server error",
    "service unavailable",
    "bad gateway",
    "endpoint connection",
    "temporarily unavailable",
)

_PERMANENT_PATTERNS = (
    "access denied",
    "accessdenied",
    "unauthorized",
    "forbidden",
    "model not found",
    "not available for this account",
    "unsupportedparams",
    "validation error",
    "validationexception",
    "bad request",
    "invalid request",
    "malformed",
    "not authorized",
    "inference profile",
    "on-demand throughput",
)

_QUIET_LOGGERS = (
    "LiteLLM",
    "LiteLLM Proxy",
    "LiteLLM Router",
    "httpx",
    "httpcore",
)

_EXC_LOG_MAX_LEN = 800


class PermanentBedrockError(Exception):
    """non-retryable bedrock or litellm failure."""


class RetryBudgetExceeded(Exception):
    """optional wall-clock retry budget was exceeded."""


def configure_bedrock_client_logging() -> None:
    """
    reduce litellm and http client noise; retries are logged from this module.
    """
    litellm.suppress_debug_info = True
    litellm.set_verbose = False

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def format_exception_for_log(exc: BaseException) -> str:
    """
    single-line exception text suitable for log lines.

    Parameters
    ----------
    exc
        caught exception from litellm or boto3.

    Returns
    -------
    str
        truncated one-line message.
    """
    text = str(exc).replace("\n", " ").strip()

    if len(text) > _EXC_LOG_MAX_LEN:
        return text[: _EXC_LOG_MAX_LEN - 3] + "..."

    return text


def _exception_blob(exc: BaseException) -> str:
    """
    flatten exception text for pattern matching.

    Parameters
    ----------
    exc
        caught exception from litellm or boto3.

    Returns
    -------
    str
        lowercased combined message text.
    """
    parts: list[str] = [str(exc), repr(exc)]
    response = getattr(exc, "response", None)

    if response is not None:
        parts.append(str(response))

    message = getattr(exc, "message", None)

    if message is not None:
        parts.append(str(message))

    body = getattr(exc, "body", None)

    if body is not None:
        parts.append(str(body))

    return " ".join(parts).lower()


def classify_error(exc: BaseException) -> str:
    """
    map an exception to a retry taxonomy label.

    Parameters
    ----------
    exc
        caught exception from litellm or boto3.

    Returns
    -------
    str
        one of retryable_throttle, retryable_transient, or permanent.
    """
    text = _exception_blob(exc)

    if any(p in text for p in _PERMANENT_PATTERNS):
        return ERROR_PERMANENT

    if any(p in text for p in _THROTTLE_PATTERNS):
        return ERROR_RETRYABLE_THROTTLE

    if any(p in text for p in _TRANSIENT_PATTERNS):
        return ERROR_RETRYABLE_TRANSIENT

    status_code = getattr(exc, "status_code", None)

    if status_code == 429:
        return ERROR_RETRYABLE_THROTTLE

    if status_code in (502, 503, 504):
        return ERROR_RETRYABLE_TRANSIENT

    if status_code in (400, 401, 403, 404, 422):
        return ERROR_PERMANENT

    return ERROR_RETRYABLE_TRANSIENT


def _extract_retry_after_sec(exc: BaseException) -> float | None:
    """
    read retry-after from an httpx-style response when present.

    Parameters
    ----------
    exc
        caught exception that may carry response headers.

    Returns
    -------
    float | None
        seconds to wait or none if unknown.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None

    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None

    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _backoff_seconds(attempt: int, retry_after: float | None) -> float:
    """
    exponential backoff with full jitter.

    Parameters
    ----------
    attempt
        zero-based attempt index after a failure.
    retry_after
        optional server hint in seconds.

    Returns
    -------
    float
        sleep duration before the next retry.
    """
    if retry_after is not None and retry_after > 0:
        return max(_BACKOFF_FLOOR_SEC, min(retry_after, _BACKOFF_CEILING_SEC))

    exp_cap = min(_BACKOFF_CEILING_SEC, _BACKOFF_FLOOR_SEC * (2 ** min(attempt, 8)))

    return random.uniform(_BACKOFF_FLOOR_SEC, exp_cap)


def _message_content(response: Any) -> str:
    """
    extract assistant text from a litellm response object.

    Parameters
    ----------
    response
        litellm completion response.

    Returns
    -------
    str
        message content or empty string.
    """
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""

    if content is None:
        return ""

    return str(content)


class AdaptiveConcurrency:
    """
    per-judge asyncio concurrency limiter that reacts to throttling.

    lowers in-flight cap after sustained throttles and restores slowly
    after a run of successful calls.
    """

    def __init__(self, default_limit: int = 8) -> None:
        """
        initialize adaptive concurrency state.

        Parameters
        ----------
        default_limit
            starting and maximum concurrent in-flight calls.
        """
        self.default_limit = max(1, default_limit)
        self._limit = self.default_limit
        self._in_flight = 0
        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition(self._lock)
        self._success_streak = 0
        self._throttle_times: list[float] = []

    @property
    def current_limit(self) -> int:
        """current concurrency cap."""
        return self._limit

    async def acquire(self) -> None:
        """wait until a concurrency slot is available."""
        async with self._cond:
            while self._in_flight >= self._limit:
                await self._cond.wait()

            self._in_flight += 1

    async def release(self) -> None:
        """release a concurrency slot."""
        async with self._cond:
            self._in_flight = max(0, self._in_flight - 1)
            self._cond.notify()

    def note_throttle(self) -> None:
        """
        record throttling and shrink concurrency when throttles cluster.

        sustained throttles in a short window reduce the limit by one
        down to a minimum of one.
        """
        now = time.monotonic()
        self._throttle_times.append(now)
        window_start = now - 60.0
        self._throttle_times = [t for t in self._throttle_times if t >= window_start]
        self._success_streak = 0

        if len(self._throttle_times) >= 3 and self._limit > 1:
            self._limit -= 1
            logger.info(
                "adaptive concurrency lowered to %s after throttling",
                self._limit,
            )

    async def note_success(self) -> None:
        """
        record success and slowly restore concurrency toward the default.

        every ten consecutive successes raises the limit by one until
        the default is reached again. waiters are woken when the limit rises.
        """
        async with self._cond:
            self._success_streak += 1

            if self._success_streak >= 10 and self._limit < self.default_limit:
                self._limit += 1
                self._success_streak = 0
                logger.info(
                    "adaptive concurrency restored to %s after success streak",
                    self._limit,
                )
                self._cond.notify_all()


async def acompletion_with_retry(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0,
    max_retry_minutes: float | None = None,
    adaptive: AdaptiveConcurrency | None = None,
    region: str | None = None,
    **kwargs: Any,
) -> str:
    """
    call litellm.acompletion with throttle-aware unbounded retries.

    Parameters
    ----------
    model
        litellm model id (for example bedrock/converse/...).
    messages
        chat messages passed to litellm.
    temperature
        sampling temperature.
    max_retry_minutes
        optional wall-clock safety valve in minutes. none means unlimited.
    adaptive
        optional adaptive concurrency helper for this judge.
    region
        aws region name. sets AWS_REGION_NAME when provided.
    **kwargs
        forwarded to litellm.acompletion.

    Returns
    -------
    str
        assistant message content.

    Raises
    ------
    PermanentBedrockError
        when the error taxonomy is permanent.
    RetryBudgetExceeded
        when max_retry_minutes elapses on a non-throttle error path.
    """
    if region:
        os.environ["AWS_REGION_NAME"] = region

    bearer = get_bedrock_bearer_token()
    completion_kwargs = dict(kwargs)
    if bearer and "api_key" not in completion_kwargs:
        completion_kwargs["api_key"] = bearer

    attempt = 0
    started = time.monotonic()

    while True:
        acquired = False
        sleep_sec = 0.0
        try:
            if adaptive is not None:
                await adaptive.acquire()
                acquired = True

            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                drop_params=True,
                **completion_kwargs,
            )
            content = _message_content(response)
            if adaptive is not None:
                await adaptive.note_success()

            if attempt > 0:
                logger.info(
                    "bedrock ok model=%s after %s failed attempt(s)",
                    model,
                    attempt,
                )

            return content
        except Exception as exc:
            error_class = classify_error(exc)
            err_msg = format_exception_for_log(exc)
            if adaptive is not None and error_class == ERROR_RETRYABLE_THROTTLE:
                adaptive.note_throttle()

            if error_class == ERROR_PERMANENT:
                logger.error(
                    "bedrock permanent failure model=%s (%s): %s",
                    model,
                    type(exc).__name__,
                    err_msg,
                )
                raise PermanentBedrockError(str(exc)) from exc

            elapsed_min = (time.monotonic() - started) / 60.0
            if max_retry_minutes is not None and elapsed_min >= max_retry_minutes:
                if error_class == ERROR_RETRYABLE_THROTTLE:
                    logger.warning(
                        "max_retry_minutes reached during throttling for %s, continuing",
                        model,
                    )
                else:
                    logger.error(
                        "bedrock retry budget exceeded model=%s after %.1f min "
                        "(%s attempts, last error %s): %s",
                        model,
                        elapsed_min,
                        attempt + 1,
                        error_class,
                        err_msg,
                    )
                    raise RetryBudgetExceeded(
                        f"retry budget exceeded after {elapsed_min:.1f} minutes"
                    ) from exc

            retry_after = _extract_retry_after_sec(exc)
            sleep_sec = _backoff_seconds(attempt, retry_after)
            logger.warning(
                "bedrock call failed model=%s attempt=%s %s (%s): %s; "
                "retrying in %.1fs",
                model,
                attempt + 1,
                error_class,
                type(exc).__name__,
                err_msg,
                sleep_sec,
            )
            attempt += 1
        finally:
            if acquired and adaptive is not None:
                await adaptive.release()

        # backoff happens after the slot is released, otherwise a failing call
        # would hold its concurrency slot for the whole sleep and throughput
        # would collapse exactly when the service is throttling us.
        if sleep_sec > 0:
            await asyncio.sleep(sleep_sec)
