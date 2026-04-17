"""Retry with exponential backoff for transient Qdrant errors.

Wraps Qdrant client calls so that ConnectionError / TimeoutError / transient
HTTP failures are retried automatically instead of immediately returning 503.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import TypeVar, Callable, Any

logger = logging.getLogger(__name__)

# Exceptions that indicate a transient Qdrant issue worth retrying.
_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,  # covers socket-level errors (BrokenPipeError, ConnectionResetError, etc.)
)

# Try to include qdrant-client specific transport errors if available.
try:
    from grpc import RpcError  # type: ignore[import-untyped]
    _TRANSIENT_EXCEPTIONS = (*_TRANSIENT_EXCEPTIONS, RpcError)
except ImportError:
    pass

try:
    from httpx import ConnectError, ReadTimeout, WriteTimeout, PoolTimeout
    _TRANSIENT_EXCEPTIONS = (
        *_TRANSIENT_EXCEPTIONS, ConnectError, ReadTimeout, WriteTimeout, PoolTimeout
    )
except ImportError:
    pass

F = TypeVar("F", bound=Callable[..., Any])


def with_retry(
    max_retries: int = 3,
    base_delay: float = 0.3,
    max_delay: float = 5.0,
    backoff_factor: float = 2.0,
) -> Callable[[F], F]:
    """Decorator: retry on transient Qdrant errors with exponential backoff.

    Parameters
    ----------
    max_retries : int
        Total number of retry attempts after the first failure.
    base_delay : float
        Initial sleep between the first and second attempt (seconds).
    max_delay : float
        Cap on the sleep duration between attempts.
    backoff_factor : float
        Multiplier applied to the delay after each retry.
    """

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            last_exc: BaseException | None = None

            for attempt in range(1, max_retries + 2):  # 1 initial + max_retries
                try:
                    return fn(*args, **kwargs)
                except _TRANSIENT_EXCEPTIONS as exc:
                    last_exc = exc
                    if attempt > max_retries:
                        logger.error(
                            "Qdrant call %s failed after %d attempts: %s",
                            fn.__qualname__, attempt, exc,
                        )
                        raise
                    logger.warning(
                        "Qdrant call %s attempt %d/%d failed (%s), retrying in %.1fs…",
                        fn.__qualname__, attempt, max_retries + 1, exc, delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

            # Unreachable, but keeps type-checkers happy.
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
