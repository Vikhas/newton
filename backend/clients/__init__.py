"""Model clients. `get_clients()` returns (spark, critic): Meta Model API + Muse Spark Open, or fakes offline."""
import time
from dataclasses import dataclass

import numpy as np

RETRY_DELAYS = (1, 2, 4)


class ModelError(RuntimeError):
    """A model call failed after retries, or could not be made at all."""


@dataclass
class Instance:
    """One object's mask in one frame. The mask is local to its box; (left, top) places it in the frame."""
    mask: np.ndarray
    left: int
    top: int
    score: float | None = None


def call_with_retries(fn, is_retryable):
    """Run fn(); retry retryable errors with 1, 2, 4 s backoff, then raise ModelError."""
    for attempt, delay in enumerate((*RETRY_DELAYS, None)):
        try:
            return fn()
        except ModelError:
            raise
        except Exception as e:  # noqa: BLE001 - wrapped below
            if delay is None or not is_retryable(e):
                raise ModelError(f"{type(e).__name__}: {e}") from e
            time.sleep(delay)


def get_clients():
    from backend import config
    if config.OFFLINE:
        from backend.clients.fake import FakeClient
        return FakeClient(), FakeClient()
    from backend.clients.meta import MetaClient
    from backend.clients.open import OpenClient
    return MetaClient(), OpenClient()
