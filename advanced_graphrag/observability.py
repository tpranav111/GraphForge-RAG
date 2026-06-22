"""Small tracing primitives for production instrumentation."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class TraceEvent:
    name: str
    elapsed_ms: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class RunTrace:
    """Collects stage timings.

    Database and model adapters can forward these events to OpenTelemetry,
    Langfuse, Phoenix, or simple JSON logs.
    """

    events: list[TraceEvent] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, **metadata: object) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.events.append(TraceEvent(name=name, elapsed_ms=elapsed_ms, metadata=dict(metadata)))

