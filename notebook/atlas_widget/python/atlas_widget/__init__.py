"""Atlas notebook debugger view model."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class EventKind(str, Enum):
    """Versioned event categories rendered by the notebook."""

    CONSTRAINT = "constraint"
    DOMAIN = "domain"
    STRATEGY = "strategy"
    RESOURCE = "resource"
    FACT = "fact"
    MODEL = "model"
    VALIDATION = "validation"
    PROVENANCE = "provenance"
    CANCELLATION = "cancellation"
    REPORT = "report"


@dataclass(frozen=True)
class EventEnvelopeV1:
    """Production event envelope accepted by the widget model."""

    stream_id: str
    sequence: int
    emitted_ms: int
    kind: EventKind
    subject: str
    payload: Mapping[str, str]
    redacted_fields: frozenset[str] = frozenset()
    compatible_extensions: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SolveReportV1:
    """Report summary exported from retained events."""

    stream_id: str
    retained_events: int
    cancelled: bool
    counts: Mapping[EventKind, int]
    redacted_fields: frozenset[str]


class EventStore:
    """Bounded event store used by live widgets and recorded replay."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._events: OrderedDict[tuple[str, int], EventEnvelopeV1] = OrderedDict()
        self.duplicate_events = 0
        self.cancelled = False

    def ingest_live(self, event: EventEnvelopeV1) -> bool:
        """Ingest one event and return whether visible state changed."""

        key = (event.stream_id, event.sequence)
        if key in self._events:
            self.duplicate_events += 1
            return False
        if event.kind is EventKind.CANCELLATION:
            self.cancelled = True
        self._events[key] = event
        while len(self._events) > self._capacity:
            self._events.popitem(last=False)
        return True

    def replay(self, events: list[EventEnvelopeV1]) -> None:
        """Replay recorded events in deterministic stream/sequence order."""

        for event in sorted(events, key=lambda item: (item.stream_id, item.sequence)):
            self.ingest_live(event)

    def state(self) -> dict[str, object]:
        """Return sanitized state for graph, detail, progress, and command panels."""

        counts: dict[EventKind, int] = {}
        detail_by_subject: dict[str, dict[str, str]] = {}
        for event in self._events.values():
            counts[event.kind] = counts.get(event.kind, 0) + 1
            detail_by_subject[event.subject] = {
                key: "<redacted>" if key in event.redacted_fields else value
                for key, value in event.payload.items()
            }
        return {
            "retained_events": len(self._events),
            "duplicate_events": self.duplicate_events,
            "cancelled": self.cancelled,
            "counts": counts,
            "detail_by_subject": detail_by_subject,
            "commands": ("focus-next", "focus-previous", "replay", "export-report"),
        }

    def export_report(self) -> SolveReportV1:
        """Export a report derived only from retained production events."""

        counts: dict[EventKind, int] = {}
        redacted_fields: set[str] = set()
        stream_id = ""
        for event in self._events.values():
            stream_id = stream_id or event.stream_id
            counts[event.kind] = counts.get(event.kind, 0) + 1
            redacted_fields.update(event.redacted_fields)
        return SolveReportV1(
            stream_id=stream_id,
            retained_events=len(self._events),
            cancelled=self.cancelled,
            counts=counts,
            redacted_fields=frozenset(redacted_fields),
        )
