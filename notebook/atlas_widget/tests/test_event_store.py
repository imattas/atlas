import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from atlas_widget import EventEnvelopeV1, EventKind, EventStore


def event(sequence: int, kind: EventKind, subject: str, payload: dict[str, str]) -> EventEnvelopeV1:
    return EventEnvelopeV1(
        stream_id="solve-1",
        sequence=sequence,
        emitted_ms=sequence * 10,
        kind=kind,
        subject=subject,
        payload=payload,
    )


class EventStoreTest(unittest.TestCase):
    def test_live_streaming_and_report_export(self) -> None:
        store = EventStore(capacity=16)
        self.assertTrue(store.ingest_live(event(1, EventKind.CONSTRAINT, "constraint:a", {"expr": "x == 7"})))
        self.assertTrue(store.ingest_live(event(2, EventKind.STRATEGY, "strategy:gf2", {"status": "running"})))
        self.assertTrue(store.ingest_live(event(3, EventKind.RESOURCE, "resource:cpu", {"ms": "12"})))

        state = store.state()
        self.assertEqual(state["retained_events"], 3)
        self.assertEqual(state["detail_by_subject"]["strategy:gf2"]["status"], "running")
        self.assertIn("focus-next", state["commands"])
        self.assertEqual(store.export_report().counts[EventKind.RESOURCE], 1)

    def test_replay_reconnect_dedup_and_large_graph_bounds(self) -> None:
        store = EventStore(capacity=3)
        store.replay(
            [
                event(3, EventKind.MODEL, "model:final", {"x": "7"}),
                event(1, EventKind.FACT, "fact:input", {"source": "trace"}),
                event(2, EventKind.DOMAIN, "domain:x", {"range": "0..10"}),
                event(2, EventKind.DOMAIN, "domain:x", {"range": "0..10"}),
                event(4, EventKind.CONSTRAINT, "constraint:tail", {"expr": "bounded"}),
            ]
        )

        state = store.state()
        self.assertEqual(state["retained_events"], 3)
        self.assertEqual(state["duplicate_events"], 1)
        self.assertNotIn("fact:input", state["detail_by_subject"])
        self.assertIn("constraint:tail", state["detail_by_subject"])

    def test_cancellation_redaction_and_unknown_compatible_fields(self) -> None:
        store = EventStore(capacity=8)
        cancelled = event(1, EventKind.CANCELLATION, "solve-1", {"reason": "budget"})
        secret = EventEnvelopeV1(
            stream_id="solve-1",
            sequence=2,
            emitted_ms=20,
            kind=EventKind.MODEL,
            subject="model:secret",
            payload={"token": "SECRET_VALUE"},
            redacted_fields=frozenset({"token"}),
            compatible_extensions={"future_ui_hint": "ignored"},
        )

        self.assertTrue(store.ingest_live(cancelled))
        self.assertTrue(store.ingest_live(secret))
        state = store.state()
        self.assertTrue(state["cancelled"])
        self.assertEqual(state["detail_by_subject"]["model:secret"]["token"], "<redacted>")
        self.assertNotIn("SECRET_VALUE", repr(state))
        self.assertIn("token", store.export_report().redacted_fields)


if __name__ == "__main__":
    unittest.main()
