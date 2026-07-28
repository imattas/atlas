//! Notebook event store behavior tests.

use std::collections::{BTreeMap, BTreeSet};

use atlas_notebook::{EventEnvelopeV1, EventKind, NotebookEventStore};

fn event(sequence: u64, kind: EventKind, subject: &str, key: &str, value: &str) -> EventEnvelopeV1 {
    EventEnvelopeV1 {
        stream_id: "solve-1".to_owned(),
        sequence,
        emitted_ms: sequence * 10,
        kind,
        subject: subject.to_owned(),
        payload: BTreeMap::from([(key.to_owned(), value.to_owned())]),
        redacted_fields: BTreeSet::new(),
        compatible_extensions: BTreeMap::new(),
    }
}

#[test]
fn live_streaming_updates_graph_detail_progress_and_report_views() {
    let mut store = NotebookEventStore::new(16);
    assert!(store.ingest_live(event(
        1,
        EventKind::Constraint,
        "constraint:a",
        "expr",
        "x == 7"
    )));
    assert!(store.ingest_live(event(
        2,
        EventKind::Strategy,
        "strategy:gf2",
        "status",
        "running"
    )));
    assert!(store.ingest_live(event(3, EventKind::Resource, "resource:cpu", "ms", "12")));
    assert!(store.ingest_live(event(
        4,
        EventKind::Validation,
        "validation:model",
        "status",
        "valid"
    )));

    let state = store.state();
    assert_eq!(state.retained_events, 4);
    assert_eq!(state.counts[&EventKind::Constraint], 1);
    assert_eq!(state.detail_by_subject["strategy:gf2"]["status"], "running");
    assert!(state.commands.contains(&"focus-next"));

    let report = store.export_report();
    assert_eq!(report.stream_id, "solve-1");
    assert_eq!(report.counts[&EventKind::Validation], 1);
}

#[test]
fn recorded_replay_orders_events_and_deduplicates_reconnects() {
    let mut store = NotebookEventStore::new(16);
    store.replay([
        event(3, EventKind::Model, "model:final", "x", "7"),
        event(1, EventKind::Fact, "fact:input", "source", "trace"),
        event(2, EventKind::Domain, "domain:x", "range", "0..10"),
        event(2, EventKind::Domain, "domain:x", "range", "0..10"),
    ]);

    let state = store.state();
    assert_eq!(state.retained_events, 3);
    assert_eq!(state.duplicate_events, 1);
    assert_eq!(state.detail_by_subject["domain:x"]["range"], "0..10");
}

#[test]
fn large_graphs_are_virtualized_by_capacity() {
    let mut store = NotebookEventStore::new(3);
    for sequence in 1..=10 {
        assert!(store.ingest_live(event(
            sequence,
            EventKind::Constraint,
            &format!("constraint:{sequence}"),
            "expr",
            "bounded"
        )));
    }

    let state = store.state();
    assert_eq!(state.retained_events, 3);
    assert!(!state.detail_by_subject.contains_key("constraint:1"));
    assert!(state.detail_by_subject.contains_key("constraint:10"));
}

#[test]
fn cancellation_and_unknown_compatible_fields_are_safe() {
    let mut event = event(1, EventKind::Cancellation, "solve-1", "reason", "budget");
    event
        .compatible_extensions
        .insert("future_ui_hint".to_owned(), "ignored".to_owned());
    let mut store = NotebookEventStore::new(8);
    assert!(store.ingest_live(event));

    let state = store.state();
    assert!(state.cancelled);
    assert_eq!(state.retained_events, 1);
}

#[test]
fn redacted_values_never_reach_visible_state() {
    let mut event = event(1, EventKind::Model, "model:secret", "token", "SECRET_VALUE");
    event.redacted_fields.insert("token".to_owned());
    let mut store = NotebookEventStore::new(8);
    assert!(store.ingest_live(event));

    let state = store.state();
    assert_eq!(
        state.detail_by_subject["model:secret"]["token"],
        "<redacted>"
    );
    assert!(!format!("{state:?}").contains("SECRET_VALUE"));
    assert!(store.export_report().redacted_fields.contains("token"));
}
