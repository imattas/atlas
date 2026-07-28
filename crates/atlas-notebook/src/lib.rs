//! Event-driven notebook debugger state for Atlas solve streams.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Versioned production event categories consumed by the notebook.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventKind {
    /// Constraint graph updates.
    Constraint,
    /// Domain and partition analysis updates.
    Domain,
    /// Strategy scheduling and result updates.
    Strategy,
    /// Resource usage samples.
    Resource,
    /// Fact-store updates.
    Fact,
    /// Candidate or final model updates.
    Model,
    /// Validation evidence updates.
    Validation,
    /// Source provenance updates.
    Provenance,
    /// Cancellation notification.
    Cancellation,
    /// Solve report export notification.
    Report,
}

/// Event envelope accepted by the notebook debugger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventEnvelopeV1 {
    /// Solve stream identifier.
    pub stream_id: String,
    /// Monotonic sequence number scoped to the stream.
    pub sequence: u64,
    /// Producer timestamp in milliseconds.
    pub emitted_ms: u64,
    /// Event category.
    pub kind: EventKind,
    /// Stable subject id for graph/detail views.
    pub subject: String,
    /// String payload for deterministic, dependency-free tests.
    pub payload: BTreeMap<String, String>,
    /// Payload fields that must never be displayed.
    pub redacted_fields: BTreeSet<String>,
    /// Forward-compatible fields ignored by core state.
    pub compatible_extensions: BTreeMap<String, String>,
}

/// Exportable notebook report summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveReportV1 {
    /// Solve stream identifier.
    pub stream_id: String,
    /// Number of retained events.
    pub retained_events: usize,
    /// Whether cancellation has been observed.
    pub cancelled: bool,
    /// Per-kind event counts.
    pub counts: BTreeMap<EventKind, usize>,
    /// Redacted fields observed across retained events.
    pub redacted_fields: BTreeSet<String>,
}

/// Projection used by graph, detail, progress, and replay panels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotebookState {
    /// Number of retained events after capacity bounding.
    pub retained_events: usize,
    /// Number of duplicate events rejected during ingest or replay.
    pub duplicate_events: usize,
    /// Whether cancellation has been observed.
    pub cancelled: bool,
    /// Latest safe detail payload by subject.
    pub detail_by_subject: BTreeMap<String, BTreeMap<String, String>>,
    /// Per-kind event counts.
    pub counts: BTreeMap<EventKind, usize>,
    /// Keyboard-accessible commands exposed by the view model.
    pub commands: Vec<&'static str>,
}

/// Bounded virtualized event store for live and recorded notebook streams.
pub struct NotebookEventStore {
    capacity: usize,
    events: VecDeque<EventEnvelopeV1>,
    seen: BTreeSet<(String, u64)>,
    duplicate_events: usize,
    cancelled: bool,
}

impl NotebookEventStore {
    /// Creates a bounded event store.
    ///
    /// # Panics
    ///
    /// Panics when `capacity` is zero because an empty store cannot produce a useful view.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "notebook event capacity must be positive");
        Self {
            capacity,
            events: VecDeque::new(),
            seen: BTreeSet::new(),
            duplicate_events: 0,
            cancelled: false,
        }
    }

    /// Ingests one live event, returning `true` when it changes state.
    pub fn ingest_live(&mut self, event: EventEnvelopeV1) -> bool {
        let key = (event.stream_id.clone(), event.sequence);
        if !self.seen.insert(key) {
            self.duplicate_events += 1;
            return false;
        }
        if event.kind == EventKind::Cancellation {
            self.cancelled = true;
        }
        self.events.push_back(event);
        while self.events.len() > self.capacity {
            self.events.pop_front();
        }
        true
    }

    /// Replays recorded events in deterministic sequence order.
    pub fn replay<I>(&mut self, events: I)
    where
        I: IntoIterator<Item = EventEnvelopeV1>,
    {
        let mut ordered: Vec<_> = events.into_iter().collect();
        ordered.sort_by(|left, right| {
            left.stream_id
                .cmp(&right.stream_id)
                .then(left.sequence.cmp(&right.sequence))
        });
        for event in ordered {
            self.ingest_live(event);
        }
    }

    /// Returns a sanitized state projection for notebook panels.
    #[must_use]
    pub fn state(&self) -> NotebookState {
        let mut counts = BTreeMap::new();
        let mut detail_by_subject = BTreeMap::new();
        for event in &self.events {
            *counts.entry(event.kind).or_insert(0) += 1;
            detail_by_subject.insert(event.subject.clone(), sanitized_payload(event));
        }
        NotebookState {
            retained_events: self.events.len(),
            duplicate_events: self.duplicate_events,
            cancelled: self.cancelled,
            detail_by_subject,
            counts,
            commands: vec!["focus-next", "focus-previous", "replay", "export-report"],
        }
    }

    /// Exports a report summary derived only from retained production events.
    #[must_use]
    pub fn export_report(&self) -> SolveReportV1 {
        let mut counts = BTreeMap::new();
        let mut redacted_fields = BTreeSet::new();
        let stream_id = self
            .events
            .front()
            .map_or_else(String::new, |event| event.stream_id.clone());
        for event in &self.events {
            *counts.entry(event.kind).or_insert(0) += 1;
            redacted_fields.extend(event.redacted_fields.iter().cloned());
        }
        SolveReportV1 {
            stream_id,
            retained_events: self.events.len(),
            cancelled: self.cancelled,
            counts,
            redacted_fields,
        }
    }
}

fn sanitized_payload(event: &EventEnvelopeV1) -> BTreeMap<String, String> {
    event
        .payload
        .iter()
        .map(|(key, value)| {
            if event.redacted_fields.contains(key) {
                (key.clone(), "<redacted>".to_owned())
            } else {
                (key.clone(), value.clone())
            }
        })
        .collect()
}
