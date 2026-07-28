//! Immutable fact store with trust and provenance validation.

use std::collections::{BTreeMap, BTreeSet};

/// Fact trust classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustLevel {
    /// Mechanically proven by trusted local reasoning.
    Proven,
    /// Certified by a configured trusted solver.
    SolverCertified,
    /// Observed by concrete execution or replay.
    ConcreteObserved,
    /// Useful for ranking only.
    Heuristic,
    /// Approximate and never mandatory.
    Approximate,
}

impl TrustLevel {
    /// Returns whether this trust level may become mandatory.
    #[must_use]
    pub fn can_be_mandatory(self) -> bool {
        matches!(
            self,
            Self::Proven | Self::SolverCertified | Self::ConcreteObserved
        )
    }
}

/// Fact payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactPayload {
    /// Exact assignment.
    Assignment {
        /// Variable name.
        variable: String,
        /// Canonical value string.
        value: String,
    },
    /// Bound fact.
    Bound {
        /// Variable name.
        variable: String,
        /// Inclusive lower bound.
        lower: String,
        /// Inclusive upper bound.
        upper: String,
    },
    /// Planner-only note.
    Note(String),
}

/// Immutable fact record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fact {
    /// Logical scope, usually a component or assumption set id.
    pub scope: String,
    /// Active assumptions required by this fact.
    pub assumptions: BTreeSet<String>,
    /// Trust classification.
    pub trust: TrustLevel,
    /// Whether consumers may treat this fact as mandatory.
    pub mandatory: bool,
    /// Backend or pass that produced the fact.
    pub producer: String,
    /// Fact payload.
    pub payload: FactPayload,
}

/// Publication outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublishOutcome {
    /// Fact accepted.
    Accepted,
    /// Duplicate fact ignored.
    Duplicate,
    /// Fact rejected with reason.
    Rejected(String),
    /// Fact conflicts with an existing fact.
    Conflict {
        /// Existing fact.
        existing: Box<Fact>,
        /// Incoming fact.
        incoming: Box<Fact>,
    },
}

/// In-memory immutable fact store.
#[derive(Debug, Clone, Default)]
pub struct FactStore {
    facts: Vec<Fact>,
    active_assumptions: BTreeSet<String>,
}

impl FactStore {
    /// Creates an empty fact store.
    #[must_use]
    pub fn new(active_assumptions: impl IntoIterator<Item = String>) -> Self {
        Self {
            facts: Vec::new(),
            active_assumptions: active_assumptions.into_iter().collect(),
        }
    }

    /// Publishes a fact after trust, assumption, and conflict validation.
    pub fn publish(&mut self, fact: Fact) -> PublishOutcome {
        if fact.mandatory && !fact.trust.can_be_mandatory() {
            return PublishOutcome::Rejected(
                "heuristic or approximate facts cannot be mandatory".to_owned(),
            );
        }
        if !fact.assumptions.is_subset(&self.active_assumptions) {
            return PublishOutcome::Rejected(
                "fact assumptions are outside active scope".to_owned(),
            );
        }
        if self.facts.iter().any(|existing| existing == &fact) {
            return PublishOutcome::Duplicate;
        }
        if let Some(existing) = self.conflicting_assignment(&fact) {
            return PublishOutcome::Conflict {
                existing: Box::new(existing.clone()),
                incoming: Box::new(fact),
            };
        }
        self.facts.push(fact);
        PublishOutcome::Accepted
    }

    /// Returns accepted facts.
    #[must_use]
    pub fn facts(&self) -> &[Fact] {
        &self.facts
    }

    fn conflicting_assignment(&self, incoming: &Fact) -> Option<&Fact> {
        let FactPayload::Assignment {
            variable,
            value: incoming_value,
        } = &incoming.payload
        else {
            return None;
        };
        self.facts.iter().find(|existing| {
            existing.scope == incoming.scope
                && existing.assumptions == incoming.assumptions
                && matches!(
                    &existing.payload,
                    FactPayload::Assignment {
                        variable: existing_variable,
                        value: existing_value
                    } if existing_variable == variable && existing_value != incoming_value
                )
        })
    }
}

/// Helper for creating assignment facts in tests and adapters.
#[must_use]
pub fn assignment(
    scope: impl Into<String>,
    variable: impl Into<String>,
    value: impl Into<String>,
    trust: TrustLevel,
    mandatory: bool,
) -> Fact {
    Fact {
        scope: scope.into(),
        assumptions: BTreeSet::new(),
        trust,
        mandatory,
        producer: "test".to_owned(),
        payload: FactPayload::Assignment {
            variable: variable.into(),
            value: value.into(),
        },
    }
}

/// Groups assignments by variable for reporting.
#[must_use]
pub fn assignment_index(facts: &[Fact]) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for fact in facts {
        if let FactPayload::Assignment { variable, value } = &fact.payload {
            out.entry(variable.clone()).or_default().push(value.clone());
        }
    }
    out
}
