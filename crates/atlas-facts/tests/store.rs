//! Fact store contract tests.

use std::collections::BTreeSet;

use atlas_facts::{assignment, Fact, FactPayload, FactStore, PublishOutcome, TrustLevel};

#[test]
fn rejects_heuristic_mandatory_facts() {
    let mut store = FactStore::new([]);
    let fact = assignment("root", "x", "1", TrustLevel::Heuristic, true);

    assert!(
        matches!(store.publish(fact), PublishOutcome::Rejected(reason) if reason.contains("mandatory"))
    );
}

#[test]
fn rejects_assumption_scope_mismatch() {
    let mut store = FactStore::new(["a".to_owned()]);
    let mut assumptions = BTreeSet::new();
    assumptions.insert("missing".to_owned());
    let fact = Fact {
        scope: "root".to_owned(),
        assumptions,
        trust: TrustLevel::Proven,
        mandatory: true,
        producer: "unit".to_owned(),
        payload: FactPayload::Note("scoped".to_owned()),
    };

    assert!(
        matches!(store.publish(fact), PublishOutcome::Rejected(reason) if reason.contains("scope"))
    );
}

#[test]
fn detects_conflicting_assignments_without_last_writer_wins() {
    let mut store = FactStore::new([]);
    assert_eq!(
        store.publish(assignment("root", "x", "1", TrustLevel::Proven, true)),
        PublishOutcome::Accepted
    );

    let outcome = store.publish(assignment("root", "x", "2", TrustLevel::Proven, true));

    assert!(matches!(outcome, PublishOutcome::Conflict { .. }));
    assert_eq!(store.facts().len(), 1);
}
