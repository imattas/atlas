//! Safe ranker tests.

use std::collections::BTreeMap;

use atlas_learning::{RankError, RankRequest, RankResponse, SafeRanker};

fn request() -> RankRequest {
    RankRequest {
        strategy_ids: vec!["gf2".to_owned(), "general-smt".to_owned()],
        features: BTreeMap::from([("vars".to_owned(), 32.0)]),
        seed: 42,
    }
}

#[test]
fn ranks_deterministically_with_offline_fallback() {
    let first = SafeRanker::rank(&request(), None).unwrap();
    let second = SafeRanker::rank(&request(), None).unwrap();

    assert_eq!(first, second);
    assert!(first.explanation.contains("offline fallback"));
}

#[test]
fn rejects_missing_features_and_corrupt_model() {
    let mut missing = request();
    missing.features.clear();

    assert_eq!(
        SafeRanker::rank(&missing, None),
        Err(RankError::MissingFeatures)
    );
    assert_eq!(
        SafeRanker::rank(&request(), Some(b"corrupt")),
        Err(RankError::CorruptModel)
    );
}

#[test]
fn validates_allowlisted_output_fields_and_unknown_strategies() {
    let response = SafeRanker::rank(&request(), Some(b"model")).unwrap();
    SafeRanker::validate_response(&response, &request().strategy_ids).unwrap();

    let bad = RankResponse {
        ordered_strategy_ids: vec!["unknown".to_owned()],
        budget_multipliers: BTreeMap::new(),
        explanation: "bad".to_owned(),
    };
    assert_eq!(
        SafeRanker::validate_response(&bad, &request().strategy_ids),
        Err(RankError::UnknownStrategy("unknown".to_owned()))
    );
}

#[test]
fn rank_response_cannot_construct_facts_or_results_by_type_shape() {
    let response = SafeRanker::rank(&request(), None).unwrap();
    let debug = format!("{response:?}");

    assert!(!debug.contains("Fact"));
    assert!(!debug.contains("Candidate"));
    assert!(!debug.contains("ResultLevel"));
    assert!(!debug.contains("Validation"));
    assert!(!debug.contains("Assumption"));
}
