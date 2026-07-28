//! Validator end-to-end contract tests.

use atlas_ucir::{Builder, Model, Value};
use atlas_validator::{Candidate, CheckerResult, ResultLevel, ValidationContext, Validator};

fn equality_graph() -> atlas_ucir::ExprGraph {
    let mut builder = Builder::new();
    let x = builder.bitvec_var("x", 8).unwrap();
    let expected = builder.bitvec_const(8, 0x41).unwrap();
    let root = builder.eq(x, expected).unwrap();
    builder.finish_with_root(root).unwrap()
}

#[test]
fn promotes_only_ucir_and_checker_valid_candidates_to_validated_sat() {
    let mut model = Model::new();
    model.insert("x".to_owned(), Value::bitvec(8, 0x41).unwrap());
    let candidate = Candidate { model };
    let context = ValidationContext {
        graph: equality_graph(),
        checker: CheckerResult::Accepted,
    };

    assert_eq!(
        Validator::validate(&candidate, &context),
        ResultLevel::ValidatedSat
    );
}

#[test]
fn checker_rejection_prevents_validated_sat() {
    let mut model = Model::new();
    model.insert("x".to_owned(), Value::bitvec(8, 0x41).unwrap());
    let candidate = Candidate { model };
    let context = ValidationContext {
        graph: equality_graph(),
        checker: CheckerResult::Rejected,
    };

    assert_eq!(
        Validator::validate(&candidate, &context),
        ResultLevel::Unknown
    );
}

#[test]
fn missing_checker_returns_model_only_after_ucir_validation() {
    let mut model = Model::new();
    model.insert("x".to_owned(), Value::bitvec(8, 0x41).unwrap());
    let candidate = Candidate { model };
    let context = ValidationContext {
        graph: equality_graph(),
        checker: CheckerResult::Unavailable,
    };

    assert_eq!(
        Validator::validate(&candidate, &context),
        ResultLevel::ModelOnly
    );
}

#[test]
fn trusted_unsat_backend_is_required_for_proven_unsat() {
    assert_eq!(Validator::proven_unsat(true), ResultLevel::ProvenUnsat);
    assert_eq!(Validator::proven_unsat(false), ResultLevel::Unknown);
}
