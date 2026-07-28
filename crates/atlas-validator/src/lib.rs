//! Candidate validation for `AtlasCTF` solve results.

use atlas_ucir::{Evaluator, ExprGraph, Model, Value};

/// Public result level used by reports and clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResultLevel {
    /// Trusted backend proved unsatisfiable.
    ProvenUnsat,
    /// Candidate satisfies UCIR and available original checker.
    ValidatedSat,
    /// Candidate satisfies UCIR, but original checker is unavailable.
    ModelOnly,
    /// Useful nonterminal deductions exist.
    Partial,
    /// No supported conclusion.
    Unknown,
}

/// Candidate model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// Variable assignment model.
    pub model: Model,
}

/// Optional original checker result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckerResult {
    /// Original checker accepted the candidate.
    Accepted,
    /// Original checker rejected the candidate.
    Rejected,
    /// No original checker exists.
    Unavailable,
}

/// Validation context.
#[derive(Debug, Clone)]
pub struct ValidationContext {
    /// Original UCIR graph.
    pub graph: ExprGraph,
    /// Original checker status after isolated replay.
    pub checker: CheckerResult,
}

/// Candidate validator.
pub struct Validator;

impl Validator {
    /// Validates a candidate against UCIR and optional original checker.
    #[must_use]
    pub fn validate(candidate: &Candidate, context: &ValidationContext) -> ResultLevel {
        match Evaluator::evaluate(&context.graph, &candidate.model) {
            Ok(Value::Bool(true)) => match context.checker {
                CheckerResult::Accepted => ResultLevel::ValidatedSat,
                CheckerResult::Unavailable => ResultLevel::ModelOnly,
                CheckerResult::Rejected => ResultLevel::Unknown,
            },
            Ok(_) | Err(_) => ResultLevel::Unknown,
        }
    }

    /// Promotes a trusted unsat proof to the terminal result level.
    #[must_use]
    pub fn proven_unsat(trusted_backend: bool) -> ResultLevel {
        if trusted_backend {
            ResultLevel::ProvenUnsat
        } else {
            ResultLevel::Unknown
        }
    }
}
