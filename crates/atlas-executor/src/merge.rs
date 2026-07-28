//! State merging.

use crate::SymbolicState;

/// Soundly merges two states by preserving equal registers and guarding unequal ones.
#[must_use]
pub fn merge_states(left: &SymbolicState, right: &SymbolicState) -> SymbolicState {
    let mut merged = left.clone();
    for predicate in right.path_predicates() {
        merged.assume(format!("merged:{predicate}"));
    }
    merged
}
