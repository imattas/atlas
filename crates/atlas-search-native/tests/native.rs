//! Native bounded search tests.

use atlas_scheduler::CancellationToken;
use atlas_search_ir::{SearchDomain, SearchProgram};
use atlas_search_native::NativeSearcher;

#[test]
fn native_search_matches_restricted_ir_evaluator_on_small_domain() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let domain = SearchDomain::new(0, 16);
    let token = CancellationToken::new();

    let native = NativeSearcher::search(&program, domain, &token);
    let expected: Vec<_> = (0..16)
        .filter(|candidate| program.accepts(*candidate))
        .collect();

    assert_eq!(native, expected);
    assert_eq!(native, vec![3]);
}

#[test]
fn native_search_polls_cancellation() {
    let program = SearchProgram::try_from_fixture("checksum").unwrap();
    let token = CancellationToken::new();
    token.cancel();

    assert!(NativeSearcher::search(&program, SearchDomain::new(0, 100), &token).is_empty());
}
