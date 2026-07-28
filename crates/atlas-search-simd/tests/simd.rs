//! SIMD differential tests.

use atlas_scheduler::CancellationToken;
use atlas_search_ir::{SearchDomain, SearchProgram};
use atlas_search_native::NativeSearcher;
use atlas_search_simd::SimdSearcher;

#[test]
fn simd_matches_native_for_widths_tails_multiple_and_no_matches() {
    let token = CancellationToken::new();
    for fixture in ["add", "xor", "checksum"] {
        let program = SearchProgram::try_from_fixture(fixture).unwrap();
        for end in [1, 7, 16, 31, 64] {
            let domain = SearchDomain::new(0, end);
            assert_eq!(
                SimdSearcher::search(&program, domain, &token, 8),
                NativeSearcher::search(&program, domain, &token)
            );
        }
    }
}

#[test]
fn simd_honors_cancellation() {
    let program = SearchProgram::try_from_fixture("checksum").unwrap();
    let token = CancellationToken::new();
    token.cancel();

    assert!(SimdSearcher::search(&program, SearchDomain::new(0, 100), &token, 4).is_empty());
}
