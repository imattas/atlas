//! SIMD differential tests.

use atlas_scheduler::CancellationToken;
use atlas_search_ir::{SearchDomain, SearchOp, SearchProgram};
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

#[test]
fn simd_preserves_native_output_bound_for_dense_matches() {
    let program = SearchProgram::new(
        64,
        vec![SearchOp::ChecksumEq {
            modulus: 1,
            target: 0,
        }],
    )
    .unwrap();
    let token = CancellationToken::new();
    let domain = SearchDomain::new(0, 2_000);

    let simd = SimdSearcher::search(&program, domain, &token, 8);
    let native = NativeSearcher::search(&program, domain, &token);

    assert_eq!(simd, native);
    assert_eq!(simd.len(), 1024);
}
