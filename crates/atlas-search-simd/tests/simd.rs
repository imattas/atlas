//! SIMD differential tests.

use atlas_scheduler::CancellationToken;
use atlas_search_ir::{SearchDomain, SearchOp, SearchProgram};
use atlas_search_native::NativeSearcher;
use atlas_search_simd::{SimdEngine, SimdSearcher};

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
fn simd_report_uses_wide_vector_engine_for_regular_batches() {
    let token = CancellationToken::new();
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let domain = SearchDomain::new(0, 512);

    let report = SimdSearcher::search_report(&program, domain, &token, 8);

    assert_eq!(report.engine, SimdEngine::WideU64x4);
    assert_eq!(
        report.matches,
        NativeSearcher::search(&program, domain, &token)
    );
}

#[test]
fn simd_wide_engine_matches_native_for_full_restricted_op_set() {
    let token = CancellationToken::new();
    let program = SearchProgram::new(
        24,
        vec![
            SearchOp::XorEq {
                mask: 0xaa,
                target: 0xff,
            },
            SearchOp::AddEq {
                addend: 1,
                target: 4,
            },
            SearchOp::ChecksumEq {
                modulus: 17,
                target: 3,
            },
            SearchOp::MulAddEq {
                multiplier: 65_537,
                addend: 0x1337,
                target: 0xC0_FF_EE,
            },
            SearchOp::RotateXorEq {
                rotate_left: 7,
                mask: 0xA5_A5_A5,
                target: 0x12_34_56,
            },
            SearchOp::ByteEq {
                byte_index: 1,
                value: b'T',
            },
        ],
    )
    .unwrap();
    let domain = SearchDomain::new(0, 4_096);

    let report = SimdSearcher::search_report(&program, domain, &token, 8);

    assert_eq!(report.engine, SimdEngine::WideU64x4);
    assert_eq!(
        report.matches,
        NativeSearcher::search(&program, domain, &token)
    );
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
