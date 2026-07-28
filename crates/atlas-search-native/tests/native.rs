//! Native bounded search tests.

use atlas_scheduler::CancellationToken;
use atlas_search_ir::{SearchDomain, SearchOp, SearchProgram};
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

#[test]
fn native_search_solves_invertible_single_ops_without_enumerating_domain() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let domain = SearchDomain::new(0, 1 << 20);
    let token = CancellationToken::new();

    let result = NativeSearcher::search_with_stats(&program, domain, &token);

    assert_eq!(result.matches.len(), 1024);
    assert_eq!(result.matches[0], 0x55);
    assert_eq!(result.matches[1], 0x155);
    assert!(result
        .matches
        .iter()
        .all(|candidate| program.accepts(*candidate)));
    assert!(result.candidates_evaluated <= 1024);
    assert!(result.used_closed_form);
}

#[test]
fn native_search_solves_checksum_residues_without_enumerating_domain() {
    let program = SearchProgram::try_from_fixture("checksum").unwrap();
    let domain = SearchDomain::new(0, 1 << 20);
    let token = CancellationToken::new();

    let result = NativeSearcher::search_with_stats(&program, domain, &token);

    assert_eq!(result.matches.len(), 1024);
    assert_eq!(result.matches[0], 3);
    assert_eq!(result.matches[1], 20);
    assert!(result
        .matches
        .iter()
        .all(|candidate| program.accepts(*candidate)));
    assert!(result.candidates_evaluated <= 1024);
    assert!(result.used_closed_form);
}

#[test]
fn native_search_handles_64_bit_checksum_modulus_one_without_enumerating_residues() {
    let program = SearchProgram::new(
        64,
        vec![SearchOp::ChecksumEq {
            modulus: 1,
            target: 0,
        }],
    )
    .unwrap();
    let token = CancellationToken::new();

    let result =
        NativeSearcher::search_with_stats(&program, SearchDomain::new(0, u64::MAX), &token);

    assert_eq!(result.matches.len(), 1024);
    assert_eq!(result.matches[0], 0);
    assert_eq!(result.matches[1023], 1023);
    assert!(result.candidates_evaluated <= 1024);
    assert!(result.used_closed_form);
}

#[test]
fn native_search_solves_ctf_rotate_xor_and_muladd_checks_without_enumerating() {
    let token = CancellationToken::new();
    let rotate_program = SearchProgram::new(
        24,
        vec![SearchOp::RotateXorEq {
            rotate_left: 7,
            mask: 0xA5_A5_A5,
            target: 0x12_34_56,
        }],
    )
    .unwrap();
    let rotate =
        NativeSearcher::search_with_stats(&rotate_program, SearchDomain::new(0, 1 << 24), &token);

    assert_eq!(rotate.matches.len(), 1);
    assert!(rotate_program.accepts(rotate.matches[0]));
    assert!(rotate.candidates_evaluated <= 1);
    assert!(rotate.used_closed_form);

    let muladd_program = SearchProgram::new(
        24,
        vec![SearchOp::MulAddEq {
            multiplier: 65_537,
            addend: 0x1337,
            target: 0xC0_FF_EE,
        }],
    )
    .unwrap();
    let muladd =
        NativeSearcher::search_with_stats(&muladd_program, SearchDomain::new(0, 1 << 24), &token);

    assert_eq!(muladd.matches.len(), 1);
    assert!(muladd_program.accepts(muladd.matches[0]));
    assert!(muladd.candidates_evaluated <= 1);
    assert!(muladd.used_closed_form);
}

#[test]
fn native_search_uses_fixed_byte_serial_constraints_to_skip_most_candidates() {
    let program = SearchProgram::new(
        32,
        vec![
            SearchOp::ByteEq {
                byte_index: 0,
                value: b'C',
            },
            SearchOp::ByteEq {
                byte_index: 1,
                value: b'T',
            },
            SearchOp::ByteEq {
                byte_index: 2,
                value: b'F',
            },
        ],
    )
    .unwrap();
    let token = CancellationToken::new();

    let result = NativeSearcher::search_with_stats(&program, SearchDomain::new(0, 1 << 32), &token);

    assert_eq!(result.matches.len(), 256);
    assert_eq!(result.matches[0] & 0x00FF_FFFF, 0x0046_5443);
    assert!(result
        .matches
        .iter()
        .all(|candidate| program.accepts(*candidate)));
    assert!(result.candidates_evaluated <= 256);
    assert!(result.used_closed_form);
}
