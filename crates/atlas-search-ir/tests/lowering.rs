//! Search IR lowering tests.

use atlas_search_ir::{SearchIrError, SearchOp, SearchProgram};

#[test]
fn accepts_arithmetic_bitwise_and_checksum_fixtures() {
    assert!(SearchProgram::try_from_fixture("add").unwrap().accepts(3));
    assert!(SearchProgram::try_from_fixture("xor")
        .unwrap()
        .accepts(0x55));
    assert!(SearchProgram::try_from_fixture("checksum")
        .unwrap()
        .accepts(20));
}

#[test]
fn rejects_forbidden_memory_aliasing_and_unbounded_loops() {
    assert_eq!(
        SearchProgram::try_from_fixture("alias"),
        Err(SearchIrError::ForbiddenMemoryAliasing)
    );
    assert_eq!(
        SearchProgram::try_from_fixture("loop"),
        Err(SearchIrError::UnboundedLoop)
    );
}

#[test]
fn supports_64_bit_candidate_widths_for_hardware_search() {
    let program = SearchProgram::new(
        64,
        vec![SearchOp::XorEq {
            mask: 1,
            target: 0x8000_0000_0000_0001,
        }],
    )
    .unwrap();

    assert!(program.accepts(0x8000_0000_0000_0000));
}

#[test]
fn rejects_unsupported_widths_and_empty_programs() {
    assert_eq!(
        SearchProgram::new(0, Vec::new()),
        Err(SearchIrError::UnsupportedWidth)
    );
    assert_eq!(
        SearchProgram::new(65, Vec::new()),
        Err(SearchIrError::UnsupportedWidth)
    );
    assert_eq!(SearchProgram::new(8, Vec::new()), Err(SearchIrError::Empty));
}
