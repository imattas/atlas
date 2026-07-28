//! Search IR lowering tests.

use atlas_search_ir::{SearchIrError, SearchProgram};

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
fn rejects_unsupported_widths_and_empty_programs() {
    assert_eq!(
        SearchProgram::new(0, Vec::new()),
        Err(SearchIrError::UnsupportedWidth)
    );
    assert_eq!(
        SearchProgram::new(33, Vec::new()),
        Err(SearchIrError::UnsupportedWidth)
    );
    assert_eq!(SearchProgram::new(8, Vec::new()), Err(SearchIrError::Empty));
}
