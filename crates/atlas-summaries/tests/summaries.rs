//! Function summary tests.

use atlas_summaries::{
    bounded_strlen, builtin_libc, MemoryEffect, ReturnEffect, SummaryError, SummaryManifest,
    SummaryRegistry, SUMMARY_SCHEMA_MAJOR,
};

#[test]
fn resolves_exact_abi_and_version() {
    let mut registry = SummaryRegistry::new();
    for summary in builtin_libc() {
        registry.register(summary).unwrap();
    }

    let strlen = registry.resolve("strlen", "sysv", 1).unwrap();

    assert_eq!(strlen.memory_effect, MemoryEffect::ReadOnly);
    assert_eq!(
        strlen.return_effect,
        ReturnEffect::BoundedLength { max: 4096 }
    );
    assert!(strlen.provenance.contains("fixture"));
}

#[test]
fn rejects_ambiguity_incompatible_schema_and_unsupported_calls() {
    let mut registry = SummaryRegistry::new();
    let summary = builtin_libc().remove(0);
    registry.register(summary.clone()).unwrap();

    assert_eq!(registry.register(summary), Err(SummaryError::Ambiguous));
    assert_eq!(
        registry.resolve("missing", "sysv", 1),
        Err(SummaryError::Unsupported)
    );
    assert_eq!(
        registry.register(SummaryManifest {
            schema_major: SUMMARY_SCHEMA_MAJOR + 1,
            symbol: "x".to_owned(),
            abi: "sysv".to_owned(),
            version: 1,
            memory_effect: MemoryEffect::ReadOnly,
            return_effect: ReturnEffect::ErrorCode { code: -1 },
            provenance: "bad".to_owned(),
        }),
        Err(SummaryError::IncompatibleSchema)
    );
}

#[test]
fn bounded_strlen_summary_matches_concrete_reference() {
    for bytes in [
        b"abc\0zzz".as_slice(),
        b"abcdef".as_slice(),
        b"\0".as_slice(),
    ] {
        assert_eq!(
            bounded_strlen(bytes, 4),
            bytes
                .iter()
                .take(4)
                .position(|b| *b == 0)
                .unwrap_or(bytes.len().min(4))
        );
    }
}
