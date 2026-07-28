//! Track 2 frontend intake tests.

use atlas_program::{detect, lower, Architecture, Artifact, ArtifactKind, FrontendError};

#[test]
fn detects_elf_metadata_without_executing_artifact() {
    let artifact = Artifact::new("checker", b"\x7fELF\x02\x01\x01".to_vec());

    assert_eq!(detect(&artifact), Ok(ArtifactKind::Elf));
    assert_eq!(
        lower(&artifact).unwrap().program.architecture,
        Architecture::X86_64
    );
}

#[test]
fn lowers_restricted_c_and_python_sources() {
    let c = Artifact::new("checker.c", b"int main(){return 0;}".to_vec());
    let py = Artifact::new("checker.py", b"def check(x): return x == 1".to_vec());

    assert_eq!(
        lower(&c).unwrap().program.architecture,
        Architecture::RestrictedC
    );
    assert_eq!(
        lower(&py).unwrap().program.architecture,
        Architecture::RestrictedPython
    );
}

#[test]
fn lowers_trace_artifacts_with_event_bounds() {
    let trace = Artifact::new("run.trace.json", br#"{"events":[]}"#.to_vec());

    assert_eq!(
        lower(&trace).unwrap().program.architecture,
        Architecture::Trace
    );
}

#[test]
fn rejects_malformed_and_dangerous_artifacts_cleanly() {
    let trace = Artifact::new("run.trace.json", br#"{"bad":[]}"#.to_vec());
    assert!(matches!(lower(&trace), Err(FrontendError::Malformed(_))));

    let py = Artifact::new("checker.py", b"import subprocess".to_vec());
    assert!(matches!(lower(&py), Err(FrontendError::Malformed(_))));
}

#[test]
fn parser_limit_rejects_oversized_artifacts() {
    let artifact = Artifact::new("large.py", vec![b'a'; 1024 * 1024 + 1]);

    assert_eq!(detect(&artifact), Err(FrontendError::TooLarge));
}
