//! Worker sandbox contract tests.

use atlas_worker::{SandboxControl, SandboxPolicy};

#[test]
fn default_sandbox_policy_requires_non_root_isolation_and_denials() {
    let policy = SandboxPolicy::default();

    assert!(policy.has(SandboxControl::NonRoot));
    assert!(policy.has(SandboxControl::NetworkDisabled));
    assert!(policy.has(SandboxControl::DenyHostEnv));
    assert!(policy.has(SandboxControl::DenyWriteFilesystem));
    assert!(policy.has(SandboxControl::DenyDockerSocket));
    assert!(policy.has(SandboxControl::DenyUnrelatedArtifacts));
    assert!(policy.has(SandboxControl::ReadOnlyArtifacts));
}

#[test]
fn sandbox_policy_rejects_unrelated_artifacts() {
    let policy = SandboxPolicy::default();

    assert!(policy.allows_artifact("challenge.bin", &["challenge.bin", "solver.wasm"]));
    assert!(!policy.allows_artifact("host-secret.env", &["challenge.bin", "solver.wasm"]));
}
