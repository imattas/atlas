//! Isolated worker contracts.

use std::collections::BTreeSet;

/// Worker capability advertisement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerCapabilities {
    /// CPU core count.
    pub cpu_cores: u32,
    /// Memory in MiB.
    pub memory_mib: u64,
    /// Capability labels.
    pub labels: BTreeSet<String>,
}

impl WorkerCapabilities {
    /// Creates capabilities from labels.
    #[must_use]
    pub fn new(labels: impl IntoIterator<Item = String>) -> Self {
        Self {
            cpu_cores: 1,
            memory_mib: 512,
            labels: labels.into_iter().collect(),
        }
    }

    /// Returns whether all required labels are available.
    #[must_use]
    pub fn satisfies(&self, required: &BTreeSet<String>) -> bool {
        required.is_subset(&self.labels)
    }
}

/// Network-disabled job sandbox policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxPolicy {
    /// Enabled sandbox controls.
    pub controls: BTreeSet<SandboxControl>,
}

/// Individual sandbox control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SandboxControl {
    /// Jobs run as a non-root user.
    NonRoot,
    /// Networking is disabled.
    NetworkDisabled,
    /// Host environment variables are denied.
    DenyHostEnv,
    /// Writes outside ephemeral job storage are denied.
    DenyWriteFilesystem,
    /// Docker socket mounts are denied.
    DenyDockerSocket,
    /// Artifacts outside the job manifest are denied.
    DenyUnrelatedArtifacts,
    /// Artifacts are mounted read-only.
    ReadOnlyArtifacts,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            controls: BTreeSet::from([
                SandboxControl::NonRoot,
                SandboxControl::NetworkDisabled,
                SandboxControl::DenyHostEnv,
                SandboxControl::DenyWriteFilesystem,
                SandboxControl::DenyDockerSocket,
                SandboxControl::DenyUnrelatedArtifacts,
                SandboxControl::ReadOnlyArtifacts,
            ]),
        }
    }
}

impl SandboxPolicy {
    /// Returns whether a control is enabled.
    #[must_use]
    pub fn has(&self, control: SandboxControl) -> bool {
        self.controls.contains(&control)
    }

    /// Returns whether a requested artifact is present in the job manifest.
    #[must_use]
    pub fn allows_artifact(&self, requested: &str, manifest_artifacts: &[&str]) -> bool {
        manifest_artifacts.contains(&requested)
    }
}

/// Worker registration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerRegistration {
    /// Worker id.
    pub worker_id: String,
    /// Certificate fingerprint.
    pub certificate_fingerprint: String,
    /// Capabilities.
    pub capabilities: WorkerCapabilities,
}
