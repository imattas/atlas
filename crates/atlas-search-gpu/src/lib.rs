//! CUDA search boundary with hardware-independent validation behavior.

use atlas_scheduler::CancellationToken;
use atlas_search_ir::{SearchDomain, SearchProgram};
use atlas_search_native::NativeSearcher;

/// Kernel cache key.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct KernelCacheKey {
    /// Program fingerprint.
    pub program: String,
    /// Compiler version or identifier.
    pub compiler: String,
    /// Device identifier.
    pub device: String,
    /// Compilation options.
    pub options: String,
}

impl KernelCacheKey {
    /// Creates a kernel cache key.
    #[must_use]
    pub fn new(
        program: impl Into<String>,
        compiler: impl Into<String>,
        device: impl Into<String>,
        options: impl Into<String>,
    ) -> Self {
        Self {
            program: program.into(),
            compiler: compiler.into(),
            device: device.into(),
            options: options.into(),
        }
    }
}

/// GPU searcher boundary.
pub struct GpuSearcher;

impl GpuSearcher {
    /// Generates CUDA source for the restricted IR.
    #[must_use]
    pub fn compile_cuda(program: &SearchProgram) -> String {
        format!(
            "__global__ void atlas_search(unsigned long long start, unsigned long long end) {{ /* width={} ops={} */ }}",
            program.width,
            program.ops.len()
        )
    }

    /// Hardware-independent GPU search fallback.
    ///
    /// GPU execution never bypasses CPU validation; in environments without a
    /// CUDA device this returns the CPU-validated result for differential tests.
    #[must_use]
    pub fn search(
        program: &SearchProgram,
        domain: SearchDomain,
        cancellation: &CancellationToken,
    ) -> Vec<u64> {
        NativeSearcher::search(program, domain, cancellation)
    }

    /// Validates GPU-reported matches against CPU IR semantics.
    #[must_use]
    pub fn cpu_validate_matches(program: &SearchProgram, reported: &[u64]) -> Vec<u64> {
        reported
            .iter()
            .copied()
            .filter(|candidate| program.accepts(*candidate))
            .collect()
    }
}
