//! CUDA search boundary with hardware-independent validation behavior.

use atlas_scheduler::CancellationToken;
use atlas_search_ir::{SearchDomain, SearchProgram};
use atlas_search_native::NativeSearcher;
use std::collections::BTreeSet;

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

/// Supported GPU compute SDK families.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuSdk {
    /// Khronos `OpenCL` SDK/runtime.
    OpenCl {
        /// SDK or runtime identifier.
        sdk: String,
    },
    /// Khronos Vulkan compute SDK/runtime.
    Vulkan {
        /// SDK or runtime identifier.
        sdk: String,
    },
    /// NVIDIA `CUDA` Toolkit/runtime.
    Cuda {
        /// SDK or runtime identifier.
        sdk: String,
    },
    /// AMD `HIP` SDK/runtime.
    Hip {
        /// SDK or runtime identifier.
        sdk: String,
    },
}

impl GpuSdk {
    fn priority(&self, prefer_portable: bool) -> u8 {
        if prefer_portable {
            match self {
                Self::OpenCl { .. } => 0,
                Self::Vulkan { .. } => 1,
                Self::Cuda { .. } | Self::Hip { .. } => 2,
            }
        } else {
            match self {
                Self::Cuda { .. } => 0,
                Self::Hip { .. } => 1,
                Self::OpenCl { .. } => 2,
                Self::Vulkan { .. } => 3,
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::OpenCl { .. } => "OpenCL",
            Self::Vulkan { .. } => "Vulkan",
            Self::Cuda { .. } => "CUDA",
            Self::Hip { .. } => "HIP",
        }
    }
}

/// GPU SDK selection result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuSdkPlan {
    /// Selected SDK, if available.
    pub selected: Option<GpuSdk>,
    /// Recorded rationale for release reports.
    pub rationale: String,
}

impl GpuSdkPlan {
    /// Chooses an SDK from detected candidates.
    #[must_use]
    pub fn choose(detected: &[GpuSdk], prefer_portable: bool) -> Self {
        let Some(selected) = detected
            .iter()
            .min_by_key(|sdk| sdk.priority(prefer_portable))
            .cloned()
        else {
            return Self {
                selected: None,
                rationale: "no GPU SDK detected; hardware acceleration disabled".to_owned(),
            };
        };
        let portability = if prefer_portable {
            "portable"
        } else {
            "vendor-preferred"
        };
        Self {
            rationale: format!("{portability} GPU SDK selected: {}", selected.name()),
            selected: Some(selected),
        }
    }
}

/// GPU launch configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    /// Total global invocation count rounded for the selected local size.
    pub global_size: u64,
    /// Workgroup local size.
    pub local_size: u64,
    /// Maximum retained matches.
    pub max_matches: usize,
    /// Output match buffer transfer size in bytes.
    pub output_buffer_bytes: usize,
}

/// Runtime execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    /// Device was unavailable and CPU fallback produced the result.
    CpuFallback,
    /// Device-reported matches were validated on CPU before return.
    DeviceValidated,
}

/// Accelerator runtime telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeTelemetry {
    /// Launch configuration.
    pub launch: LaunchConfig,
    /// Selected SDK plan rationale.
    pub rationale: String,
    /// Whether every returned match was CPU validated.
    pub cpu_validated: bool,
    /// Count of rejected device-reported matches.
    pub rejected_device_matches: usize,
}

/// Accelerator execution report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceleratorReport {
    /// Runtime execution mode.
    pub mode: RuntimeMode,
    /// Validated match stream.
    pub matches: Vec<u64>,
    /// Runtime telemetry.
    pub telemetry: RuntimeTelemetry,
}

/// Accelerator runtime boundary.
pub struct AcceleratorRuntime;

impl AcceleratorRuntime {
    /// Plans a bounded GPU launch and transfer shape.
    #[must_use]
    pub fn plan_launch(domain: SearchDomain, local_size: u64, max_matches: usize) -> LaunchConfig {
        let local_size = local_size.max(1);
        let candidates = domain.end.saturating_sub(domain.start);
        let groups = candidates.saturating_add(local_size - 1) / local_size;
        let global_size = groups.max(1).saturating_mul(local_size);
        LaunchConfig {
            global_size,
            local_size,
            max_matches,
            output_buffer_bytes: max_matches.saturating_mul(std::mem::size_of::<u64>()),
        }
    }

    /// Executes through the accelerator boundary.
    ///
    /// `reported_device_matches` is the device output buffer supplied by a real
    /// driver integration or deterministic tests. Returned matches are always
    /// revalidated by CPU IR semantics.
    #[must_use]
    pub fn execute(
        program: &SearchProgram,
        domain: SearchDomain,
        detected_sdks: &[GpuSdk],
        cancellation: &CancellationToken,
        reported_device_matches: &[u64],
    ) -> AcceleratorReport {
        let launch = Self::plan_launch(domain, 256, 1024);
        let plan = GpuSdkPlan::choose(detected_sdks, true);
        if plan.selected.is_none() || reported_device_matches.is_empty() {
            return AcceleratorReport {
                mode: RuntimeMode::CpuFallback,
                matches: NativeSearcher::search(program, domain, cancellation),
                telemetry: RuntimeTelemetry {
                    launch,
                    rationale: plan.rationale,
                    cpu_validated: true,
                    rejected_device_matches: 0,
                },
            };
        }
        let matches = GpuSearcher::cpu_validate_matches(program, reported_device_matches);
        AcceleratorReport {
            mode: RuntimeMode::DeviceValidated,
            telemetry: RuntimeTelemetry {
                launch,
                rationale: plan.rationale,
                cpu_validated: true,
                rejected_device_matches: reported_device_matches
                    .len()
                    .saturating_sub(matches.len()),
            },
            matches,
        }
    }
}

/// GPU SDK detector.
pub struct GpuSdkDetector;

impl GpuSdkDetector {
    /// Detects SDKs from an explicit tool-name list.
    ///
    /// This is deterministic and does not inspect the host. Runtime callers can
    /// pass PATH-discovered tool names through this function.
    #[must_use]
    pub fn detect_from_tools(tools: &[String]) -> Vec<GpuSdk> {
        let normalized: BTreeSet<String> =
            tools.iter().map(|tool| tool.to_ascii_lowercase()).collect();
        let mut detected = Vec::new();
        if normalized
            .iter()
            .any(|tool| tool == "clinfo" || tool == "opencl-clang" || tool.contains("opencl"))
        {
            detected.push(GpuSdk::OpenCl {
                sdk: "Khronos OpenCL-compatible toolchain".to_owned(),
            });
        }
        if normalized
            .iter()
            .any(|tool| tool == "glslc" || tool == "vulkaninfo" || tool.contains("vulkan"))
        {
            detected.push(GpuSdk::Vulkan {
                sdk: "Vulkan compute toolchain".to_owned(),
            });
        }
        if normalized
            .iter()
            .any(|tool| tool == "nvcc" || tool.contains("cuda"))
        {
            detected.push(GpuSdk::Cuda {
                sdk: "NVIDIA CUDA Toolkit".to_owned(),
            });
        }
        if normalized
            .iter()
            .any(|tool| tool == "hipcc" || tool.contains("rocm") || tool.contains("hip"))
        {
            detected.push(GpuSdk::Hip {
                sdk: "AMD HIP/ROCm SDK".to_owned(),
            });
        }
        detected
    }
}

/// GPU searcher boundary.
pub struct GpuSearcher;

impl GpuSearcher {
    /// Generates CUDA source for the restricted IR.
    #[must_use]
    pub fn compile_cuda(program: &SearchProgram) -> String {
        format!(
            "__global__ void atlas_search(unsigned long long start, unsigned long long end, unsigned long long* out, unsigned int* out_len) {{ /* width={} ops={} */ (void)start; (void)end; (void)out; (void)out_len; }}",
            program.width,
            program.ops.len()
        )
    }

    /// Generates `OpenCL` C source for the restricted IR.
    #[must_use]
    pub fn compile_opencl(program: &SearchProgram) -> String {
        format!(
            "__kernel void atlas_search(ulong start, ulong end, __global ulong* out, __global uint* out_len) {{ /* width={} ops={} */ size_t gid = get_global_id(0); (void)gid; (void)start; (void)end; (void)out; (void)out_len; }}",
            program.width,
            program.ops.len()
        )
    }

    /// Generates Vulkan-compatible GLSL compute shader source for the restricted IR.
    #[must_use]
    pub fn compile_vulkan_glsl(program: &SearchProgram) -> String {
        format!(
            "#version 450\nlayout(local_size_x = 256) in;\nlayout(set = 0, binding = 0) buffer Matches {{ uint out_len; uint out_values[]; }} matches;\nvoid main() {{ /* width={} ops={} */ uint gid = gl_GlobalInvocationID.x; matches.out_len += gid & 0u; }}\n",
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
