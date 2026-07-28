//! CUDA search boundary with hardware-independent validation behavior.

use atlas_placement::{
    PlacementCalibration, PlacementCapabilities, PlacementModel, PlacementTarget, SearchFeatures,
};
use atlas_scheduler::CancellationToken;
use atlas_search_ir::{SearchDomain, SearchOp, SearchProgram};
use atlas_search_native::NativeSearcher;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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

    fn runtime_identity(&self) -> &str {
        match self {
            Self::OpenCl { sdk }
            | Self::Vulkan { sdk }
            | Self::Cuda { sdk }
            | Self::Hip { sdk } => sdk,
        }
    }
}

fn sdk_supports_program(sdk: &GpuSdk, program: &SearchProgram) -> bool {
    match sdk {
        GpuSdk::OpenCl { sdk } if program.width > 32 => sdk_has_feature(sdk, "int64"),
        GpuSdk::Vulkan { sdk } if program.width > 32 => sdk_has_feature(sdk, "shaderint64"),
        GpuSdk::Cuda { sdk } if program.width > 32 => sdk_has_feature(sdk, "int64"),
        GpuSdk::Hip { sdk } if program.width > 32 => sdk_has_feature(sdk, "int64"),
        GpuSdk::OpenCl { .. }
        | GpuSdk::Vulkan { .. }
        | GpuSdk::Cuda { .. }
        | GpuSdk::Hip { .. } => true,
    }
}

fn sdk_has_feature(sdk: &str, feature: &str) -> bool {
    let normalized_feature = feature.to_ascii_lowercase();
    let tokens = sdk
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    tokens.iter().enumerate().any(|(index, token)| {
        token == &normalized_feature && !feature_token_is_negated(&tokens, index)
    })
}

fn feature_token_is_negated(tokens: &[String], index: usize) -> bool {
    index > 0 && matches!(tokens[index - 1].as_str(), "no" | "non" | "without")
}

#[cfg(test)]
mod tests {
    use super::{append_sdk_feature, sdk_has_feature, GpuSdk};

    #[test]
    fn adapter_feature_append_overrides_negated_identity_text() {
        let mut sdk = GpuSdk::OpenCl {
            sdk: "OpenCL runtime no-int64".to_owned(),
        };

        append_sdk_feature(&mut sdk, "int64");

        let GpuSdk::OpenCl { sdk } = sdk else {
            panic!("expected OpenCL SDK");
        };
        assert!(sdk.ends_with(" int64"));
        assert!(sdk_has_feature(&sdk, "int64"));
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

    /// Chooses an SDK that can execute the supplied search program.
    #[must_use]
    pub fn choose_for_program(
        detected: &[GpuSdk],
        prefer_portable: bool,
        program: &SearchProgram,
    ) -> Self {
        let compatible = detected
            .iter()
            .filter(|sdk| sdk_supports_program(sdk, program))
            .cloned()
            .collect::<Vec<_>>();
        if compatible.is_empty() && !detected.is_empty() {
            return Self {
                selected: None,
                rationale: "no compatible GPU SDK detected for search program".to_owned(),
            };
        }
        Self::choose(&compatible, prefer_portable)
    }
}

/// SDK-specific driver command plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverCommandPlan {
    /// SDK selected for this command plan.
    pub sdk: GpuSdk,
    /// Checked-in kernel template artifact path.
    pub template_file: String,
    /// Generated kernel source path used by the compiler command.
    pub source_file: String,
    /// Generated kernel source text.
    pub kernel_source: String,
    /// Compiled kernel artifact path.
    pub artifact_file: String,
    /// Command used to compile the kernel artifact.
    pub compile_command: Vec<String>,
    /// Command used by a host-side driver adapter to launch the kernel.
    pub launch_command: Vec<String>,
    /// Kernel cache key for the compiled artifact.
    pub cache_key: KernelCacheKey,
}

impl DriverCommandPlan {
    /// Builds a deterministic driver command plan for an SDK.
    #[must_use]
    pub fn for_sdk(sdk: &GpuSdk, program: &SearchProgram, output_dir: &str) -> Self {
        Self::for_launch(
            sdk,
            program,
            SearchDomain::new(0, 0),
            LaunchConfig {
                global_size: 1,
                local_size: 1,
                max_matches: 0,
                output_buffer_bytes: 0,
            },
            output_dir,
        )
    }

    /// Builds a deterministic driver command plan for one bounded launch.
    #[must_use]
    pub fn for_launch(
        sdk: &GpuSdk,
        program: &SearchProgram,
        domain: SearchDomain,
        launch: LaunchConfig,
        output_dir: &str,
    ) -> Self {
        let (template_file, source_name, artifact_name, compiler, options, kernel_source) =
            match sdk {
                GpuSdk::OpenCl { .. } => (
                    "gpu/opencl/atlas_search.cl",
                    "atlas_search.cl",
                    "atlas_search.opencl.bin",
                    "atlas-gpu-opencl-run",
                    "--compile-check",
                    GpuSearcher::compile_opencl(program),
                ),
                GpuSdk::Vulkan { .. } => (
                    "gpu/vulkan/atlas_search.comp",
                    "atlas_search.comp",
                    "atlas_search.spv",
                    "atlas-gpu-vulkan-run",
                    "--compile-check",
                    GpuSearcher::compile_vulkan_glsl(program),
                ),
                GpuSdk::Cuda { .. } => (
                    "gpu/cuda/atlas_search.cu",
                    "atlas_search.cu",
                    "atlas_search.ptx",
                    "atlas-gpu-cuda-run",
                    "--compile-check",
                    GpuSearcher::compile_cuda(program),
                ),
                GpuSdk::Hip { .. } => (
                    "gpu/hip/atlas_search.hip",
                    "atlas_search.hip",
                    "atlas_search.hsaco",
                    "atlas-gpu-hip-run",
                    "--compile-check",
                    GpuSearcher::compile_hip(program),
                ),
            };
        let cache_key = KernelCacheKey::new(
            format!(
                "{program:?};kernel_source={:016x}",
                stable_text_hash(&kernel_source)
            ),
            compiler,
            sdk.runtime_identity(),
            options,
        );
        let cache_dir = join_path(output_dir, &kernel_cache_id(&cache_key));
        let source_file = join_path(&cache_dir, source_name);
        let artifact_file = join_path(&cache_dir, artifact_name);
        let compile_command = compile_command_for(compiler, options, &source_file, &artifact_file);
        let launch_input = match sdk {
            GpuSdk::OpenCl { .. }
            | GpuSdk::Vulkan { .. }
            | GpuSdk::Cuda { .. }
            | GpuSdk::Hip { .. } => artifact_file.clone(),
        };
        let launch_command = vec![
            format!("atlas-gpu-{}-run", sdk.name().to_ascii_lowercase()),
            launch_input,
            "--start".to_owned(),
            domain.start.to_string(),
            "--end".to_owned(),
            domain.end.to_string(),
            "--max-matches".to_owned(),
            launch.max_matches.to_string(),
            "--global-size".to_owned(),
            launch.global_size.to_string(),
            "--local-size".to_owned(),
            launch.local_size.to_string(),
        ];
        Self {
            sdk: sdk.clone(),
            template_file: template_file.to_owned(),
            source_file,
            kernel_source,
            artifact_file,
            compile_command,
            launch_command,
            cache_key,
        }
    }
}

/// Driver execution output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverRunOutput {
    /// Process-style exit code.
    pub exit_code: i32,
    /// Matches reported by the device output buffer.
    pub reported_matches: Vec<u64>,
    /// Captured standard output from the driver adapter.
    pub stdout: String,
    /// Captured standard error from the driver adapter.
    pub stderr: String,
}

impl DriverRunOutput {
    /// Parses newline-separated device matches from launcher output.
    ///
    /// Accepted tokens are decimal values, hexadecimal values prefixed by
    /// `0x`, or `match=<value>` lines using either numeric form.
    #[must_use]
    pub fn parse_reported_matches(stdout: &str) -> Vec<u64> {
        stdout
            .lines()
            .filter_map(|line| parse_match_token(line.trim()))
            .collect()
    }
}

/// Driver runner abstraction for SDK command execution.
pub trait DriverRunner {
    /// Runs the compile and launch plan and returns device-reported output.
    fn run(&self, plan: &DriverCommandPlan) -> DriverRunOutput;
}

/// Process command runner abstraction.
pub trait CommandRunner {
    /// Runs one command and returns process-style output.
    fn run_command(&self, command: &[String]) -> DriverRunOutput;
}

/// Process-backed driver runner.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessDriverRunner;

impl DriverRunner for ProcessDriverRunner {
    fn run(&self, plan: &DriverCommandPlan) -> DriverRunOutput {
        Self::run_with_command_runner(plan, self)
    }
}

impl CommandRunner for ProcessDriverRunner {
    fn run_command(&self, command: &[String]) -> DriverRunOutput {
        run_command(command)
    }
}

impl ProcessDriverRunner {
    /// Writes generated source, runs compile command, then runs launch command.
    #[must_use]
    pub fn run_with_command_runner(
        plan: &DriverCommandPlan,
        runner: &dyn CommandRunner,
    ) -> DriverRunOutput {
        if let Err(error) = write_generated_source(plan) {
            return DriverRunOutput {
                exit_code: 127,
                reported_matches: Vec::new(),
                stdout: String::new(),
                stderr: error,
            };
        }
        let compile = if can_reuse_compiled_artifact(plan) {
            DriverRunOutput {
                exit_code: 0,
                reported_matches: Vec::new(),
                stdout: String::new(),
                stderr: String::new(),
            }
        } else {
            let compile = runner.run_command(&plan.compile_command);
            if compile.exit_code != 0 {
                return compile;
            }
            compile
        };
        let mut launch = runner.run_command(&plan.launch_command);
        launch.stdout = [compile.stdout, launch.stdout].join("");
        launch.stderr = [compile.stderr, launch.stderr].join("");
        launch
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

/// Runtime execution policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RuntimePolicy {
    /// Launch the selected GPU driver even when placement would normally choose
    /// scalar CPU execution for the workload shape.
    pub force_gpu: bool,
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

const DEFAULT_GPU_LOCAL_SIZE: u64 = 256;
const MAX_DRIVER_LAUNCH_CANDIDATES: u64 = u32::MAX as u64;

impl AcceleratorRuntime {
    /// Plans a bounded GPU launch and transfer shape.
    #[must_use]
    pub fn plan_launch(domain: SearchDomain, local_size: u64, max_matches: usize) -> LaunchConfig {
        let local_size = local_size.max(1);
        let candidates = domain.end.saturating_sub(domain.start);
        let groups = candidates.saturating_add(local_size - 1) / local_size;
        let mut global_size = groups.max(1).saturating_mul(local_size);
        let mut local_size = local_size;
        if global_size < candidates {
            local_size = 1;
            global_size = candidates.max(1);
        }
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
        if cancellation.is_cancelled() {
            return Self::cancelled_report(program, domain, cancellation);
        }
        let launch = Self::plan_launch(domain, 256, 1024);
        let plan = GpuSdkPlan::choose_for_program(detected_sdks, true, program);
        if reported_device_matches.is_empty() {
            let cached_kernel_keys =
                persisted_kernel_cache_keys(program, domain, detected_sdks, "target/atlas-gpu");
            return Self::execute_with_detected_driver_and_explicit_cache(
                program,
                domain,
                detected_sdks,
                cancellation,
                &cached_kernel_keys,
                &ProcessDriverRunner,
            );
        }
        if plan.selected.is_none() {
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
        let validation =
            validate_device_matches(program, domain, reported_device_matches, launch.max_matches);
        let matches = validation.matches;
        if matches.is_empty() {
            return AcceleratorReport {
                mode: RuntimeMode::CpuFallback,
                matches: NativeSearcher::search(program, domain, cancellation),
                telemetry: RuntimeTelemetry {
                    launch,
                    rationale: format!(
                        "{}; no valid device matches after CPU validation",
                        plan.rationale
                    ),
                    cpu_validated: true,
                    rejected_device_matches: validation.rejected,
                },
            };
        }
        AcceleratorReport {
            mode: RuntimeMode::DeviceValidated,
            telemetry: RuntimeTelemetry {
                launch,
                rationale: plan.rationale,
                cpu_validated: true,
                rejected_device_matches: validation.rejected,
            },
            matches,
        }
    }

    /// Selects the best detected SDK, executes through the supplied GPU driver
    /// runner, and validates device-reported matches.
    #[must_use]
    pub fn execute_with_detected_driver(
        program: &SearchProgram,
        domain: SearchDomain,
        detected_sdks: &[GpuSdk],
        cancellation: &CancellationToken,
        runner: &dyn DriverRunner,
    ) -> AcceleratorReport {
        Self::execute_with_detected_driver_and_explicit_cache(
            program,
            domain,
            detected_sdks,
            cancellation,
            &[],
            runner,
        )
    }

    /// Selects the best detected SDK, accounts for warmed kernel cache keys,
    /// executes through the supplied GPU driver runner, and validates
    /// device-reported matches.
    #[must_use]
    pub fn execute_with_detected_driver_and_kernel_cache(
        program: &SearchProgram,
        domain: SearchDomain,
        detected_sdks: &[GpuSdk],
        cancellation: &CancellationToken,
        cached_kernel_keys: &[KernelCacheKey],
        runner: &dyn DriverRunner,
    ) -> AcceleratorReport {
        Self::execute_with_detected_driver_and_explicit_cache(
            program,
            domain,
            detected_sdks,
            cancellation,
            cached_kernel_keys,
            runner,
        )
    }

    fn execute_with_detected_driver_and_explicit_cache(
        program: &SearchProgram,
        domain: SearchDomain,
        detected_sdks: &[GpuSdk],
        cancellation: &CancellationToken,
        cached_kernel_keys: &[KernelCacheKey],
        runner: &dyn DriverRunner,
    ) -> AcceleratorReport {
        Self::execute_with_detected_driver_and_policy(
            program,
            domain,
            detected_sdks,
            cancellation,
            RuntimePolicy::default(),
            cached_kernel_keys,
            runner,
        )
    }

    /// Selects the best detected SDK, applies an execution policy, accounts for
    /// warmed kernel cache keys, executes through the supplied GPU driver
    /// runner, and validates device-reported matches.
    #[must_use]
    pub fn execute_with_detected_driver_and_policy(
        program: &SearchProgram,
        domain: SearchDomain,
        detected_sdks: &[GpuSdk],
        cancellation: &CancellationToken,
        policy: RuntimePolicy,
        cached_kernel_keys: &[KernelCacheKey],
        runner: &dyn DriverRunner,
    ) -> AcceleratorReport {
        if cancellation.is_cancelled() {
            return Self::cancelled_report(program, domain, cancellation);
        }
        let plan = GpuSdkPlan::choose_for_program(detected_sdks, true, program);
        let Some(selected) = plan.selected else {
            let launch = Self::plan_launch(domain, 256, 1024);
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
        };
        let launch = Self::plan_launch(domain, 256, 1024);
        let driver_plan =
            DriverCommandPlan::for_launch(&selected, program, domain, launch, "target/atlas-gpu");
        let kernel_cache_hit = cached_kernel_keys.contains(&driver_plan.cache_key);
        if !policy.force_gpu {
            let placement = PlacementModel::choose_with_calibration(
                SearchFeatures {
                    candidates: domain.end.saturating_sub(domain.start),
                    regular: true,
                    kernel_cache_hit,
                },
                PlacementCapabilities {
                    scalar: true,
                    simd: false,
                    gpu: true,
                },
                PlacementCalibration::default(),
            );
            if placement.target != PlacementTarget::Gpu {
                return AcceleratorReport {
                    mode: RuntimeMode::CpuFallback,
                    matches: NativeSearcher::search(program, domain, cancellation),
                    telemetry: RuntimeTelemetry {
                        launch,
                        rationale: format!(
                            "{:?} placement selected: {}; {}",
                            placement.target,
                            placement.rationale,
                            selected.name()
                        ),
                        cpu_validated: true,
                        rejected_device_matches: 0,
                    },
                };
            }
        }
        Self::execute_with_driver(program, domain, &selected, cancellation, runner)
    }

    /// Detects SDKs from supplied PATH directories, executes through the best
    /// available GPU driver runner, and validates device-reported matches.
    #[must_use]
    pub fn execute_with_path_detected_driver(
        program: &SearchProgram,
        domain: SearchDomain,
        path_dirs: impl IntoIterator<Item = PathBuf>,
        cancellation: &CancellationToken,
        runner: &dyn DriverRunner,
    ) -> AcceleratorReport {
        let detected_sdks = GpuSdkDetector::detect_from_path_dirs(path_dirs);
        let cached_kernel_keys =
            persisted_kernel_cache_keys(program, domain, &detected_sdks, "target/atlas-gpu");
        Self::execute_with_detected_driver_and_explicit_cache(
            program,
            domain,
            &detected_sdks,
            cancellation,
            &cached_kernel_keys,
            runner,
        )
    }

    /// Detects SDKs from the host `PATH`, executes through the process-backed
    /// GPU driver, and validates device-reported matches.
    #[must_use]
    pub fn execute_with_host_driver(
        program: &SearchProgram,
        domain: SearchDomain,
        cancellation: &CancellationToken,
    ) -> AcceleratorReport {
        let detected_sdks =
            GpuSdkDetector::detect_from_host_path_with_adapter_features(&ProcessDriverRunner);
        let cached_kernel_keys =
            persisted_kernel_cache_keys(program, domain, &detected_sdks, "target/atlas-gpu");
        Self::execute_with_detected_driver_and_explicit_cache(
            program,
            domain,
            &detected_sdks,
            cancellation,
            &cached_kernel_keys,
            &ProcessDriverRunner,
        )
    }

    /// Executes through a GPU driver runner and validates device-reported matches.
    #[must_use]
    pub fn execute_with_driver(
        program: &SearchProgram,
        domain: SearchDomain,
        sdk: &GpuSdk,
        cancellation: &CancellationToken,
        runner: &dyn DriverRunner,
    ) -> AcceleratorReport {
        if cancellation.is_cancelled() {
            return Self::cancelled_report(program, domain, cancellation);
        }
        let launch = Self::plan_launch(domain, DEFAULT_GPU_LOCAL_SIZE, 1024);
        let launch_domains = driver_launch_domains(domain);
        let mut reported_matches = Vec::new();
        for launch_domain in launch_domains.iter().copied() {
            if cancellation.is_cancelled() {
                return Self::cancelled_report(program, domain, cancellation);
            }
            let chunk_launch = Self::plan_launch(launch_domain, DEFAULT_GPU_LOCAL_SIZE, 1024);
            let command_plan = DriverCommandPlan::for_launch(
                sdk,
                program,
                launch_domain,
                chunk_launch,
                "target/atlas-gpu",
            );
            let output = runner.run(&command_plan);
            if output.exit_code != 0 {
                let base_rationale = driver_failure_rationale(sdk, &output);
                return AcceleratorReport {
                    mode: RuntimeMode::CpuFallback,
                    matches: NativeSearcher::search(program, domain, cancellation),
                    telemetry: RuntimeTelemetry {
                        launch,
                        rationale: base_rationale,
                        cpu_validated: true,
                        rejected_device_matches: 0,
                    },
                };
            }
            reported_matches.extend(output.reported_matches);
        }
        let base_rationale = format!(
            "{}; driver exit 0; driver launches {}",
            sdk.name(),
            launch_domains.len()
        );
        if reported_matches.is_empty() {
            return AcceleratorReport {
                mode: RuntimeMode::CpuFallback,
                matches: NativeSearcher::search(program, domain, cancellation),
                telemetry: RuntimeTelemetry {
                    launch,
                    rationale: base_rationale,
                    cpu_validated: true,
                    rejected_device_matches: 0,
                },
            };
        }
        let validation =
            validate_device_matches(program, domain, &reported_matches, launch.max_matches);
        let matches = validation.matches;
        if matches.is_empty() {
            return AcceleratorReport {
                mode: RuntimeMode::CpuFallback,
                matches: NativeSearcher::search(program, domain, cancellation),
                telemetry: RuntimeTelemetry {
                    launch,
                    rationale: format!("{base_rationale}; no valid device matches"),
                    cpu_validated: true,
                    rejected_device_matches: validation.rejected,
                },
            };
        }
        AcceleratorReport {
            mode: RuntimeMode::DeviceValidated,
            telemetry: RuntimeTelemetry {
                launch,
                rationale: base_rationale,
                cpu_validated: true,
                rejected_device_matches: validation.rejected,
            },
            matches,
        }
    }

    fn cancelled_report(
        program: &SearchProgram,
        domain: SearchDomain,
        cancellation: &CancellationToken,
    ) -> AcceleratorReport {
        AcceleratorReport {
            mode: RuntimeMode::CpuFallback,
            matches: NativeSearcher::search(program, domain, cancellation),
            telemetry: RuntimeTelemetry {
                launch: Self::plan_launch(domain, 256, 1024),
                rationale: "cancelled before GPU driver launch".to_owned(),
                cpu_validated: true,
                rejected_device_matches: 0,
            },
        }
    }
}

fn compile_command_for(
    compiler: &str,
    options: &str,
    source_file: &str,
    artifact_file: &str,
) -> Vec<String> {
    let mut command = vec![compiler.to_owned()];
    command.extend(options.split_whitespace().map(str::to_owned));
    command.push(source_file.to_owned());
    command.push("-o".to_owned());
    command.push(artifact_file.to_owned());
    command
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeviceValidation {
    matches: Vec<u64>,
    rejected: usize,
}

fn driver_failure_rationale(sdk: &GpuSdk, output: &DriverRunOutput) -> String {
    let stderr = output.stderr.trim();
    if stderr.is_empty() {
        format!("{}; driver exit {}", sdk.name(), output.exit_code)
    } else {
        format!(
            "{}; driver exit {}; stderr: {}",
            sdk.name(),
            output.exit_code,
            single_line_summary(stderr)
        )
    }
}

fn single_line_summary(text: &str) -> String {
    const MAX_SUMMARY_CHARS: usize = 240;

    let summary = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if summary.chars().count() <= MAX_SUMMARY_CHARS {
        summary
    } else {
        let mut truncated = summary.chars().take(MAX_SUMMARY_CHARS).collect::<String>();
        truncated.push('…');
        truncated
    }
}

fn validate_device_matches(
    program: &SearchProgram,
    domain: SearchDomain,
    reported: &[u64],
    max_matches: usize,
) -> DeviceValidation {
    let mut rejected = 0;
    let mut matches = Vec::new();
    for candidate in reported.iter().copied() {
        if candidate >= domain.start && candidate < domain.end && program.accepts(candidate) {
            matches.push(candidate);
        } else {
            rejected += 1;
        }
    }
    matches.sort_unstable();
    matches.dedup();
    if matches.len() > max_matches {
        rejected += matches.len() - max_matches;
        matches.truncate(max_matches);
    }
    DeviceValidation { matches, rejected }
}

fn driver_launch_domains(domain: SearchDomain) -> Vec<SearchDomain> {
    let mut domains = Vec::new();
    let mut start = domain.start;
    while start < domain.end {
        let remaining = domain.end - start;
        let chunk_len = remaining.min(MAX_DRIVER_LAUNCH_CANDIDATES);
        let end = start + chunk_len;
        domains.push(SearchDomain::new(start, end));
        start = end;
    }
    domains
}

fn join_path(output_dir: &str, artifact_name: &str) -> String {
    let output_dir = output_dir.trim_end_matches(['/', '\\']);
    format!("{output_dir}/{artifact_name}")
}

fn kernel_cache_id(key: &KernelCacheKey) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for field in [&key.program, &key.compiler, &key.device, &key.options] {
        for byte in field.as_bytes().iter().copied().chain([0]) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

fn stable_text_hash(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn write_generated_source(plan: &DriverCommandPlan) -> Result<(), String> {
    let source_path = Path::new(&plan.source_file);
    if let Some(parent) = source_path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(source_path, &plan.kernel_source).map_err(|error| error.to_string())
}

fn can_reuse_compiled_artifact(plan: &DriverCommandPlan) -> bool {
    plan.launch_command
        .get(1)
        .is_some_and(|launch_input| launch_input == &plan.artifact_file)
        && Path::new(&plan.artifact_file).is_file()
}

fn persisted_kernel_cache_keys(
    program: &SearchProgram,
    domain: SearchDomain,
    detected_sdks: &[GpuSdk],
    output_dir: &str,
) -> Vec<KernelCacheKey> {
    let launch = AcceleratorRuntime::plan_launch(domain, 256, 1024);
    detected_sdks
        .iter()
        .filter_map(|sdk| {
            let plan = DriverCommandPlan::for_launch(sdk, program, domain, launch, output_dir);
            Path::new(&plan.artifact_file)
                .is_file()
                .then_some(plan.cache_key)
        })
        .collect()
}

fn run_command(command: &[String]) -> DriverRunOutput {
    let Some((program, args)) = command.split_first() else {
        return DriverRunOutput {
            exit_code: 127,
            reported_matches: Vec::new(),
            stdout: String::new(),
            stderr: "empty driver command".to_owned(),
        };
    };
    if let Some(output) = run_resolved_sdk_tool_command(program, args) {
        return output;
    }
    match Command::new(program).args(args).output() {
        Ok(output) => DriverRunOutput {
            exit_code: output.status.code().unwrap_or(1),
            reported_matches: DriverRunOutput::parse_reported_matches(&String::from_utf8_lossy(
                &output.stdout,
            )),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        },
        Err(error) => run_resolved_command(program, args).unwrap_or_else(|| DriverRunOutput {
            exit_code: 127,
            reported_matches: Vec::new(),
            stdout: String::new(),
            stderr: error.to_string(),
        }),
    }
}

fn run_resolved_command(program: &str, args: &[String]) -> Option<DriverRunOutput> {
    let program_path = resolve_adjacent_adapter_program(program)?;
    let output = Command::new(program_path).args(args).output().ok()?;
    Some(DriverRunOutput {
        exit_code: output.status.code().unwrap_or(1),
        reported_matches: DriverRunOutput::parse_reported_matches(&String::from_utf8_lossy(
            &output.stdout,
        )),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

fn run_resolved_sdk_tool_command(program: &str, args: &[String]) -> Option<DriverRunOutput> {
    let program_path = resolve_sdk_tool_program(program)?;
    let output = Command::new(program_path).args(args).output().ok()?;
    Some(DriverRunOutput {
        exit_code: output.status.code().unwrap_or(1),
        reported_matches: DriverRunOutput::parse_reported_matches(&String::from_utf8_lossy(
            &output.stdout,
        )),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

fn resolve_adjacent_adapter_program(program: &str) -> Option<PathBuf> {
    let plain_name = Path::new(program).file_name()?.to_str()?;
    if plain_name != program || !plain_name.starts_with("atlas-gpu-") {
        return None;
    }
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let mut dirs = vec![exe_dir.clone()];
    if let Some(parent) = exe_dir.parent() {
        dirs.push(parent.to_path_buf());
    }
    dirs.into_iter()
        .flat_map(|dir| adapter_program_candidates(&dir, plain_name))
        .find(|candidate| candidate.is_file())
}

fn adapter_program_candidates(dir: &Path, plain_name: &str) -> Vec<PathBuf> {
    let mut candidates = vec![dir.join(plain_name)];
    #[cfg(windows)]
    {
        candidates.push(dir.join(format!("{plain_name}.exe")));
        candidates.push(dir.join(format!("{plain_name}.cmd")));
        candidates.push(dir.join(format!("{plain_name}.bat")));
    }
    candidates
}

fn resolve_sdk_tool_program(program: &str) -> Option<PathBuf> {
    let plain_name = Path::new(program).file_name()?.to_str()?;
    if plain_name != program || !is_known_sdk_tool(plain_name) {
        return None;
    }
    sdk_root_dirs()
        .into_iter()
        .flat_map(|root| {
            [root.clone(), root.join("bin")]
                .into_iter()
                .flat_map(move |dir| sdk_tool_candidates(&dir, plain_name))
        })
        .find(|candidate| candidate.is_file())
}

fn is_known_sdk_tool(name: &str) -> bool {
    matches!(name.to_ascii_lowercase().as_str(), "hipcc" | "nvcc")
}

fn sdk_root_dirs() -> Vec<PathBuf> {
    let mut roots = [
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDA_ROOT",
        "HIP_PATH",
        "ROCM_PATH",
        "ROCM_HOME",
    ]
    .into_iter()
    .filter_map(std::env::var_os)
    .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
    .collect::<Vec<_>>();
    roots.extend(standard_sdk_root_dirs());
    roots
}

fn sdk_tool_candidates(dir: &Path, plain_name: &str) -> Vec<PathBuf> {
    let mut candidates = vec![dir.join(plain_name)];
    #[cfg(windows)]
    {
        candidates.push(dir.join(format!("{plain_name}.exe")));
        candidates.push(dir.join(format!("{plain_name}.cmd")));
        candidates.push(dir.join(format!("{plain_name}.bat")));
    }
    candidates
}

fn parse_match_token(token: &str) -> Option<u64> {
    let token = token.strip_prefix("match=").unwrap_or(token);
    if let Some(hex) = token.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).ok()
    } else {
        token.parse().ok()
    }
}

/// GPU SDK detector.
pub struct GpuSdkDetector;

impl GpuSdkDetector {
    /// Detects SDKs by scanning executable names in the current host `PATH`.
    #[must_use]
    pub fn detect_from_host_path() -> Vec<GpuSdk> {
        let mut paths = std::env::var_os("PATH")
            .map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
            .unwrap_or_default();
        paths.extend(
            [
                "CUDA_PATH",
                "CUDA_HOME",
                "CUDA_ROOT",
                "HIP_PATH",
                "ROCM_PATH",
                "ROCM_HOME",
                "VULKAN_SDK",
                "VK_SDK_PATH",
                "OPENCL_SDK",
                "OCL_ROOT",
                "INTELOCLSDKROOT",
                "AMDAPPSDKROOT",
            ]
            .into_iter()
            .filter_map(std::env::var_os)
            .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>()),
        );
        paths.extend(standard_sdk_root_dirs());
        Self::detect_from_path_dirs(paths)
    }

    /// Detects SDKs from the host path and augments adapter-backed runtime
    /// capabilities by querying checked-in adapter CLIs.
    #[must_use]
    pub fn detect_from_host_path_with_adapter_features(runner: &dyn CommandRunner) -> Vec<GpuSdk> {
        let mut detected = Self::detect_from_host_path();
        augment_with_adapter_features(&mut detected, runner);
        detect_from_adapter_features(&mut detected, runner);
        detected
    }

    /// Detects SDKs by scanning executable names in supplied path directories.
    ///
    /// Directory path components are also considered so explicit SDK roots such
    /// as `CUDA_PATH`, `HIP_PATH`, or `VULKAN_SDK` can advertise a runtime even
    /// when the caller has not appended their `bin` directories to `PATH`.
    /// Unreadable directories are ignored so optional SDK absence produces an
    /// empty detection result rather than a hard failure.
    #[must_use]
    pub fn detect_from_path_dirs(paths: impl IntoIterator<Item = PathBuf>) -> Vec<GpuSdk> {
        let tools = paths
            .into_iter()
            .flat_map(|path| {
                let mut names = path
                    .components()
                    .filter_map(|component| component.as_os_str().to_str().map(str::to_owned))
                    .collect::<Vec<_>>();
                if let Ok(entries) = fs::read_dir(&path) {
                    names.extend(
                        entries
                            .filter_map(Result::ok)
                            .filter_map(|entry| entry.file_name().into_string().ok()),
                    );
                }
                names
            })
            .map(|name| normalize_tool_name(&name))
            .collect::<Vec<_>>();
        Self::detect_from_tools(&tools)
    }

    /// Detects SDKs from an explicit tool-name list.
    ///
    /// This is deterministic and does not inspect the host. Runtime callers can
    /// pass PATH-discovered tool names through this function.
    #[must_use]
    pub fn detect_from_tools(tools: &[String]) -> Vec<GpuSdk> {
        let normalized: BTreeSet<String> =
            tools.iter().map(|tool| normalize_tool_name(tool)).collect();
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
            let shader_int64 = normalized
                .iter()
                .any(|tool| tool.contains("shaderint64") || tool.contains("shader_int64"));
            let sdk = if shader_int64 {
                "Vulkan compute toolchain shaderInt64"
            } else {
                "Vulkan compute toolchain"
            };
            detected.push(GpuSdk::Vulkan {
                sdk: sdk.to_owned(),
            });
        }
        if normalized.iter().any(|tool| {
            tool == "nvcc"
                || tool == "nvidia-smi"
                || tool.starts_with("nvrtc64_")
                || tool.starts_with("nvrtc")
                || tool.contains("cuda")
        }) {
            detected.push(GpuSdk::Cuda {
                sdk: "NVIDIA CUDA runtime/toolchain".to_owned(),
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

    /// Detects SDKs from tool names and augments adapter-backed runtime
    /// capabilities by querying adapter CLIs.
    #[must_use]
    pub fn detect_from_tools_with_adapter_features(
        tools: &[String],
        runner: &dyn CommandRunner,
    ) -> Vec<GpuSdk> {
        let mut detected = Self::detect_from_tools(tools);
        augment_with_adapter_features(&mut detected, runner);
        detect_from_adapter_features(&mut detected, runner);
        detected
    }
}

fn detect_from_adapter_features(detected: &mut Vec<GpuSdk>, runner: &dyn CommandRunner) {
    for mut sdk in adapter_backed_sdk_candidates() {
        if detected.iter().any(|detected_sdk| {
            std::mem::discriminant(detected_sdk) == std::mem::discriminant(&sdk)
        }) {
            continue;
        }
        let (adapter, feature) = adapter_feature_probe(&sdk);
        let output = runner.run_command(&[adapter.to_owned(), "--features".to_owned()]);
        if output.exit_code == 0 {
            if adapter_features_include(&output.stdout, feature) {
                append_sdk_feature(&mut sdk, feature);
            }
            detected.push(sdk);
        }
    }
}

fn adapter_backed_sdk_candidates() -> Vec<GpuSdk> {
    vec![
        GpuSdk::OpenCl {
            sdk: "OpenCL adapter runtime".to_owned(),
        },
        GpuSdk::Vulkan {
            sdk: "Vulkan adapter runtime".to_owned(),
        },
        GpuSdk::Cuda {
            sdk: "CUDA adapter runtime".to_owned(),
        },
        GpuSdk::Hip {
            sdk: "HIP adapter runtime".to_owned(),
        },
    ]
}

fn augment_with_adapter_features(detected: &mut [GpuSdk], runner: &dyn CommandRunner) {
    for sdk in detected {
        let (adapter, feature) = adapter_feature_probe(sdk);
        let output = runner.run_command(&[adapter.to_owned(), "--features".to_owned()]);
        if output.exit_code == 0 && adapter_features_include(&output.stdout, feature) {
            append_sdk_feature(sdk, feature);
        }
    }
}

fn adapter_feature_probe(sdk: &GpuSdk) -> (&'static str, &'static str) {
    match sdk {
        GpuSdk::OpenCl { .. } => ("atlas-gpu-opencl-run", "int64"),
        GpuSdk::Vulkan { .. } => ("atlas-gpu-vulkan-run", "shaderInt64"),
        GpuSdk::Cuda { .. } => ("atlas-gpu-cuda-run", "int64"),
        GpuSdk::Hip { .. } => ("atlas-gpu-hip-run", "int64"),
    }
}

fn append_sdk_feature(sdk: &mut GpuSdk, feature: &str) {
    let identity = match sdk {
        GpuSdk::OpenCl { sdk }
        | GpuSdk::Vulkan { sdk }
        | GpuSdk::Cuda { sdk }
        | GpuSdk::Hip { sdk } => sdk,
    };
    if !sdk_has_feature(identity, feature) {
        identity.push(' ');
        identity.push_str(feature);
    }
}

fn adapter_features_include(stdout: &str, feature: &str) -> bool {
    stdout.lines().any(|line| {
        line.trim()
            .strip_prefix("feature=")
            .is_some_and(|reported| reported.eq_ignore_ascii_case(feature))
    })
}

fn standard_sdk_root_dirs() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    #[cfg(windows)]
    {
        for base in ["ProgramFiles", "ProgramFiles(x86)"]
            .into_iter()
            .filter_map(std::env::var_os)
            .map(PathBuf::from)
        {
            let cuda_base = base.join("NVIDIA GPU Computing Toolkit").join("CUDA");
            let mut cuda_versioned_roots = Vec::new();
            if let Ok(entries) = fs::read_dir(&cuda_base) {
                cuda_versioned_roots.extend(
                    entries
                        .filter_map(Result::ok)
                        .map(|entry| entry.path())
                        .filter(|path| path.is_dir())
                        .filter(|path| {
                            path.file_name()
                                .and_then(|name| name.to_str())
                                .is_some_and(|name| name.starts_with('v'))
                        }),
                );
            }
            cuda_versioned_roots.sort_by(|left, right| {
                sdk_version_key(right)
                    .cmp(&sdk_version_key(left))
                    .then_with(|| right.cmp(left))
            });
            roots.extend(cuda_versioned_roots);
            push_existing_dir(&mut roots, cuda_base.clone());
            let rocm_base = base.join("AMD").join("ROCm");
            let mut rocm_versioned_roots = Vec::new();
            if let Ok(entries) = fs::read_dir(&rocm_base) {
                for path in entries.filter_map(Result::ok).map(|entry| entry.path()) {
                    if path.is_dir() {
                        rocm_versioned_roots.push(path);
                    }
                }
            }
            rocm_versioned_roots.sort_by(|left, right| {
                sdk_version_key(right)
                    .cmp(&sdk_version_key(left))
                    .then_with(|| right.cmp(left))
            });
            for path in rocm_versioned_roots {
                roots.push(path.clone());
                push_existing_dir(&mut roots, path.join("hip"));
            }
            push_existing_dir(&mut roots, rocm_base.clone());
            push_existing_dir(&mut roots, base.join("Khronos").join("OpenCL-SDK"));
        }
        if let Some(drive) = std::env::var_os("SystemDrive").map(PathBuf::from) {
            let vulkan_base = drive.join("VulkanSDK");
            push_existing_dir(&mut roots, vulkan_base.clone());
            if let Ok(entries) = fs::read_dir(&vulkan_base) {
                roots.extend(
                    entries
                        .filter_map(Result::ok)
                        .map(|entry| entry.path())
                        .filter(|path| path.is_dir()),
                );
            }
        }
    }
    #[cfg(not(windows))]
    {
        roots.push(PathBuf::from("/usr/local/cuda"));
    }
    let mut deduped = Vec::new();
    for root in roots {
        if !deduped.iter().any(|existing| existing == &root) {
            deduped.push(root);
        }
    }
    deduped
}

fn sdk_version_key(path: &Path) -> Vec<u32> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.trim_start_matches('v'))
        .map(|version| {
            version
                .split('.')
                .map(|component| component.parse::<u32>().unwrap_or(0))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn push_existing_dir(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if path.is_dir() {
        paths.push(path);
    }
}

fn normalize_tool_name(name: &str) -> String {
    let name = name.to_ascii_lowercase();
    name.strip_suffix(".exe")
        .or_else(|| name.strip_suffix(".dll"))
        .or_else(|| name.strip_suffix(".cmd"))
        .or_else(|| name.strip_suffix(".bat"))
        .unwrap_or(&name)
        .to_owned()
}

/// GPU searcher boundary.
pub struct GpuSearcher;

const CUDA_SELF_CONTAINED_DEVICE_HEADER: &str = r"#if defined(__clang__)
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __constant__ __attribute__((constant))
#endif
__device__ unsigned int atlas_global_id_x() {
#if defined(__clang__)
  return __nvvm_read_ptx_sreg_ctaid_x() * __nvvm_read_ptx_sreg_ntid_x() + __nvvm_read_ptx_sreg_tid_x();
#else
  return blockIdx.x * blockDim.x + threadIdx.x;
#endif
}
__device__ unsigned int atlas_atomic_add_u32(unsigned int* ptr, unsigned int value) {
#if defined(__clang__)
  return __atomic_fetch_add(ptr, value, __ATOMIC_RELAXED);
#else
  return atomicAdd(ptr, value);
#endif
}";

const HIP_SELF_CONTAINED_DEVICE_HEADER: &str = r"#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
static __device__ unsigned int atlas_global_id_x() {
  return __builtin_amdgcn_workgroup_id_x() * __builtin_amdgcn_workgroup_size_x() + __builtin_amdgcn_workitem_id_x();
}
static __device__ unsigned int atlas_atomic_add_u32(unsigned int* ptr, unsigned int value) {
  return __atomic_fetch_add(ptr, value, __ATOMIC_RELAXED);
}";

impl GpuSearcher {
    /// Generates CUDA source for the restricted IR.
    #[must_use]
    pub fn compile_cuda(program: &SearchProgram) -> String {
        let mask = width_mask(program.width);
        if program.width <= 32 {
            let predicates = program
                .ops
                .iter()
                .map(|op| cuda_predicate_32(op, program.width))
                .collect::<Vec<_>>()
                .join(" &&\n      ");
            return format!(
                r#"{cuda_header}
extern "C" __device__ __constant__ unsigned int atlas_search_u32_abi = 1U;

__device__ unsigned int rotate_left_width(unsigned int value, unsigned int amount, unsigned int width) {{
  unsigned int mask = width == 32U ? 4294967295U : ((1U << width) - 1U);
  value = value & mask;
  amount = amount % width;
  return amount == 0U ? value : (((value << amount) | (value >> (width - amount))) & mask);
}}

extern "C" __global__ void atlas_search(unsigned int start_lo, unsigned int start_hi, unsigned int end_lo, unsigned int end_hi, unsigned int* out_words, unsigned int* out_len, unsigned int max_matches) {{
  /* width={} ops={} */
  unsigned int gid = atlas_global_id_x();
  unsigned int raw_low = start_lo + gid;
  unsigned int raw_high = start_hi + (raw_low < start_lo ? 1U : 0U);
  unsigned int mask = {mask}U;
  if (raw_high > end_hi || (raw_high == end_hi && raw_low >= end_lo)) {{
    return;
  }}
  unsigned int raw_candidate = raw_low;
  unsigned int candidate = raw_candidate & mask;
  if ({predicates}) {{
    unsigned int slot = atlas_atomic_add_u32(out_len, 1U);
    if (slot < max_matches) {{
      unsigned int word_index = slot * 2U;
      out_words[word_index] = raw_low;
      out_words[word_index + 1U] = raw_high;
    }}
  }}
}}"#,
                program.width,
                program.ops.len(),
                cuda_header = CUDA_SELF_CONTAINED_DEVICE_HEADER
            );
        }
        let predicates = program
            .ops
            .iter()
            .map(|op| cuda_predicate(op, program.width))
            .collect::<Vec<_>>()
            .join(" &&\n      ");
        format!(
            r#"{cuda_header}
__device__ unsigned long long rotate_left_width(unsigned long long value, unsigned int amount, unsigned int width) {{
  unsigned long long mask = width == 64U ? 18446744073709551615ULL : ((1ULL << width) - 1ULL);
  value = value & mask;
  amount = amount % width;
  return amount == 0U ? value : (((value << amount) | (value >> (width - amount))) & mask);
}}

extern "C" __global__ void atlas_search(unsigned long long start, unsigned long long end, unsigned long long* out, unsigned int* out_len, unsigned int max_matches) {{
  /* width={} ops={} */
  unsigned long long gid = (unsigned long long)(atlas_global_id_x());
  unsigned long long raw_candidate = start + gid;
  unsigned long long mask = {mask}ULL;
  if (raw_candidate >= end) {{
    return;
  }}
  unsigned long long candidate = raw_candidate & mask;
  if ({predicates}) {{
    unsigned int slot = atlas_atomic_add_u32(out_len, 1U);
    if (slot < max_matches) {{
      out[slot] = raw_candidate;
    }}
  }}
}}"#,
            program.width,
            program.ops.len(),
            cuda_header = CUDA_SELF_CONTAINED_DEVICE_HEADER
        )
    }

    /// Generates HIP source for the restricted IR.
    #[must_use]
    pub fn compile_hip(program: &SearchProgram) -> String {
        let mask = width_mask(program.width);
        if program.width <= 32 {
            let predicates = program
                .ops
                .iter()
                .map(|op| cuda_predicate_32(op, program.width))
                .collect::<Vec<_>>()
                .join(" &&\n      ");
            return format!(
                r#"{hip_header}
extern "C" __device__ unsigned int atlas_search_u32_abi = 1U;

__device__ unsigned int rotate_left_width(unsigned int value, unsigned int amount, unsigned int width) {{
  unsigned int mask = width == 32U ? 4294967295U : ((1U << width) - 1U);
  value = value & mask;
  amount = amount % width;
  return amount == 0U ? value : (((value << amount) | (value >> (width - amount))) & mask);
}}

extern "C" __global__ void atlas_search(unsigned int start_lo, unsigned int start_hi, unsigned int end_lo, unsigned int end_hi, unsigned int* out_words, unsigned int* out_len, unsigned int max_matches) {{
  /* width={} ops={} */
  unsigned int gid = atlas_global_id_x();
  unsigned int raw_low = start_lo + gid;
  unsigned int raw_high = start_hi + (raw_low < start_lo ? 1U : 0U);
  unsigned int mask = {mask}U;
  if (raw_high > end_hi || (raw_high == end_hi && raw_low >= end_lo)) {{
    return;
  }}
  unsigned int raw_candidate = raw_low;
  unsigned int candidate = raw_candidate & mask;
  if ({predicates}) {{
    unsigned int slot = atlas_atomic_add_u32(out_len, 1U);
    if (slot < max_matches) {{
      unsigned int word_index = slot * 2U;
      out_words[word_index] = raw_low;
      out_words[word_index + 1U] = raw_high;
    }}
  }}
}}"#,
                program.width,
                program.ops.len(),
                hip_header = HIP_SELF_CONTAINED_DEVICE_HEADER
            );
        }
        let predicates = program
            .ops
            .iter()
            .map(|op| cuda_predicate(op, program.width))
            .collect::<Vec<_>>()
            .join(" &&\n      ");
        format!(
            r#"{hip_header}
__device__ unsigned long long rotate_left_width(unsigned long long value, unsigned int amount, unsigned int width) {{
  unsigned long long mask = width == 64U ? 18446744073709551615ULL : ((1ULL << width) - 1ULL);
  value = value & mask;
  amount = amount % width;
  return amount == 0U ? value : (((value << amount) | (value >> (width - amount))) & mask);
}}

extern "C" __global__ void atlas_search(unsigned long long start, unsigned long long end, unsigned long long* out, unsigned int* out_len, unsigned int max_matches) {{
  /* width={} ops={} */
  unsigned long long gid = (unsigned long long)(atlas_global_id_x());
  unsigned long long raw_candidate = start + gid;
  unsigned long long mask = {mask}ULL;
  if (raw_candidate >= end) {{
    return;
  }}
  unsigned long long candidate = raw_candidate & mask;
  if ({predicates}) {{
    unsigned int slot = atlas_atomic_add_u32(out_len, 1U);
    if (slot < max_matches) {{
      out[slot] = raw_candidate;
    }}
  }}
}}"#,
            program.width,
            program.ops.len(),
            hip_header = HIP_SELF_CONTAINED_DEVICE_HEADER
        )
    }

    /// Generates `OpenCL` C source for the restricted IR.
    #[must_use]
    pub fn compile_opencl(program: &SearchProgram) -> String {
        let mask = width_mask(program.width);
        if program.width <= 32 {
            let predicates = program
                .ops
                .iter()
                .map(|op| opencl_predicate_32(op, program.width))
                .collect::<Vec<_>>()
                .join(" &&\n      ");
            return format!(
                r"/* atlas-opencl-u32-abi */
uint rotate_left_width(uint value, uint amount, uint width) {{
  uint mask = width == 32U ? 4294967295U : ((1U << width) - 1U);
  value = value & mask;
  amount = amount % width;
  return amount == 0U ? value : (((value << amount) | (value >> (width - amount))) & mask);
}}

__kernel void atlas_search(uint start_lo, uint start_hi, uint end_lo, uint end_hi, __global uint* out_words, __global uint* out_len, uint max_matches) {{
  /* width={} ops={} */
  uint gid = (uint)get_global_id(0);
  uint raw_low = start_lo + gid;
  uint raw_high = start_hi + (uint)(raw_low < start_lo);
  uint mask = {mask}U;
  if (raw_high > end_hi || (raw_high == end_hi && raw_low >= end_lo)) {{
    return;
  }}
  uint raw_candidate = raw_low;
  uint candidate = raw_candidate & mask;
  if ({predicates}) {{
    uint slot = atomic_inc(out_len);
    if (slot < max_matches) {{
      uint word_index = slot * 2U;
      out_words[word_index] = raw_low;
      out_words[word_index + 1U] = raw_high;
    }}
  }}
}}",
                program.width,
                program.ops.len()
            );
        }
        let predicates = program
            .ops
            .iter()
            .map(|op| opencl_predicate(op, program.width))
            .collect::<Vec<_>>()
            .join(" &&\n      ");
        format!(
            r"ulong rotate_left_width(ulong value, uint amount, uint width) {{
  ulong mask = width == 64U ? 18446744073709551615UL : ((1UL << width) - 1UL);
  value = value & mask;
  amount = amount % width;
  return amount == 0U ? value : (((value << amount) | (value >> (width - amount))) & mask);
}}

__kernel void atlas_search(ulong start, ulong end, __global ulong* out, __global uint* out_len, uint max_matches) {{
  /* width={} ops={} */
  ulong gid = (ulong)get_global_id(0);
  ulong raw_candidate = start + gid;
  ulong mask = {mask}UL;
  if (raw_candidate >= end) {{
    return;
  }}
  ulong candidate = raw_candidate & mask;
  if ({predicates}) {{
    uint slot = atomic_inc(out_len);
    if (slot < max_matches) {{
      out[slot] = raw_candidate;
    }}
  }}
}}",
            program.width,
            program.ops.len()
        )
    }

    /// Generates Vulkan-compatible GLSL compute shader source for the restricted IR.
    #[must_use]
    pub fn compile_vulkan_glsl(program: &SearchProgram) -> String {
        let mask = width_mask(program.width);
        if program.width <= 32 {
            let predicates = program
                .ops
                .iter()
                .map(|op| glsl_predicate_32(op, program.width))
                .collect::<Vec<_>>()
                .join(" &&\n      ");
            return format!(
                r"#version 450

layout(local_size_x = 256) in;

layout(push_constant) uniform SearchParams {{
  uint start_lo;
  uint start_hi;
  uint end_lo;
  uint end_hi;
  uint max_matches;
}} params;

layout(set = 0, binding = 0) buffer Matches {{
  uint out_len;
  uint _pad;
  uint out_words[];
}} matches;

uint rotate_left_width(uint value, uint amount, uint width) {{
  uint mask = width == 32U ? 4294967295U : ((1U << width) - 1U);
  value = value & mask;
  amount = amount % width;
  return amount == 0U ? value : (((value << amount) | (value >> (width - amount))) & mask);
}}

void main() {{
  /* width={} ops={} */
  uint gid = gl_GlobalInvocationID.x;
  uint raw_low = params.start_lo + gid;
  uint raw_high = params.start_hi + uint(raw_low < params.start_lo);
  uint mask = {mask}U;
  if (raw_high > params.end_hi || (raw_high == params.end_hi && raw_low >= params.end_lo)) {{
    return;
  }}
  uint raw_candidate = raw_low;
  uint candidate = raw_candidate & mask;
  if ({predicates}) {{
    uint slot = atomicAdd(matches.out_len, 1U);
    if (slot < params.max_matches) {{
      uint word_index = slot * 2U;
      matches.out_words[word_index] = raw_low;
      matches.out_words[word_index + 1U] = raw_high;
    }}
  }}
}}
",
                program.width,
                program.ops.len()
            );
        }
        let predicates = program
            .ops
            .iter()
            .map(|op| glsl_predicate(op, program.width))
            .collect::<Vec<_>>()
            .join(" &&\n      ");
        format!(
            r"#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 256) in;

layout(push_constant) uniform SearchParams {{
  uint64_t start;
  uint64_t end;
  uint max_matches;
}} params;

layout(set = 0, binding = 0) buffer Matches {{
  uint out_len;
  uint64_t out_values[];
}} matches;

uint64_t rotate_left_width(uint64_t value, uint amount, uint width) {{
  uint64_t mask = width == 64U ? 18446744073709551615UL : ((1UL << width) - 1UL);
  value = value & mask;
  amount = amount % width;
  return amount == 0U ? value : (((value << amount) | (value >> (width - amount))) & mask);
}}

void main() {{
  /* width={} ops={} */
  uint64_t gid = uint64_t(gl_GlobalInvocationID.x);
  uint64_t raw_candidate = params.start + gid;
  uint64_t mask = {mask}UL;
  if (raw_candidate >= params.end) {{
    return;
  }}
  uint64_t candidate = raw_candidate & mask;
  if ({predicates}) {{
    uint slot = atomicAdd(matches.out_len, 1U);
    if (slot < params.max_matches) {{
      matches.out_values[slot] = raw_candidate;
    }}
  }}
}}
",
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

fn width_mask(width: u32) -> u64 {
    if width == 64 {
        u64::MAX
    } else {
        (1_u64 << width) - 1
    }
}

fn opencl_predicate(op: &SearchOp, width: u32) -> String {
    match *op {
        SearchOp::AddEq { addend, target } => {
            format!("((candidate + {addend}UL) & mask) == {target}UL")
        }
        SearchOp::XorEq { mask, target } => {
            format!("((candidate ^ {mask}UL) & mask) == {target}UL")
        }
        SearchOp::ChecksumEq { modulus, target } => {
            format!("(candidate % {modulus}UL) == {target}UL")
        }
        SearchOp::MulAddEq {
            multiplier,
            addend,
            target,
        } => {
            format!("((candidate * {multiplier}UL + {addend}UL) & mask) == {target}UL")
        }
        SearchOp::RotateXorEq {
            rotate_left,
            mask,
            target,
        } => {
            format!(
                "((rotate_left_width(candidate, {rotate_left}U, {width}U) ^ {mask}UL) & mask) == {target}UL"
            )
        }
        SearchOp::ByteEq { byte_index, value } => {
            let shift = byte_index.saturating_mul(8);
            format!(
                "((candidate >> {shift}U) & 255UL) == {}UL",
                u64::from(value)
            )
        }
    }
}

fn opencl_predicate_32(op: &SearchOp, width: u32) -> String {
    let width_mask = width_mask(width);
    match *op {
        SearchOp::AddEq { addend, target } => {
            if target > width_mask {
                return false_predicate_32();
            }
            format!(
                "((candidate + {}U) & mask) == {}U",
                low_u32(addend),
                exact_u32(target)
            )
        }
        SearchOp::XorEq { mask, target } => {
            if target > width_mask {
                return false_predicate_32();
            }
            format!(
                "((candidate ^ {}U) & mask) == {}U",
                low_u32(mask),
                exact_u32(target)
            )
        }
        SearchOp::ChecksumEq { modulus, target } => {
            if modulus == 0 || target > u64::from(u32::MAX) {
                return false_predicate_32();
            }
            if modulus > u64::from(u32::MAX) {
                return format!("candidate == {}U", exact_u32(target));
            }
            if target >= modulus {
                return false_predicate_32();
            }
            format!(
                "(candidate % {}U) == {}U",
                exact_u32(modulus),
                exact_u32(target)
            )
        }
        SearchOp::MulAddEq {
            multiplier,
            addend,
            target,
        } => {
            if target > width_mask {
                return false_predicate_32();
            }
            format!(
                "((candidate * {}U + {}U) & mask) == {}U",
                low_u32(multiplier),
                low_u32(addend),
                exact_u32(target)
            )
        }
        SearchOp::RotateXorEq {
            rotate_left,
            mask,
            target,
        } => {
            if target > width_mask {
                return false_predicate_32();
            }
            format!(
                "((rotate_left_width(candidate, {rotate_left}U, {width}U) ^ {}U) & mask) == {}U",
                low_u32(mask),
                exact_u32(target)
            )
        }
        SearchOp::ByteEq { byte_index, value } => {
            let shift = byte_index.saturating_mul(8);
            format!("((candidate >> {shift}U) & 255U) == {}U", u64::from(value))
        }
    }
}

fn false_predicate_32() -> String {
    "0U == 1U".to_owned()
}

fn cuda_predicate_32(op: &SearchOp, width: u32) -> String {
    let width_mask = width_mask(width);
    match *op {
        SearchOp::AddEq { addend, target } => {
            if target > width_mask {
                return false_predicate_32();
            }
            format!(
                "((candidate + {}U) & mask) == {}U",
                low_u32(addend),
                exact_u32(target)
            )
        }
        SearchOp::XorEq { mask, target } => {
            if target > width_mask {
                return false_predicate_32();
            }
            format!(
                "((candidate ^ {}U) & mask) == {}U",
                low_u32(mask),
                exact_u32(target)
            )
        }
        SearchOp::ChecksumEq { modulus, target } => {
            if modulus == 0 || target > u64::from(u32::MAX) {
                return false_predicate_32();
            }
            if modulus > u64::from(u32::MAX) {
                return format!("candidate == {}U", exact_u32(target));
            }
            if target >= modulus {
                return false_predicate_32();
            }
            format!(
                "(candidate % {}U) == {}U",
                exact_u32(modulus),
                exact_u32(target)
            )
        }
        SearchOp::MulAddEq {
            multiplier,
            addend,
            target,
        } => {
            if target > width_mask {
                return false_predicate_32();
            }
            format!(
                "((candidate * {}U + {}U) & mask) == {}U",
                low_u32(multiplier),
                low_u32(addend),
                exact_u32(target)
            )
        }
        SearchOp::RotateXorEq {
            rotate_left,
            mask,
            target,
        } => {
            if target > width_mask {
                return false_predicate_32();
            }
            format!(
                "((rotate_left_width(candidate, {rotate_left}U, {width}U) ^ {}U) & mask) == {}U",
                low_u32(mask),
                exact_u32(target)
            )
        }
        SearchOp::ByteEq { byte_index, value } => {
            let shift = byte_index.saturating_mul(8);
            format!("((candidate >> {shift}U) & 255U) == {}U", u64::from(value))
        }
    }
}

fn low_u32(value: u64) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn exact_u32(value: u64) -> u32 {
    match u32::try_from(value) {
        Ok(value) => value,
        Err(_) => unreachable!("value is checked before OpenCL u32 literal emission"),
    }
}

fn cuda_predicate(op: &SearchOp, width: u32) -> String {
    match *op {
        SearchOp::AddEq { addend, target } => {
            format!("((candidate + {addend}ULL) & mask) == {target}ULL")
        }
        SearchOp::XorEq { mask, target } => {
            format!("((candidate ^ {mask}ULL) & mask) == {target}ULL")
        }
        SearchOp::ChecksumEq { modulus, target } => {
            format!("(candidate % {modulus}ULL) == {target}ULL")
        }
        SearchOp::MulAddEq {
            multiplier,
            addend,
            target,
        } => {
            format!("((candidate * {multiplier}ULL + {addend}ULL) & mask) == {target}ULL")
        }
        SearchOp::RotateXorEq {
            rotate_left,
            mask,
            target,
        } => {
            format!(
                "((rotate_left_width(candidate, {rotate_left}U, {width}U) ^ {mask}ULL) & mask) == {target}ULL"
            )
        }
        SearchOp::ByteEq { byte_index, value } => {
            let shift = byte_index.saturating_mul(8);
            format!(
                "((candidate >> {shift}U) & 255ULL) == {}ULL",
                u64::from(value)
            )
        }
    }
}

fn glsl_predicate_32(op: &SearchOp, width: u32) -> String {
    match *op {
        SearchOp::AddEq { addend, target } => {
            format!("((candidate + {addend}U) & mask) == {target}U")
        }
        SearchOp::XorEq { mask, target } => {
            format!("((candidate ^ {mask}U) & mask) == {target}U")
        }
        SearchOp::ChecksumEq { modulus, target } => {
            format!("(candidate % {modulus}U) == {target}U")
        }
        SearchOp::MulAddEq {
            multiplier,
            addend,
            target,
        } => {
            format!("((candidate * {multiplier}U + {addend}U) & mask) == {target}U")
        }
        SearchOp::RotateXorEq {
            rotate_left,
            mask,
            target,
        } => {
            format!(
                "((rotate_left_width(candidate, {rotate_left}U, {width}U) ^ {mask}U) & mask) == {target}U"
            )
        }
        SearchOp::ByteEq { byte_index, value } => {
            let shift = byte_index.saturating_mul(8);
            format!("((candidate >> {shift}U) & 255U) == {}U", u64::from(value))
        }
    }
}

fn glsl_predicate(op: &SearchOp, width: u32) -> String {
    match *op {
        SearchOp::AddEq { addend, target } => {
            format!("((candidate + {addend}UL) & mask) == {target}UL")
        }
        SearchOp::XorEq { mask, target } => {
            format!("((candidate ^ {mask}UL) & mask) == {target}UL")
        }
        SearchOp::ChecksumEq { modulus, target } => {
            format!("(candidate % {modulus}UL) == {target}UL")
        }
        SearchOp::MulAddEq {
            multiplier,
            addend,
            target,
        } => {
            format!("((candidate * {multiplier}UL + {addend}UL) & mask) == {target}UL")
        }
        SearchOp::RotateXorEq {
            rotate_left,
            mask,
            target,
        } => {
            format!(
                "((rotate_left_width(candidate, {rotate_left}U, {width}U) ^ {mask}UL) & mask) == {target}UL"
            )
        }
        SearchOp::ByteEq { byte_index, value } => {
            let shift = byte_index.saturating_mul(8);
            format!(
                "((candidate >> {shift}U) & 255UL) == {}UL",
                u64::from(value)
            )
        }
    }
}
