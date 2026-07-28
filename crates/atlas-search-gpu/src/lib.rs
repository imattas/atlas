//! CUDA search boundary with hardware-independent validation behavior.

use atlas_scheduler::CancellationToken;
use atlas_search_ir::{SearchDomain, SearchOp, SearchProgram};
use atlas_search_native::NativeSearcher;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
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
                    "glslc",
                    "-O",
                    GpuSearcher::compile_vulkan_glsl(program),
                ),
                GpuSdk::Cuda { .. } => (
                    "gpu/cuda/atlas_search.cu",
                    "atlas_search.cu",
                    "atlas_search.ptx",
                    "nvcc",
                    "-ptx -O2",
                    GpuSearcher::compile_cuda(program),
                ),
                GpuSdk::Hip { .. } => (
                    "gpu/hip/atlas_search.hip",
                    "atlas_search.hip",
                    "atlas_search.hsaco",
                    "hipcc",
                    "--genco -O2",
                    GpuSearcher::compile_hip(program),
                ),
            };
        let source_file = join_path(output_dir, source_name);
        let artifact_file = join_path(output_dir, artifact_name);
        let compile_command = compile_command_for(compiler, options, &source_file, &artifact_file);
        let launch_input = if matches!(sdk, GpuSdk::OpenCl { .. }) {
            source_file.clone()
        } else {
            artifact_file.clone()
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
            cache_key: KernelCacheKey::new(format!("{program:?}"), compiler, sdk.name(), options),
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
        let compile = runner.run_command(&plan.compile_command);
        if compile.exit_code != 0 {
            return compile;
        }
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

    /// Executes through a GPU driver runner and validates device-reported matches.
    #[must_use]
    pub fn execute_with_driver(
        program: &SearchProgram,
        domain: SearchDomain,
        sdk: &GpuSdk,
        cancellation: &CancellationToken,
        runner: &dyn DriverRunner,
    ) -> AcceleratorReport {
        let launch = Self::plan_launch(domain, 256, 1024);
        let command_plan =
            DriverCommandPlan::for_launch(sdk, program, domain, launch, "target/atlas-gpu");
        let output = runner.run(&command_plan);
        let base_rationale = format!("{}; driver exit {}", sdk.name(), output.exit_code);
        if output.exit_code != 0 || output.reported_matches.is_empty() {
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
        let matches = GpuSearcher::cpu_validate_matches(program, &output.reported_matches);
        AcceleratorReport {
            mode: RuntimeMode::DeviceValidated,
            telemetry: RuntimeTelemetry {
                launch,
                rationale: base_rationale,
                cpu_validated: true,
                rejected_device_matches: output
                    .reported_matches
                    .len()
                    .saturating_sub(matches.len()),
            },
            matches,
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

fn join_path(output_dir: &str, artifact_name: &str) -> String {
    let output_dir = output_dir.trim_end_matches(['/', '\\']);
    format!("{output_dir}/{artifact_name}")
}

fn write_generated_source(plan: &DriverCommandPlan) -> Result<(), String> {
    let source_path = Path::new(&plan.source_file);
    if let Some(parent) = source_path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(source_path, &plan.kernel_source).map_err(|error| error.to_string())
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
    match Command::new(program).args(args).output() {
        Ok(output) => DriverRunOutput {
            exit_code: output.status.code().unwrap_or(1),
            reported_matches: DriverRunOutput::parse_reported_matches(&String::from_utf8_lossy(
                &output.stdout,
            )),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        },
        Err(error) => DriverRunOutput {
            exit_code: 127,
            reported_matches: Vec::new(),
            stdout: String::new(),
            stderr: error.to_string(),
        },
    }
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
        let mask = width_mask(program.width);
        let predicates = program
            .ops
            .iter()
            .map(|op| cuda_predicate(op, program.width))
            .collect::<Vec<_>>()
            .join(" &&\n      ");
        format!(
            r#"__device__ unsigned long long rotate_left_width(unsigned long long value, unsigned int amount, unsigned int width) {{
  unsigned long long mask = width == 64U ? 18446744073709551615ULL : ((1ULL << width) - 1ULL);
  value = value & mask;
  amount = amount % width;
  return amount == 0U ? value : (((value << amount) | (value >> (width - amount))) & mask);
}}

extern "C" __global__ void atlas_search(unsigned long long start, unsigned long long end, unsigned long long* out, unsigned int* out_len, unsigned int max_matches) {{
  /* width={} ops={} */
  unsigned long long gid = (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned long long raw_candidate = start + gid;
  unsigned long long mask = {mask}ULL;
  if (raw_candidate >= end) {{
    return;
  }}
  unsigned long long candidate = raw_candidate & mask;
  if ({predicates}) {{
    unsigned int slot = atomicAdd(out_len, 1U);
    if (slot < max_matches) {{
      out[slot] = raw_candidate;
    }}
  }}
}}"#,
            program.width,
            program.ops.len()
        )
    }

    /// Generates HIP source for the restricted IR.
    #[must_use]
    pub fn compile_hip(program: &SearchProgram) -> String {
        let mask = width_mask(program.width);
        let predicates = program
            .ops
            .iter()
            .map(|op| cuda_predicate(op, program.width))
            .collect::<Vec<_>>()
            .join(" &&\n      ");
        format!(
            r#"#include <hip/hip_runtime.h>

__device__ unsigned long long rotate_left_width(unsigned long long value, unsigned int amount, unsigned int width) {{
  unsigned long long mask = width == 64U ? 18446744073709551615ULL : ((1ULL << width) - 1ULL);
  value = value & mask;
  amount = amount % width;
  return amount == 0U ? value : (((value << amount) | (value >> (width - amount))) & mask);
}}

extern "C" __global__ void atlas_search(unsigned long long start, unsigned long long end, unsigned long long* out, unsigned int* out_len, unsigned int max_matches) {{
  /* width={} ops={} */
  unsigned long long gid = (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned long long raw_candidate = start + gid;
  unsigned long long mask = {mask}ULL;
  if (raw_candidate >= end) {{
    return;
  }}
  unsigned long long candidate = raw_candidate & mask;
  if ({predicates}) {{
    unsigned int slot = atomicAdd(out_len, 1U);
    if (slot < max_matches) {{
      out[slot] = raw_candidate;
    }}
  }}
}}"#,
            program.width,
            program.ops.len()
        )
    }

    /// Generates `OpenCL` C source for the restricted IR.
    #[must_use]
    pub fn compile_opencl(program: &SearchProgram) -> String {
        let mask = width_mask(program.width);
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
