//! GPU boundary and differential tests.

use atlas_scheduler::CancellationToken;
use atlas_search_gpu::{
    AcceleratorRuntime, CommandRunner, DriverCommandPlan, DriverRunOutput, DriverRunner, GpuSdk,
    GpuSdkDetector, GpuSdkPlan, GpuSearcher, KernelCacheKey, ProcessDriverRunner, RuntimeMode,
    RuntimePolicy,
};
use atlas_search_ir::{SearchDomain, SearchOp, SearchProgram};
use atlas_search_native::NativeSearcher;
use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Debug)]
struct FixtureDriverRunner {
    output: DriverRunOutput,
}

impl DriverRunner for FixtureDriverRunner {
    fn run(&self, _plan: &DriverCommandPlan) -> DriverRunOutput {
        self.output.clone()
    }
}

#[derive(Debug)]
struct RecordingCommandRunner {
    commands: RefCell<Vec<Vec<String>>>,
}

impl RecordingCommandRunner {
    fn new() -> Self {
        Self {
            commands: RefCell::new(Vec::new()),
        }
    }
}

impl CommandRunner for RecordingCommandRunner {
    fn run_command(&self, command: &[String]) -> DriverRunOutput {
        self.commands.borrow_mut().push(command.to_vec());
        DriverRunOutput {
            exit_code: 0,
            reported_matches: Vec::new(),
            stdout: String::new(),
            stderr: String::new(),
        }
    }
}

#[derive(Debug)]
struct VulkanFeaturesCommandRunner {
    stdout: &'static str,
}

impl CommandRunner for VulkanFeaturesCommandRunner {
    fn run_command(&self, command: &[String]) -> DriverRunOutput {
        if command == ["atlas-gpu-vulkan-run", "--features"] {
            DriverRunOutput {
                exit_code: 0,
                reported_matches: Vec::new(),
                stdout: self.stdout.to_owned(),
                stderr: String::new(),
            }
        } else {
            DriverRunOutput {
                exit_code: 1,
                reported_matches: Vec::new(),
                stdout: String::new(),
                stderr: "adapter unavailable".to_owned(),
            }
        }
    }
}

#[derive(Debug)]
struct AdapterFeaturesCommandRunner {
    commands: RefCell<Vec<Vec<String>>>,
}

impl AdapterFeaturesCommandRunner {
    fn new() -> Self {
        Self {
            commands: RefCell::new(Vec::new()),
        }
    }
}

impl CommandRunner for AdapterFeaturesCommandRunner {
    fn run_command(&self, command: &[String]) -> DriverRunOutput {
        self.commands.borrow_mut().push(command.to_vec());
        let stdout = match command.first().map(String::as_str) {
            Some("atlas-gpu-vulkan-run") => "feature=shaderInt64\n",
            Some("atlas-gpu-opencl-run" | "atlas-gpu-cuda-run" | "atlas-gpu-hip-run") => {
                "feature=int64\n"
            }
            Some(other) => panic!("unexpected adapter feature command: {other}"),
            None => panic!("empty adapter feature command"),
        };
        DriverRunOutput {
            exit_code: 0,
            reported_matches: Vec::new(),
            stdout: stdout.to_owned(),
            stderr: String::new(),
        }
    }
}

#[derive(Debug)]
struct CountingDriverRunner {
    calls: RefCell<usize>,
    output: DriverRunOutput,
}

impl DriverRunner for CountingDriverRunner {
    fn run(&self, _plan: &DriverCommandPlan) -> DriverRunOutput {
        *self.calls.borrow_mut() += 1;
        self.output.clone()
    }
}

#[derive(Debug)]
struct RecordingPlanRunner {
    launch_domains: RefCell<Vec<SearchDomain>>,
}

impl RecordingPlanRunner {
    fn new() -> Self {
        Self {
            launch_domains: RefCell::new(Vec::new()),
        }
    }
}

impl DriverRunner for RecordingPlanRunner {
    fn run(&self, plan: &DriverCommandPlan) -> DriverRunOutput {
        let start = plan
            .launch_command
            .windows(2)
            .find_map(|args| (args[0] == "--start").then(|| args[1].parse::<u64>().unwrap()))
            .unwrap();
        let end = plan
            .launch_command
            .windows(2)
            .find_map(|args| (args[0] == "--end").then(|| args[1].parse::<u64>().unwrap()))
            .unwrap();
        self.launch_domains
            .borrow_mut()
            .push(SearchDomain::new(start, end));
        DriverRunOutput {
            exit_code: 0,
            reported_matches: vec![start],
            stdout: "device completed".to_owned(),
            stderr: String::new(),
        }
    }
}

#[derive(Debug)]
struct FailingThenSuccessfulSdkRunner {
    attempted_sdks: RefCell<Vec<&'static str>>,
}

impl FailingThenSuccessfulSdkRunner {
    fn new() -> Self {
        Self {
            attempted_sdks: RefCell::new(Vec::new()),
        }
    }
}

impl DriverRunner for FailingThenSuccessfulSdkRunner {
    fn run(&self, plan: &DriverCommandPlan) -> DriverRunOutput {
        match &plan.sdk {
            GpuSdk::OpenCl { .. } => {
                self.attempted_sdks.borrow_mut().push("OpenCL");
                DriverRunOutput {
                    exit_code: 1,
                    reported_matches: Vec::new(),
                    stdout: String::new(),
                    stderr: "OpenCL runtime launch failed".to_owned(),
                }
            }
            GpuSdk::Vulkan { .. } => {
                self.attempted_sdks.borrow_mut().push("Vulkan");
                DriverRunOutput {
                    exit_code: 0,
                    reported_matches: vec![3],
                    stdout: "device completed".to_owned(),
                    stderr: String::new(),
                }
            }
            other => panic!("unexpected SDK attempted during failover: {other:?}"),
        }
    }
}

#[test]
fn gpu_boundary_matches_native_without_hardware() {
    let token = CancellationToken::new();
    for fixture in ["add", "xor", "checksum"] {
        let program = SearchProgram::try_from_fixture(fixture).unwrap();
        let domain = SearchDomain::new(0, 64);
        assert_eq!(
            GpuSearcher::search(&program, domain, &token),
            NativeSearcher::search(&program, domain, &token)
        );
    }
}

#[test]
fn cuda_codegen_is_hardware_independent_and_mentions_shape() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let cuda = GpuSearcher::compile_cuda(&program);

    assert!(cuda.contains("__global__"));
    assert!(cuda.contains("width=8"));
}

#[test]
fn cuda_codegen_emits_restricted_ir_predicates_and_preserves_full_candidate() {
    let program = SearchProgram::new(
        64,
        vec![
            SearchOp::XorEq {
                mask: 0xaa,
                target: 0xff,
            },
            SearchOp::AddEq {
                addend: 1,
                target: 4,
            },
            SearchOp::ChecksumEq {
                modulus: 17,
                target: 3,
            },
            SearchOp::MulAddEq {
                multiplier: 65_537,
                addend: 0x1337,
                target: 0xC0_FF_EE,
            },
            SearchOp::RotateXorEq {
                rotate_left: 7,
                mask: 0xA5_A5_A5,
                target: 0x12_34_56,
            },
            SearchOp::ByteEq {
                byte_index: 1,
                value: b'T',
            },
        ],
    )
    .unwrap();

    let cuda = GpuSearcher::compile_cuda(&program);

    assert!(cuda.contains("unsigned long long raw_candidate = start + gid"));
    assert!(cuda.contains("unsigned long long candidate = raw_candidate & mask"));
    assert!(cuda.contains("((candidate ^ 170ULL) & mask) == 255ULL"));
    assert!(cuda.contains("((candidate + 1ULL) & mask) == 4ULL"));
    assert!(cuda.contains("(candidate % 17ULL) == 3ULL"));
    assert!(cuda.contains("((candidate * 65537ULL + 4919ULL) & mask) == 12648430ULL"));
    assert!(cuda.contains("rotate_left_width(candidate, 7U, 64U)"));
    assert!(cuda.contains("((candidate >> 8U) & 255ULL) == 84ULL"));
    assert!(cuda.contains("atlas_atomic_add_u32(out_len, 1U)"));
    assert!(cuda.contains("out[slot] = raw_candidate"));
}

#[test]
fn cuda_32_bit_codegen_does_not_require_64_bit_device_integer_ops() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();

    let cuda = GpuSearcher::compile_cuda(&program);

    assert!(!cuda.contains("unsigned long long"));
    assert!(!cuda.contains("ULL"));
    assert!(cuda.contains("atlas_search_u32_abi"));
    assert!(cuda.contains("unsigned int raw_candidate"));
    assert!(cuda.contains("unsigned int candidate"));
    assert!(cuda.contains("unsigned int* out_words"));
    assert!(cuda.contains("out_words[word_index] = raw_low"));
    assert!(cuda.contains("out_words[word_index + 1U] = raw_high"));
}

#[test]
fn cuda_codegen_is_self_contained_for_headerless_clang_device_compilation() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();

    let cuda = GpuSearcher::compile_cuda(&program);

    assert!(cuda.contains("#define __device__ __attribute__((device))"));
    assert!(cuda.contains("#define __global__ __attribute__((global))"));
    assert!(cuda.contains("__nvvm_read_ptx_sreg_tid_x"));
    assert!(cuda.contains("__atomic_fetch_add"));
    assert!(cuda.contains("atlas_global_id_x()"));
    assert!(cuda.contains("atlas_atomic_add_u32(out_len, 1U)"));
}

#[test]
fn cuda_32_bit_codegen_keeps_literals_in_32_bit_range() {
    let program = SearchProgram::new(
        8,
        vec![
            SearchOp::XorEq {
                mask: 0x1_0000_00aa,
                target: 0xff,
            },
            SearchOp::AddEq {
                addend: 0x1_0000_0001,
                target: 4,
            },
            SearchOp::ChecksumEq {
                modulus: 0x1_0000_0000,
                target: 0xff,
            },
        ],
    )
    .unwrap();

    let cuda = GpuSearcher::compile_cuda(&program);

    assert!(cuda.contains("((candidate ^ 170U) & mask) == 255U"));
    assert!(cuda.contains("((candidate + 1U) & mask) == 4U"));
    assert!(cuda.contains("candidate == 255U"));
    assert!(!cuda.contains("4294967296U"));
    assert!(!cuda.contains("ULL"));
}

#[test]
fn hip_driver_plan_uses_generated_hip_kernel_source() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let plan = DriverCommandPlan::for_sdk(
        &GpuSdk::Hip {
            sdk: "AMD HIP SDK".to_owned(),
        },
        &program,
        "target/atlas-gpu",
    );

    assert_eq!(plan.template_file, "gpu/hip/atlas_search.hip");
    assert!(plan.source_file.starts_with("target/atlas-gpu/"));
    assert!(plan.source_file.ends_with("/atlas_search.hip"));
    assert!(!plan.kernel_source.contains("#include <hip/hip_runtime.h>"));
    assert!(plan
        .kernel_source
        .contains("__builtin_amdgcn_workitem_id_x"));
    assert!(plan.kernel_source.contains("__global__ void atlas_search"));
    assert!(plan
        .kernel_source
        .contains("((candidate ^ 170U) & mask) == 255U"));
    assert!(plan.kernel_source.contains("atlas_search_u32_abi"));
}

#[test]
fn hip_32_bit_codegen_does_not_require_64_bit_device_integer_ops() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();

    let hip = GpuSearcher::compile_hip(&program);

    assert!(!hip.contains("unsigned long long"));
    assert!(!hip.contains("ULL"));
    assert!(hip.contains("atlas_search_u32_abi"));
    assert!(hip.contains("unsigned int raw_candidate"));
    assert!(hip.contains("unsigned int candidate"));
    assert!(hip.contains("unsigned int* out_words"));
    assert!(hip.contains("out_words[word_index] = raw_low"));
    assert!(hip.contains("out_words[word_index + 1U] = raw_high"));
}

#[test]
fn hip_codegen_is_self_contained_for_headerless_device_compilation() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();

    let hip = GpuSearcher::compile_hip(&program);

    assert!(!hip.contains("#include <hip/hip_runtime.h>"));
    assert!(hip.contains("#define __device__ __attribute__((device))"));
    assert!(hip.contains("#define __global__ __attribute__((global))"));
    assert!(hip.contains("__builtin_amdgcn_workgroup_id_x"));
    assert!(hip.contains("__atomic_fetch_add"));
    assert!(hip.contains("atlas_global_id_x()"));
    assert!(hip.contains("atlas_atomic_add_u32(out_len, 1U)"));
}

#[test]
fn hip_32_bit_codegen_keeps_literals_in_32_bit_range() {
    let program = SearchProgram::new(
        8,
        vec![
            SearchOp::XorEq {
                mask: 0x1_0000_00aa,
                target: 0xff,
            },
            SearchOp::AddEq {
                addend: 0x1_0000_0001,
                target: 4,
            },
            SearchOp::ChecksumEq {
                modulus: 0x1_0000_0000,
                target: 0xff,
            },
        ],
    )
    .unwrap();

    let hip = GpuSearcher::compile_hip(&program);

    assert!(hip.contains("((candidate ^ 170U) & mask) == 255U"));
    assert!(hip.contains("((candidate + 1U) & mask) == 4U"));
    assert!(hip.contains("candidate == 255U"));
    assert!(!hip.contains("4294967296U"));
    assert!(!hip.contains("ULL"));
}

#[test]
fn hip_driver_plan_compiles_loadable_code_object_for_adapter() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let plan = DriverCommandPlan::for_sdk(
        &GpuSdk::Hip {
            sdk: "AMD HIP SDK".to_owned(),
        },
        &program,
        "target/atlas-gpu",
    );

    assert!(plan.artifact_file.starts_with("target/atlas-gpu/"));
    assert!(plan.artifact_file.ends_with("/atlas_search.hsaco"));
    assert_eq!(plan.compile_command[0], "atlas-gpu-hip-run");
    assert!(plan
        .compile_command
        .iter()
        .any(|arg| arg == "--compile-check"));
    assert!(plan.compile_command.iter().any(|arg| arg == "-o"));
    assert!(plan
        .compile_command
        .iter()
        .any(|arg| arg == &plan.artifact_file));
    assert!(plan
        .launch_command
        .starts_with(&["atlas-gpu-hip-run".to_owned(), plan.artifact_file.clone()]));
}

#[test]
fn opencl_and_vulkan_codegen_are_hardware_independent_and_encode_shape() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let opencl = GpuSearcher::compile_opencl(&program);
    let vulkan = GpuSearcher::compile_vulkan_glsl(&program);

    assert!(opencl.contains("__kernel void atlas_search"));
    assert!(opencl.contains("width=8"));
    assert!(vulkan.contains("#version 450"));
    assert!(vulkan.contains("layout(local_size_x"));
    assert!(vulkan.contains("width=8"));
}

#[test]
fn vulkan_32_bit_codegen_does_not_require_shader_int64() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let vulkan = GpuSearcher::compile_vulkan_glsl(&program);

    assert!(!vulkan.contains("GL_EXT_shader_explicit_arithmetic_types_int64"));
    assert!(!vulkan.contains("uint64_t"));
    assert!(vulkan.contains("uint raw_candidate"));
    assert!(vulkan.contains("uint candidate"));
}

#[test]
fn vulkan_codegen_emits_restricted_ir_predicates_and_preserves_full_candidate() {
    let program = SearchProgram::new(
        24,
        vec![
            SearchOp::XorEq {
                mask: 0xaa,
                target: 0xff,
            },
            SearchOp::AddEq {
                addend: 1,
                target: 4,
            },
            SearchOp::ChecksumEq {
                modulus: 17,
                target: 3,
            },
            SearchOp::MulAddEq {
                multiplier: 65_537,
                addend: 0x1337,
                target: 0xC0_FF_EE,
            },
            SearchOp::RotateXorEq {
                rotate_left: 7,
                mask: 0xA5_A5_A5,
                target: 0x12_34_56,
            },
            SearchOp::ByteEq {
                byte_index: 1,
                value: b'T',
            },
        ],
    )
    .unwrap();

    let glsl = GpuSearcher::compile_vulkan_glsl(&program);

    assert!(glsl.contains("uint raw_candidate = raw_low"));
    assert!(glsl.contains("uint candidate = raw_candidate & mask"));
    assert!(glsl.contains("((candidate ^ 170U) & mask) == 255U"));
    assert!(glsl.contains("((candidate + 1U) & mask) == 4U"));
    assert!(glsl.contains("(candidate % 17U) == 3U"));
    assert!(glsl.contains("((candidate * 65537U + 4919U) & mask) == 12648430U"));
    assert!(glsl.contains("rotate_left_width(candidate, 7U, 24U)"));
    assert!(glsl.contains("((candidate >> 8U) & 255U) == 84U"));
    assert!(glsl.contains("atomicAdd(matches.out_len, 1U)"));
    assert!(glsl.contains("matches.out_words[word_index] = raw_low"));
    assert!(glsl.contains("matches.out_words[word_index + 1U] = raw_high"));
}

#[test]
fn opencl_codegen_emits_restricted_ir_predicates() {
    let program = SearchProgram::new(
        24,
        vec![
            SearchOp::XorEq {
                mask: 0xaa,
                target: 0xff,
            },
            SearchOp::AddEq {
                addend: 1,
                target: 4,
            },
            SearchOp::ChecksumEq {
                modulus: 17,
                target: 3,
            },
            SearchOp::MulAddEq {
                multiplier: 65_537,
                addend: 0x1337,
                target: 0xC0_FF_EE,
            },
            SearchOp::RotateXorEq {
                rotate_left: 7,
                mask: 0xA5_A5_A5,
                target: 0x12_34_56,
            },
            SearchOp::ByteEq {
                byte_index: 1,
                value: b'T',
            },
        ],
    )
    .unwrap();

    let opencl = GpuSearcher::compile_opencl(&program);

    assert!(opencl.contains("uint raw_candidate = raw_low"));
    assert!(opencl.contains("uint candidate = raw_candidate & mask"));
    assert!(opencl.contains("((candidate ^ 170U) & mask) == 255U"));
    assert!(opencl.contains("((candidate + 1U) & mask) == 4U"));
    assert!(opencl.contains("(candidate % 17U) == 3U"));
    assert!(opencl.contains("((candidate * 65537U + 4919U) & mask) == 12648430U"));
    assert!(opencl.contains("rotate_left_width(candidate, 7U, 24U)"));
    assert!(opencl.contains("((candidate >> 8U) & 255U) == 84U"));
    assert!(opencl.contains("atomic_inc(out_len)"));
}

#[test]
fn opencl_codegen_preserves_full_candidate_when_outputting_matches() {
    let program = SearchProgram::new(64, vec![SearchOp::XorEq { mask: 0, target: 0 }]).unwrap();

    let opencl = GpuSearcher::compile_opencl(&program);

    assert!(opencl.contains("ulong raw_candidate = start + gid"));
    assert!(opencl.contains("ulong candidate = raw_candidate & mask"));
    assert!(opencl.contains("out[slot] = raw_candidate"));
}

#[test]
fn opencl_32_bit_codegen_does_not_require_ulong() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();

    let opencl = GpuSearcher::compile_opencl(&program);

    assert!(!opencl.contains("ulong"));
    assert!(opencl.contains("uint raw_candidate"));
    assert!(opencl.contains("__global uint* out_words"));
    assert!(opencl.contains("out_words[word_index] = raw_low"));
    assert!(opencl.contains("out_words[word_index + 1U] = raw_high"));
}

#[test]
fn opencl_32_bit_codegen_keeps_literals_in_32_bit_range() {
    let program = SearchProgram::new(
        8,
        vec![
            SearchOp::XorEq {
                mask: 0x1_0000_00aa,
                target: 0xff,
            },
            SearchOp::AddEq {
                addend: 0x1_0000_0001,
                target: 4,
            },
            SearchOp::ChecksumEq {
                modulus: 0x1_0000_0000,
                target: 0xff,
            },
        ],
    )
    .unwrap();

    let opencl = GpuSearcher::compile_opencl(&program);

    assert!(opencl.contains("((candidate ^ 170U) & mask) == 255U"));
    assert!(opencl.contains("((candidate + 1U) & mask) == 4U"));
    assert!(opencl.contains("candidate == 255U"));
    assert!(!opencl.contains("4294967296U"));
    assert!(!opencl.contains("ulong"));
}

#[test]
fn kernel_cache_key_changes_across_compiler_device_and_options() {
    let base = KernelCacheKey::new("p", "nvcc-1", "sm_90", "-O2");

    assert_ne!(base, KernelCacheKey::new("p", "nvcc-2", "sm_90", "-O2"));
    assert_ne!(base, KernelCacheKey::new("p", "nvcc-1", "sm_80", "-O2"));
    assert_ne!(base, KernelCacheKey::new("p", "nvcc-1", "sm_90", "-G"));
}

#[test]
fn driver_cache_key_includes_generated_kernel_source() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let plan = DriverCommandPlan::for_sdk(
        &GpuSdk::Vulkan {
            sdk: "Vulkan runtime".to_owned(),
        },
        &program,
        "target/atlas-gpu",
    );

    assert_ne!(plan.cache_key.program, format!("{program:?}"));
    assert_ne!(
        plan.cache_key,
        KernelCacheKey::new(
            format!("{program:?}"),
            "atlas-gpu-vulkan-run",
            "Vulkan runtime",
            "--compile-check"
        )
    );
}

#[test]
fn driver_cache_key_changes_across_detected_device_identity() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let first_runtime = DriverCommandPlan::for_sdk(
        &GpuSdk::OpenCl {
            sdk: "OpenCL AMD Radeon RX 7900 XTX driver 1".to_owned(),
        },
        &program,
        "target/atlas-gpu",
    );
    let second_runtime = DriverCommandPlan::for_sdk(
        &GpuSdk::OpenCl {
            sdk: "OpenCL AMD Radeon RX 7900 XTX driver 2".to_owned(),
        },
        &program,
        "target/atlas-gpu",
    );

    assert_eq!(
        first_runtime.cache_key.device,
        "OpenCL AMD Radeon RX 7900 XTX driver 1"
    );
    assert_ne!(first_runtime.cache_key, second_runtime.cache_key);
}

#[test]
fn driver_plan_uses_cache_keyed_output_paths_for_distinct_kernels() {
    let add = SearchProgram::try_from_fixture("add").unwrap();
    let xor = SearchProgram::try_from_fixture("xor").unwrap();
    let sdk = GpuSdk::OpenCl {
        sdk: "test OpenCL runtime".to_owned(),
    };

    let add_plan = DriverCommandPlan::for_sdk(&sdk, &add, "target/atlas-gpu");
    let xor_plan = DriverCommandPlan::for_sdk(&sdk, &xor, "target/atlas-gpu");

    assert_ne!(add_plan.cache_key, xor_plan.cache_key);
    assert_ne!(add_plan.source_file, xor_plan.source_file);
    assert_ne!(add_plan.artifact_file, xor_plan.artifact_file);
    assert!(add_plan.source_file.starts_with("target/atlas-gpu/"));
    assert!(xor_plan.source_file.starts_with("target/atlas-gpu/"));
}

#[test]
fn cpu_validation_rejects_injected_false_gpu_match() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let validated = GpuSearcher::cpu_validate_matches(&program, &[3, 4]);

    assert_eq!(validated, vec![3]);
}

#[test]
fn plans_portable_gpu_sdks_before_vendor_specific_cuda_when_requested() {
    let plan = GpuSdkPlan::choose(
        &[
            GpuSdk::Vulkan {
                sdk: "LunarG Vulkan SDK".to_owned(),
            },
            GpuSdk::OpenCl {
                sdk: "Khronos OpenCL SDK".to_owned(),
            },
            GpuSdk::Cuda {
                sdk: "NVIDIA CUDA Toolkit".to_owned(),
            },
        ],
        true,
    );

    assert_eq!(
        plan.selected,
        Some(GpuSdk::OpenCl {
            sdk: "Khronos OpenCL SDK".to_owned(),
        })
    );
    assert!(plan.rationale.contains("portable"));
}

#[test]
fn reports_missing_gpu_sdk_without_claiming_hardware_acceleration() {
    let plan = GpuSdkPlan::choose(&[], true);

    assert_eq!(plan.selected, None);
    assert!(plan.rationale.contains("no GPU SDK"));
}

#[test]
fn detects_gpu_sdks_from_tool_names_without_touching_host_environment() {
    let detected = GpuSdkDetector::detect_from_tools(&[
        "clang".to_owned(),
        "clinfo".to_owned(),
        "glslc".to_owned(),
        "nvcc".to_owned(),
        "hipcc".to_owned(),
    ]);

    assert!(detected
        .iter()
        .any(|sdk| matches!(sdk, GpuSdk::OpenCl { .. })));
    assert!(detected
        .iter()
        .any(|sdk| matches!(sdk, GpuSdk::Vulkan { .. })));
    assert!(detected
        .iter()
        .any(|sdk| matches!(sdk, GpuSdk::Cuda { .. })));
    assert!(detected.iter().any(|sdk| matches!(sdk, GpuSdk::Hip { .. })));
}

#[test]
fn detects_gpu_sdks_from_raw_windows_tool_names() {
    let detected = GpuSdkDetector::detect_from_tools(&[
        "clinfo.exe".to_owned(),
        "glslc.cmd".to_owned(),
        "nvidia-smi.exe".to_owned(),
        "nvrtc64_120_0.dll".to_owned(),
        "hipcc.bat".to_owned(),
    ]);

    assert!(detected
        .iter()
        .any(|sdk| matches!(sdk, GpuSdk::OpenCl { .. })));
    assert!(detected
        .iter()
        .any(|sdk| matches!(sdk, GpuSdk::Vulkan { .. })));
    assert!(detected
        .iter()
        .any(|sdk| matches!(sdk, GpuSdk::Cuda { .. })));
    assert!(detected.iter().any(|sdk| matches!(sdk, GpuSdk::Hip { .. })));
}

#[test]
fn detects_vulkan_shader_int64_capability_from_tool_inventory() {
    let detected =
        GpuSdkDetector::detect_from_tools(&["vulkaninfo.exe".to_owned(), "shaderInt64".to_owned()]);

    assert!(detected.iter().any(|sdk| matches!(
        sdk,
        GpuSdk::Vulkan { sdk } if sdk.contains("shaderInt64")
    )));

    let program = SearchProgram::new(64, vec![SearchOp::XorEq { mask: 0, target: 0 }]).unwrap();
    let plan = GpuSdkPlan::choose_for_program(&detected, false, &program);

    assert!(matches!(plan.selected, Some(GpuSdk::Vulkan { .. })));
}

#[test]
fn detects_vulkan_shader_int64_capability_from_adapter_features() {
    let detected = GpuSdkDetector::detect_from_tools_with_adapter_features(
        &["vulkaninfo.exe".to_owned()],
        &VulkanFeaturesCommandRunner {
            stdout: "feature=shaderInt64\n",
        },
    );

    let program = SearchProgram::new(64, vec![SearchOp::XorEq { mask: 0, target: 0 }]).unwrap();
    let plan = GpuSdkPlan::choose_for_program(&detected, false, &program);

    assert!(matches!(plan.selected, Some(GpuSdk::Vulkan { .. })));
}

#[test]
fn adapter_feature_detection_queries_all_detected_backend_adapters() {
    let runner = AdapterFeaturesCommandRunner::new();
    let detected = GpuSdkDetector::detect_from_tools_with_adapter_features(
        &[
            "clinfo".to_owned(),
            "vulkaninfo".to_owned(),
            "nvidia-smi".to_owned(),
            "hipcc.exe".to_owned(),
        ],
        &runner,
    );

    assert_eq!(
        runner.commands.borrow().as_slice(),
        &[
            vec!["atlas-gpu-opencl-run".to_owned(), "--features".to_owned()],
            vec!["atlas-gpu-vulkan-run".to_owned(), "--features".to_owned()],
            vec!["atlas-gpu-cuda-run".to_owned(), "--features".to_owned()],
            vec!["atlas-gpu-hip-run".to_owned(), "--features".to_owned()],
        ]
    );
    assert!(detected.iter().any(|sdk| matches!(
        sdk,
        GpuSdk::OpenCl { sdk } if sdk.contains("int64")
    )));
    assert!(detected.iter().any(|sdk| matches!(
        sdk,
        GpuSdk::Vulkan { sdk } if sdk.contains("shaderInt64")
    )));
    assert!(detected.iter().any(|sdk| matches!(
        sdk,
        GpuSdk::Cuda { sdk } if sdk.contains("int64")
    )));
    assert!(detected.iter().any(|sdk| matches!(
        sdk,
        GpuSdk::Hip { sdk } if sdk.contains("int64")
    )));
}

#[test]
fn adapter_feature_detection_discovers_runtimes_without_path_tools() {
    let runner = AdapterFeaturesCommandRunner::new();
    let detected = GpuSdkDetector::detect_from_tools_with_adapter_features(&[], &runner);

    assert_eq!(
        runner.commands.borrow().clone(),
        vec![
            vec!["atlas-gpu-opencl-run".to_owned(), "--features".to_owned()],
            vec!["atlas-gpu-vulkan-run".to_owned(), "--features".to_owned()],
            vec!["atlas-gpu-cuda-run".to_owned(), "--features".to_owned()],
            vec!["atlas-gpu-hip-run".to_owned(), "--features".to_owned()],
        ]
    );
    assert!(detected.iter().any(|sdk| matches!(
        sdk,
        GpuSdk::OpenCl { sdk } if sdk.contains("int64")
    )));
    assert!(detected.iter().any(|sdk| matches!(
        sdk,
        GpuSdk::Vulkan { sdk } if sdk.contains("shaderInt64")
    )));
    assert!(detected.iter().any(|sdk| matches!(
        sdk,
        GpuSdk::Cuda { sdk } if sdk.contains("int64")
    )));
    assert!(detected.iter().any(|sdk| matches!(
        sdk,
        GpuSdk::Hip { sdk } if sdk.contains("int64")
    )));
}

#[test]
fn detects_cuda_from_runtime_and_driver_tools_without_nvcc() {
    let detected = GpuSdkDetector::detect_from_tools(&[
        "nvidia-smi.exe".to_owned(),
        "nvrtc64_120_0.dll".to_owned(),
    ]);

    assert!(detected.iter().any(|sdk| matches!(
        sdk,
        GpuSdk::Cuda { sdk } if sdk.contains("runtime")
    )));
}

#[test]
fn detects_gpu_sdks_from_path_directories() {
    let tool_dir = std::env::temp_dir().join(format!("atlas-gpu-tools-{}", std::process::id()));
    fs::create_dir_all(&tool_dir).unwrap();
    for tool in ["clinfo.exe", "vulkaninfo.exe", "hipcc.exe"] {
        fs::write(tool_dir.join(tool), "").unwrap();
    }

    let detected = GpuSdkDetector::detect_from_path_dirs([tool_dir.clone()]);

    assert!(detected
        .iter()
        .any(|sdk| matches!(sdk, GpuSdk::OpenCl { .. })));
    assert!(detected
        .iter()
        .any(|sdk| matches!(sdk, GpuSdk::Vulkan { .. })));
    assert!(detected.iter().any(|sdk| matches!(sdk, GpuSdk::Hip { .. })));
    let _ = fs::remove_dir_all(tool_dir);
}

#[test]
fn detects_gpu_sdks_from_standard_sdk_root_directories() {
    let root = std::env::temp_dir().join(format!("atlas-gpu-sdk-roots-{}", std::process::id()));
    let cuda = root
        .join("NVIDIA GPU Computing Toolkit")
        .join("CUDA")
        .join("v12.4");
    let hip = root.join("AMD").join("ROCm").join("hip");
    let vulkan = root.join("VulkanSDK").join("1.3.290.0");
    let opencl = root.join("Khronos").join("OpenCL-SDK");
    for sdk_root in [&cuda, &hip, &vulkan, &opencl] {
        fs::create_dir_all(sdk_root).unwrap();
    }

    let detected = GpuSdkDetector::detect_from_path_dirs([
        cuda.clone(),
        hip.clone(),
        vulkan.clone(),
        opencl.clone(),
    ]);

    assert!(detected
        .iter()
        .any(|sdk| matches!(sdk, GpuSdk::OpenCl { .. })));
    assert!(detected
        .iter()
        .any(|sdk| matches!(sdk, GpuSdk::Vulkan { .. })));
    assert!(detected
        .iter()
        .any(|sdk| matches!(sdk, GpuSdk::Cuda { .. })));
    assert!(detected.iter().any(|sdk| matches!(sdk, GpuSdk::Hip { .. })));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn detects_gpu_sdks_from_common_sdk_root_alias_env_vars() {
    let _env_guard = env_lock();
    let root = std::env::temp_dir().join(format!("atlas-gpu-sdk-aliases-{}", std::process::id()));
    let cuda = root.join("cuda-toolkit");
    let rocm = root.join("rocm-runtime");
    let vulkan = root.join("vulkan-sdk");
    for sdk_root in [&cuda, &rocm, &vulkan] {
        fs::create_dir_all(sdk_root).unwrap();
    }
    let original_path = std::env::var_os("PATH");
    let original_cuda_home = std::env::var_os("CUDA_HOME");
    let original_rocm_home = std::env::var_os("ROCM_HOME");
    let original_vk_sdk_path = std::env::var_os("VK_SDK_PATH");
    std::env::set_var("PATH", "");
    std::env::set_var("CUDA_HOME", &cuda);
    std::env::set_var("ROCM_HOME", &rocm);
    std::env::set_var("VK_SDK_PATH", &vulkan);

    let detected = GpuSdkDetector::detect_from_host_path();

    restore_env("PATH", original_path);
    restore_env("CUDA_HOME", original_cuda_home);
    restore_env("ROCM_HOME", original_rocm_home);
    restore_env("VK_SDK_PATH", original_vk_sdk_path);
    let _ = fs::remove_dir_all(root);
    assert!(
        detected
            .iter()
            .any(|sdk| matches!(sdk, GpuSdk::Cuda { .. })),
        "expected CUDA detection from CUDA_HOME, got {detected:?}"
    );
    assert!(
        detected.iter().any(|sdk| matches!(sdk, GpuSdk::Hip { .. })),
        "expected HIP detection from ROCM_HOME, got {detected:?}"
    );
    assert!(
        detected
            .iter()
            .any(|sdk| matches!(sdk, GpuSdk::Vulkan { .. })),
        "expected Vulkan detection from VK_SDK_PATH, got {detected:?}"
    );
}

#[test]
fn detects_gpu_sdks_from_cuda_root_and_opencl_root_alias_env_vars() {
    let _env_guard = env_lock();
    let root =
        std::env::temp_dir().join(format!("atlas-gpu-sdk-root-aliases-{}", std::process::id()));
    let cuda = root.join("cuda-root");
    let opencl = root.join("opencl-root");
    for sdk_root in [&cuda, &opencl] {
        fs::create_dir_all(sdk_root).unwrap();
    }
    let original_path = std::env::var_os("PATH");
    let original_cuda_root = std::env::var_os("CUDA_ROOT");
    let original_ocl_root = std::env::var_os("OCL_ROOT");
    std::env::set_var("PATH", "");
    std::env::set_var("CUDA_ROOT", &cuda);
    std::env::set_var("OCL_ROOT", &opencl);

    let detected = GpuSdkDetector::detect_from_host_path();

    restore_env("PATH", original_path);
    restore_env("CUDA_ROOT", original_cuda_root);
    restore_env("OCL_ROOT", original_ocl_root);
    let _ = fs::remove_dir_all(root);
    assert!(
        detected
            .iter()
            .any(|sdk| matches!(sdk, GpuSdk::Cuda { .. })),
        "expected CUDA detection from CUDA_ROOT, got {detected:?}"
    );
    assert!(
        detected
            .iter()
            .any(|sdk| matches!(sdk, GpuSdk::OpenCl { .. })),
        "expected OpenCL detection from OCL_ROOT, got {detected:?}"
    );
}

#[test]
fn detects_cuda_from_standard_program_files_toolkit_layout() {
    let _env_guard = env_lock();
    let root = std::env::temp_dir().join(format!("atlas-gpu-program-files-{}", std::process::id()));
    let cuda = root
        .join("NVIDIA GPU Computing Toolkit")
        .join("CUDA")
        .join("v12.4");
    fs::create_dir_all(&cuda).unwrap();
    let original_path = std::env::var_os("PATH");
    let original_program_files = std::env::var_os("ProgramFiles");
    std::env::set_var("PATH", "");
    std::env::set_var("ProgramFiles", &root);

    let detected = GpuSdkDetector::detect_from_host_path();

    restore_env("PATH", original_path);
    restore_env("ProgramFiles", original_program_files);
    let _ = fs::remove_dir_all(root);
    assert!(
        detected
            .iter()
            .any(|sdk| matches!(sdk, GpuSdk::Cuda { .. })),
        "expected CUDA detection from standard Toolkit layout, got {detected:?}"
    );
}

#[test]
fn detects_cuda_from_uppercase_windows_tool_suffixes() {
    let root =
        std::env::temp_dir().join(format!("atlas-gpu-uppercase-tool-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("NVCC.EXE"), []).unwrap();

    let detected = GpuSdkDetector::detect_from_path_dirs([root.clone()]);

    let _ = fs::remove_dir_all(root);
    assert!(
        detected
            .iter()
            .any(|sdk| matches!(sdk, GpuSdk::Cuda { .. })),
        "expected CUDA detection from uppercase NVCC.EXE, got {detected:?}"
    );
}

#[test]
fn detects_cuda_from_windows_command_wrapper_tools() {
    let root = std::env::temp_dir().join(format!("atlas-gpu-wrapper-tool-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("nvcc.cmd"), []).unwrap();

    let detected = GpuSdkDetector::detect_from_path_dirs([root.clone()]);

    let _ = fs::remove_dir_all(root);
    assert!(
        detected
            .iter()
            .any(|sdk| matches!(sdk, GpuSdk::Cuda { .. })),
        "expected CUDA detection from nvcc.cmd, got {detected:?}"
    );
}

#[test]
fn detects_hip_from_standard_program_files_rocm_layout() {
    let _env_guard = env_lock();
    let root = std::env::temp_dir().join(format!("atlas-gpu-amd-files-{}", std::process::id()));
    let hip = root.join("AMD").join("ROCm").join("6.1").join("hip");
    fs::create_dir_all(&hip).unwrap();
    let original_path = std::env::var_os("PATH");
    let original_program_files = std::env::var_os("ProgramFiles");
    let original_hip_path = std::env::var_os("HIP_PATH");
    let original_rocm_path = std::env::var_os("ROCM_PATH");
    let original_rocm_home = std::env::var_os("ROCM_HOME");
    std::env::set_var("PATH", "");
    std::env::set_var("ProgramFiles", &root);
    std::env::remove_var("HIP_PATH");
    std::env::remove_var("ROCM_PATH");
    std::env::remove_var("ROCM_HOME");

    let detected = GpuSdkDetector::detect_from_host_path();

    restore_env("PATH", original_path);
    restore_env("ProgramFiles", original_program_files);
    restore_env("HIP_PATH", original_hip_path);
    restore_env("ROCM_PATH", original_rocm_path);
    restore_env("ROCM_HOME", original_rocm_home);
    let _ = fs::remove_dir_all(root);
    assert!(
        detected.iter().any(|sdk| matches!(sdk, GpuSdk::Hip { .. })),
        "expected HIP detection from standard ROCm layout, got {detected:?}"
    );
}

#[test]
fn detects_vulkan_from_standard_sdk_layout() {
    let _env_guard = env_lock();
    let root = std::env::temp_dir().join(format!("atlas-gpu-khr-files-{}", std::process::id()));
    let vulkan = root.join("VulkanSDK").join("1.3.290.0");
    fs::create_dir_all(&vulkan).unwrap();
    let original_path = std::env::var_os("PATH");
    let original_vulkan_sdk = std::env::var_os("VULKAN_SDK");
    let original_vk_sdk_path = std::env::var_os("VK_SDK_PATH");
    let original_system_drive = std::env::var_os("SystemDrive");
    std::env::set_var("PATH", "");
    std::env::remove_var("VULKAN_SDK");
    std::env::remove_var("VK_SDK_PATH");
    std::env::set_var("SystemDrive", &root);

    let detected = GpuSdkDetector::detect_from_host_path();

    restore_env("PATH", original_path);
    restore_env("VULKAN_SDK", original_vulkan_sdk);
    restore_env("VK_SDK_PATH", original_vk_sdk_path);
    restore_env("SystemDrive", original_system_drive);
    let _ = fs::remove_dir_all(root);
    assert!(
        detected
            .iter()
            .any(|sdk| matches!(sdk, GpuSdk::Vulkan { .. })),
        "expected Vulkan detection from standard SDK layout, got {detected:?}"
    );
}

#[test]
fn detects_opencl_from_standard_sdk_layout() {
    let _env_guard = env_lock();
    let root = std::env::temp_dir().join(format!("atlas-gpu-open-files-{}", std::process::id()));
    let opencl = root.join("Khronos").join("OpenCL-SDK");
    fs::create_dir_all(&opencl).unwrap();
    let original_path = std::env::var_os("PATH");
    let original_program_files = std::env::var_os("ProgramFiles");
    let original_opencl_sdk = std::env::var_os("OPENCL_SDK");
    let original_ocl_root = std::env::var_os("OCL_ROOT");
    let original_intel = std::env::var_os("INTELOCLSDKROOT");
    let original_amd = std::env::var_os("AMDAPPSDKROOT");
    std::env::set_var("PATH", "");
    std::env::set_var("ProgramFiles", &root);
    std::env::remove_var("OPENCL_SDK");
    std::env::remove_var("OCL_ROOT");
    std::env::remove_var("INTELOCLSDKROOT");
    std::env::remove_var("AMDAPPSDKROOT");

    let detected = GpuSdkDetector::detect_from_host_path();

    restore_env("PATH", original_path);
    restore_env("ProgramFiles", original_program_files);
    restore_env("OPENCL_SDK", original_opencl_sdk);
    restore_env("OCL_ROOT", original_ocl_root);
    restore_env("INTELOCLSDKROOT", original_intel);
    restore_env("AMDAPPSDKROOT", original_amd);
    let _ = fs::remove_dir_all(root);
    assert!(
        detected
            .iter()
            .any(|sdk| matches!(sdk, GpuSdk::OpenCl { .. })),
        "expected OpenCL detection from standard SDK layout, got {detected:?}"
    );
}

#[test]
fn empty_standard_install_bases_do_not_report_gpu_sdks() {
    let _env_guard = env_lock();
    let root = std::env::temp_dir().join(format!("atlas-gpu-empty-bases-{}", std::process::id()));
    let program_files = root.join("ProgramFiles");
    let system_drive = root.join("SystemDrive");
    fs::create_dir_all(&program_files).unwrap();
    fs::create_dir_all(&system_drive).unwrap();
    let original_path = std::env::var_os("PATH");
    let original_program_files = std::env::var_os("ProgramFiles");
    let original_program_files_x86 = std::env::var_os("ProgramFiles(x86)");
    let original_system_drive = std::env::var_os("SystemDrive");
    let original_cuda_path = std::env::var_os("CUDA_PATH");
    let original_cuda_home = std::env::var_os("CUDA_HOME");
    let original_cuda_root = std::env::var_os("CUDA_ROOT");
    let original_hip_path = std::env::var_os("HIP_PATH");
    let original_rocm_path = std::env::var_os("ROCM_PATH");
    let original_rocm_home = std::env::var_os("ROCM_HOME");
    let original_vulkan_sdk = std::env::var_os("VULKAN_SDK");
    let original_vk_sdk_path = std::env::var_os("VK_SDK_PATH");
    let original_opencl_sdk = std::env::var_os("OPENCL_SDK");
    let original_ocl_root = std::env::var_os("OCL_ROOT");
    let original_intel = std::env::var_os("INTELOCLSDKROOT");
    let original_amd = std::env::var_os("AMDAPPSDKROOT");
    std::env::set_var("PATH", "");
    std::env::set_var("ProgramFiles", &program_files);
    std::env::remove_var("ProgramFiles(x86)");
    std::env::set_var("SystemDrive", &system_drive);
    for key in [
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
    ] {
        std::env::remove_var(key);
    }

    let detected = GpuSdkDetector::detect_from_host_path();

    restore_env("PATH", original_path);
    restore_env("ProgramFiles", original_program_files);
    restore_env("ProgramFiles(x86)", original_program_files_x86);
    restore_env("SystemDrive", original_system_drive);
    restore_env("CUDA_PATH", original_cuda_path);
    restore_env("CUDA_HOME", original_cuda_home);
    restore_env("CUDA_ROOT", original_cuda_root);
    restore_env("HIP_PATH", original_hip_path);
    restore_env("ROCM_PATH", original_rocm_path);
    restore_env("ROCM_HOME", original_rocm_home);
    restore_env("VULKAN_SDK", original_vulkan_sdk);
    restore_env("VK_SDK_PATH", original_vk_sdk_path);
    restore_env("OPENCL_SDK", original_opencl_sdk);
    restore_env("OCL_ROOT", original_ocl_root);
    restore_env("INTELOCLSDKROOT", original_intel);
    restore_env("AMDAPPSDKROOT", original_amd);
    let _ = fs::remove_dir_all(root);
    assert_eq!(detected, Vec::<GpuSdk>::new());
}

#[test]
fn launch_config_bounds_workgroups_and_output_transfer_capacity() {
    let config = AcceleratorRuntime::plan_launch(SearchDomain::new(0, 1_000_000), 256, 1024);

    assert_eq!(config.local_size, 256);
    assert_eq!(config.global_size % config.local_size, 0);
    assert!(config.global_size >= 1_000_000);
    assert_eq!(config.max_matches, 1024);
    assert_eq!(config.output_buffer_bytes, 1024 * 8);
}

#[test]
fn launch_config_never_under_covers_large_64_bit_domains() {
    let domain = SearchDomain::new(0, u64::MAX);
    let config = AcceleratorRuntime::plan_launch(domain, 256, 1024);

    assert!(
        config.global_size >= domain.end - domain.start,
        "global_size {} under-covers candidate count {}",
        config.global_size,
        domain.end - domain.start
    );
}

#[test]
fn runtime_falls_back_with_telemetry_when_no_device_is_available() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();

    let report = AcceleratorRuntime::execute(&program, SearchDomain::new(0, 64), &[], &token, &[]);

    assert_eq!(report.mode, RuntimeMode::CpuFallback);
    assert_eq!(
        report.matches,
        NativeSearcher::search(&program, SearchDomain::new(0, 64), &token)
    );
    assert!(report.telemetry.rationale.contains("no GPU SDK"));
    assert!(report.telemetry.cpu_validated);
}

#[test]
fn runtime_validates_reported_device_matches_before_returning_them() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    let sdk = GpuSdk::OpenCl {
        sdk: "test OpenCL".to_owned(),
    };

    let report =
        AcceleratorRuntime::execute(&program, SearchDomain::new(0, 64), &[sdk], &token, &[3, 4]);

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert_eq!(report.matches, vec![3]);
    assert!(report.telemetry.cpu_validated);
    assert_eq!(report.telemetry.rejected_device_matches, 1);
}

#[test]
fn runtime_canonicalizes_validated_device_matches() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    let sdk = GpuSdk::OpenCl {
        sdk: "test OpenCL".to_owned(),
    };

    let report = AcceleratorRuntime::execute(
        &program,
        SearchDomain::new(0, 512),
        &[sdk],
        &token,
        &[259, 3, 3],
    );

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert_eq!(report.matches, vec![3, 259]);
    assert!(report.telemetry.cpu_validated);
    assert_eq!(report.telemetry.rejected_device_matches, 0);
}

#[test]
fn runtime_rejects_device_matches_outside_launch_domain() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    let sdk = GpuSdk::OpenCl {
        sdk: "test OpenCL".to_owned(),
    };

    let report =
        AcceleratorRuntime::execute(&program, SearchDomain::new(4, 64), &[sdk], &token, &[3]);

    assert_eq!(report.mode, RuntimeMode::CpuFallback);
    assert!(report.matches.is_empty());
    assert!(report.telemetry.cpu_validated);
    assert_eq!(report.telemetry.rejected_device_matches, 1);
    assert!(report
        .telemetry
        .rationale
        .contains("no valid device matches"));
}

#[test]
fn public_execute_honors_cancellation_before_promoting_reported_device_matches() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    token.cancel();
    let sdk = GpuSdk::OpenCl {
        sdk: "test OpenCL".to_owned(),
    };

    let report =
        AcceleratorRuntime::execute(&program, SearchDomain::new(0, 64), &[sdk], &token, &[3]);

    assert_eq!(report.mode, RuntimeMode::CpuFallback);
    assert!(report.matches.is_empty());
    assert!(report.telemetry.rationale.contains("cancelled"));
}

#[test]
fn driver_command_plan_selects_sdk_specific_sources_and_compilers() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();

    let opencl = DriverCommandPlan::for_sdk(
        &GpuSdk::OpenCl {
            sdk: "Khronos OpenCL SDK".to_owned(),
        },
        &program,
        "target/atlas-gpu",
    );
    let vulkan = DriverCommandPlan::for_sdk(
        &GpuSdk::Vulkan {
            sdk: "LunarG Vulkan SDK".to_owned(),
        },
        &program,
        "target/atlas-gpu",
    );
    let cuda = DriverCommandPlan::for_sdk(
        &GpuSdk::Cuda {
            sdk: "NVIDIA CUDA Toolkit".to_owned(),
        },
        &program,
        "target/atlas-gpu",
    );

    assert_eq!(opencl.template_file, "gpu/opencl/atlas_search.cl");
    assert!(opencl.source_file.starts_with("target/atlas-gpu/"));
    assert!(opencl.source_file.ends_with("/atlas_search.cl"));
    assert_eq!(opencl.compile_command[0], "atlas-gpu-opencl-run");
    assert!(opencl
        .compile_command
        .iter()
        .any(|arg| arg == "--compile-check"));
    assert_eq!(vulkan.template_file, "gpu/vulkan/atlas_search.comp");
    assert_eq!(vulkan.compile_command[0], "atlas-gpu-vulkan-run");
    assert!(vulkan
        .compile_command
        .iter()
        .any(|arg| arg == "--compile-check"));
    assert_eq!(cuda.template_file, "gpu/cuda/atlas_search.cu");
    assert_eq!(cuda.compile_command[0], "atlas-gpu-cuda-run");
    assert!(cuda
        .compile_command
        .iter()
        .any(|arg| arg == "--compile-check"));
    assert_ne!(opencl.cache_key, vulkan.cache_key);
}

#[test]
fn vulkan_driver_plan_uses_adapter_compiler_without_external_glslc() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let launch = AcceleratorRuntime::plan_launch(SearchDomain::new(0x50, 0x160), 256, 8);
    let sdk = GpuSdk::Vulkan {
        sdk: "Vulkan runtime".to_owned(),
    };

    let plan = DriverCommandPlan::for_launch(
        &sdk,
        &program,
        SearchDomain::new(0x50, 0x160),
        launch,
        "target/atlas-gpu",
    );

    assert_eq!(plan.compile_command[0], "atlas-gpu-vulkan-run");
    assert!(plan
        .compile_command
        .iter()
        .any(|arg| arg == "--compile-check"));
    assert!(plan.compile_command[2].starts_with("target/atlas-gpu/"));
    assert!(plan.compile_command[2].ends_with("/atlas_search.comp"));
    assert_eq!(plan.launch_command[1], plan.artifact_file);
    assert!(plan.artifact_file.ends_with("/atlas_search.spv"));
}

#[test]
fn cuda_driver_plan_uses_adapter_compiler_without_external_nvcc() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let launch = AcceleratorRuntime::plan_launch(SearchDomain::new(0x50, 0x160), 64, 8);
    let sdk = GpuSdk::Cuda {
        sdk: "CUDA runtime".to_owned(),
    };

    let plan = DriverCommandPlan::for_launch(
        &sdk,
        &program,
        SearchDomain::new(0x50, 0x160),
        launch,
        "target/atlas-gpu",
    );

    assert_eq!(plan.compile_command[0], "atlas-gpu-cuda-run");
    assert!(plan
        .compile_command
        .iter()
        .any(|arg| arg == "--compile-check"));
    assert!(plan.compile_command[2].starts_with("target/atlas-gpu/"));
    assert!(plan.compile_command[2].ends_with("/atlas_search.cu"));
    assert_eq!(plan.launch_command[1], plan.artifact_file);
    assert!(plan.artifact_file.ends_with("/atlas_search.ptx"));
}

#[test]
fn driver_command_plans_reference_checked_in_kernel_artifacts() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    for sdk in [
        GpuSdk::OpenCl {
            sdk: "Khronos OpenCL SDK".to_owned(),
        },
        GpuSdk::Vulkan {
            sdk: "LunarG Vulkan SDK".to_owned(),
        },
        GpuSdk::Cuda {
            sdk: "NVIDIA CUDA Toolkit".to_owned(),
        },
        GpuSdk::Hip {
            sdk: "AMD HIP SDK".to_owned(),
        },
    ] {
        let plan = DriverCommandPlan::for_sdk(&sdk, &program, "target/atlas-gpu");

        let workspace_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(&plan.template_file);

        assert!(
            workspace_path.exists(),
            "missing planned kernel source {}",
            plan.template_file
        );
    }
}

#[test]
fn opencl_driver_plan_carries_generated_semantic_kernel_source() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let sdk = GpuSdk::OpenCl {
        sdk: "Khronos OpenCL SDK".to_owned(),
    };

    let plan = DriverCommandPlan::for_sdk(&sdk, &program, "target/atlas-gpu");

    assert!(plan.source_file.starts_with("target/atlas-gpu/"));
    assert!(plan.source_file.ends_with("/atlas_search.cl"));
    assert!(plan.kernel_source.contains("uint raw_candidate = raw_low"));
    assert!(plan
        .kernel_source
        .contains("((candidate ^ 170U) & mask) == 255U"));
    assert_eq!(plan.compile_command[0], "atlas-gpu-opencl-run");
    assert!(plan
        .compile_command
        .iter()
        .any(|arg| arg == "--compile-check"));
    assert!(plan.compile_command.iter().any(|arg| arg == "-o"));
    assert!(plan
        .compile_command
        .iter()
        .any(|arg| arg == &plan.artifact_file));
    assert!(plan
        .compile_command
        .iter()
        .any(|arg| arg == &plan.source_file));
    assert_eq!(plan.launch_command[1], plan.artifact_file);
}

#[test]
fn process_driver_runner_writes_generated_source_before_compile() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let sdk = GpuSdk::OpenCl {
        sdk: "Khronos OpenCL SDK".to_owned(),
    };
    let output_dir = std::env::temp_dir().join(format!("atlas-gpu-test-{}", std::process::id()));
    let output_dir_text = output_dir.to_string_lossy().into_owned();
    let plan = DriverCommandPlan::for_sdk(&sdk, &program, &output_dir_text);
    let runner = RecordingCommandRunner::new();

    let output = ProcessDriverRunner::run_with_command_runner(&plan, &runner);

    let written_source = fs::read_to_string(&plan.source_file).unwrap();
    assert_eq!(output.exit_code, 0);
    assert!(written_source.contains("uint raw_candidate = raw_low"));
    assert_eq!(runner.commands.borrow().len(), 2);
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
fn process_driver_runner_reuses_cached_compiled_artifact_without_recompile() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let sdk = GpuSdk::Vulkan {
        sdk: "Vulkan runtime".to_owned(),
    };
    let output_dir =
        std::env::temp_dir().join(format!("atlas-gpu-cache-hit-{}", std::process::id()));
    let output_dir_text = output_dir.to_string_lossy().into_owned();
    let plan = DriverCommandPlan::for_sdk(&sdk, &program, &output_dir_text);
    fs::create_dir_all(Path::new(&plan.artifact_file).parent().unwrap()).unwrap();
    fs::write(&plan.artifact_file, [0x03, 0x02, 0x23, 0x07]).unwrap();
    let runner = RecordingCommandRunner::new();

    let output = ProcessDriverRunner::run_with_command_runner(&plan, &runner);

    assert_eq!(output.exit_code, 0);
    assert_eq!(runner.commands.borrow().as_slice(), &[plan.launch_command]);
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
fn process_driver_runner_resolves_adjacent_adapter_command_when_not_on_path() {
    let adapter_name = "atlas-gpu-fallback-test-run";
    let adapter_path = write_adjacent_test_adapter(adapter_name);
    let command = vec![adapter_name.to_owned()];

    let output = ProcessDriverRunner.run_command(&command);

    let _ = fs::remove_file(adapter_path);
    assert_eq!(output.exit_code, 7);
    assert!(output.stdout.contains("adapter-ok"));
}

#[test]
fn process_driver_runner_resolves_hipcc_from_hip_sdk_root_when_not_on_path() {
    let _env_guard = env_lock();
    let sdk_root = std::env::temp_dir().join(format!("atlas-hip-sdk-root-{}", std::process::id()));
    let hipcc_path = write_sdk_tool(&sdk_root, "hipcc", "hipcc-ok", 9);
    let original_hip_path = std::env::var_os("HIP_PATH");
    std::env::set_var("HIP_PATH", &sdk_root);

    let output = ProcessDriverRunner.run_command(&["hipcc".to_owned()]);

    restore_env("HIP_PATH", original_hip_path);
    let _ = fs::remove_file(hipcc_path);
    let _ = fs::remove_dir_all(sdk_root);
    assert_eq!(output.exit_code, 9);
    assert!(output.stdout.contains("hipcc-ok"));
}

#[test]
fn process_driver_runner_resolves_nvcc_from_cuda_sdk_root_when_not_on_path() {
    let _env_guard = env_lock();
    let sdk_root = std::env::temp_dir().join(format!("atlas-cuda-sdk-root-{}", std::process::id()));
    let nvcc_path = write_sdk_tool(&sdk_root, "nvcc", "nvcc-ok", 9);
    let original_path = std::env::var_os("PATH");
    let original_cuda_path = std::env::var_os("CUDA_PATH");
    std::env::set_var("PATH", "");
    std::env::set_var("CUDA_PATH", &sdk_root);

    let output = ProcessDriverRunner.run_command(&["nvcc".to_owned()]);

    restore_env("PATH", original_path);
    restore_env("CUDA_PATH", original_cuda_path);
    let _ = fs::remove_file(nvcc_path);
    let _ = fs::remove_dir_all(sdk_root);
    assert_eq!(output.exit_code, 9);
    assert!(output.stdout.contains("nvcc-ok"));
}

#[cfg(windows)]
#[test]
fn process_driver_runner_resolves_nvcc_from_standard_cuda_toolkit_layout_when_not_on_path() {
    let _env_guard = env_lock();
    let root = std::env::temp_dir().join(format!(
        "atlas-cuda-standard-toolkit-{}",
        std::process::id()
    ));
    let sdk_root = root
        .join("NVIDIA GPU Computing Toolkit")
        .join("CUDA")
        .join("v12.4");
    let nvcc_path = write_sdk_tool(&sdk_root, "nvcc", "standard-nvcc-ok", 9);
    let original_path = std::env::var_os("PATH");
    let original_cuda_path = std::env::var_os("CUDA_PATH");
    let original_cuda_home = std::env::var_os("CUDA_HOME");
    let original_cuda_root = std::env::var_os("CUDA_ROOT");
    let original_program_files = std::env::var_os("ProgramFiles");
    let original_program_files_x86 = std::env::var_os("ProgramFiles(x86)");
    std::env::set_var("PATH", "");
    std::env::remove_var("CUDA_PATH");
    std::env::remove_var("CUDA_HOME");
    std::env::remove_var("CUDA_ROOT");
    std::env::set_var("ProgramFiles", &root);
    std::env::remove_var("ProgramFiles(x86)");

    let output = ProcessDriverRunner.run_command(&["nvcc".to_owned()]);

    restore_env("PATH", original_path);
    restore_env("CUDA_PATH", original_cuda_path);
    restore_env("CUDA_HOME", original_cuda_home);
    restore_env("CUDA_ROOT", original_cuda_root);
    restore_env("ProgramFiles", original_program_files);
    restore_env("ProgramFiles(x86)", original_program_files_x86);
    let _ = fs::remove_file(nvcc_path);
    let _ = fs::remove_dir_all(root);
    assert_eq!(output.exit_code, 9);
    assert!(output.stdout.contains("standard-nvcc-ok"));
}

#[cfg(windows)]
#[test]
fn process_driver_runner_prefers_newest_standard_cuda_toolkit_nvcc() {
    let _env_guard = env_lock();
    let root = std::env::temp_dir().join(format!(
        "atlas-cuda-standard-toolkit-order-{}",
        std::process::id()
    ));
    let cuda_base = root.join("NVIDIA GPU Computing Toolkit").join("CUDA");
    let base_nvcc_path = write_sdk_tool(&cuda_base, "nvcc", "base-nvcc", 7);
    let old_nvcc_path = write_sdk_tool(&cuda_base.join("v9.0"), "nvcc", "old-nvcc", 8);
    let new_nvcc_path = write_sdk_tool(&cuda_base.join("v12.4"), "nvcc", "new-nvcc", 9);
    let original_path = std::env::var_os("PATH");
    let original_cuda_path = std::env::var_os("CUDA_PATH");
    let original_cuda_home = std::env::var_os("CUDA_HOME");
    let original_cuda_root = std::env::var_os("CUDA_ROOT");
    let original_program_files = std::env::var_os("ProgramFiles");
    let original_program_files_x86 = std::env::var_os("ProgramFiles(x86)");
    std::env::set_var("PATH", "");
    std::env::remove_var("CUDA_PATH");
    std::env::remove_var("CUDA_HOME");
    std::env::remove_var("CUDA_ROOT");
    std::env::set_var("ProgramFiles", &root);
    std::env::remove_var("ProgramFiles(x86)");

    let output = ProcessDriverRunner.run_command(&["nvcc".to_owned()]);

    restore_env("PATH", original_path);
    restore_env("CUDA_PATH", original_cuda_path);
    restore_env("CUDA_HOME", original_cuda_home);
    restore_env("CUDA_ROOT", original_cuda_root);
    restore_env("ProgramFiles", original_program_files);
    restore_env("ProgramFiles(x86)", original_program_files_x86);
    let _ = fs::remove_file(base_nvcc_path);
    let _ = fs::remove_file(old_nvcc_path);
    let _ = fs::remove_file(new_nvcc_path);
    let _ = fs::remove_dir_all(root);
    assert_eq!(output.exit_code, 9);
    assert!(output.stdout.contains("new-nvcc"));
}

#[test]
fn process_driver_runner_resolves_hipcc_from_rocm_home_when_not_on_path() {
    let _env_guard = env_lock();
    let sdk_root = std::env::temp_dir().join(format!("atlas-rocm-home-{}", std::process::id()));
    let hipcc_path = write_sdk_tool(&sdk_root, "hipcc", "rocm-home-hipcc-ok", 9);
    let original_path = std::env::var_os("PATH");
    let original_rocm_home = std::env::var_os("ROCM_HOME");
    std::env::set_var("PATH", "");
    std::env::set_var("ROCM_HOME", &sdk_root);

    let output = ProcessDriverRunner.run_command(&["hipcc".to_owned()]);

    restore_env("PATH", original_path);
    restore_env("ROCM_HOME", original_rocm_home);
    let _ = fs::remove_file(hipcc_path);
    let _ = fs::remove_dir_all(sdk_root);
    assert_eq!(output.exit_code, 9);
    assert!(output.stdout.contains("rocm-home-hipcc-ok"));
}

#[cfg(windows)]
#[test]
fn process_driver_runner_prefers_newest_standard_rocm_hipcc() {
    let _env_guard = env_lock();
    let root = std::env::temp_dir().join(format!(
        "atlas-rocm-standard-toolkit-order-{}",
        std::process::id()
    ));
    let rocm_base = root.join("AMD").join("ROCm");
    let base_hipcc_path = write_sdk_tool(&rocm_base, "hipcc", "base-hipcc", 7);
    let old_hipcc_path = write_sdk_tool(&rocm_base.join("5.0"), "hipcc", "old-hipcc", 8);
    let new_hipcc_path = write_sdk_tool(&rocm_base.join("6.1"), "hipcc", "new-hipcc", 9);
    let original_path = std::env::var_os("PATH");
    let original_hip_path = std::env::var_os("HIP_PATH");
    let original_rocm_path = std::env::var_os("ROCM_PATH");
    let original_rocm_home = std::env::var_os("ROCM_HOME");
    let original_program_files = std::env::var_os("ProgramFiles");
    let original_program_files_x86 = std::env::var_os("ProgramFiles(x86)");
    std::env::set_var("PATH", "");
    std::env::remove_var("HIP_PATH");
    std::env::remove_var("ROCM_PATH");
    std::env::remove_var("ROCM_HOME");
    std::env::set_var("ProgramFiles", &root);
    std::env::remove_var("ProgramFiles(x86)");

    let output = ProcessDriverRunner.run_command(&["hipcc".to_owned()]);

    restore_env("PATH", original_path);
    restore_env("HIP_PATH", original_hip_path);
    restore_env("ROCM_PATH", original_rocm_path);
    restore_env("ROCM_HOME", original_rocm_home);
    restore_env("ProgramFiles", original_program_files);
    restore_env("ProgramFiles(x86)", original_program_files_x86);
    let _ = fs::remove_file(base_hipcc_path);
    let _ = fs::remove_file(old_hipcc_path);
    let _ = fs::remove_file(new_hipcc_path);
    let _ = fs::remove_dir_all(root);
    assert_eq!(output.exit_code, 9);
    assert!(output.stdout.contains("new-hipcc"));
}

#[cfg(windows)]
#[test]
fn process_driver_runner_ignores_duplicate_nested_standard_rocm_bin_roots() {
    let _env_guard = env_lock();
    let root = std::env::temp_dir().join(format!(
        "atlas-rocm-standard-toolkit-clean-{}",
        std::process::id()
    ));
    let invalid_nested_hipcc_path = write_sdk_tool(
        &root.join("AMD").join("ROCm").join("6.1").join("bin"),
        "hipcc",
        "nested-hipcc",
        13,
    );
    let original_path = std::env::var_os("PATH");
    let original_hip_path = std::env::var_os("HIP_PATH");
    let original_rocm_path = std::env::var_os("ROCM_PATH");
    let original_rocm_home = std::env::var_os("ROCM_HOME");
    let original_program_files = std::env::var_os("ProgramFiles");
    let original_program_files_x86 = std::env::var_os("ProgramFiles(x86)");
    std::env::set_var("PATH", "");
    std::env::remove_var("HIP_PATH");
    std::env::remove_var("ROCM_PATH");
    std::env::remove_var("ROCM_HOME");
    std::env::set_var("ProgramFiles", &root);
    std::env::remove_var("ProgramFiles(x86)");

    let output = ProcessDriverRunner.run_command(&["hipcc".to_owned()]);

    restore_env("PATH", original_path);
    restore_env("HIP_PATH", original_hip_path);
    restore_env("ROCM_PATH", original_rocm_path);
    restore_env("ROCM_HOME", original_rocm_home);
    restore_env("ProgramFiles", original_program_files);
    restore_env("ProgramFiles(x86)", original_program_files_x86);
    let _ = fs::remove_file(invalid_nested_hipcc_path);
    let _ = fs::remove_dir_all(root);
    assert_eq!(output.exit_code, 127);
    assert!(!output.stdout.contains("nested-hipcc"));
}

#[test]
fn driver_launch_plan_carries_domain_and_output_capacity() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let launch = AcceleratorRuntime::plan_launch(SearchDomain::new(10, 99), 128, 17);
    let sdk = GpuSdk::OpenCl {
        sdk: "Khronos OpenCL SDK".to_owned(),
    };

    let plan = DriverCommandPlan::for_launch(
        &sdk,
        &program,
        SearchDomain::new(10, 99),
        launch,
        "target/atlas-gpu",
    );

    assert!(plan
        .launch_command
        .windows(2)
        .any(|args| args == ["--start", "10"]));
    assert!(plan
        .launch_command
        .windows(2)
        .any(|args| args == ["--end", "99"]));
    assert!(plan
        .launch_command
        .windows(2)
        .any(|args| args == ["--max-matches", "17"]));
    assert!(plan
        .launch_command
        .windows(2)
        .any(|args| args == ["--global-size", "128"]));
    assert_eq!(plan.launch_command[1], plan.artifact_file);
}

fn write_adjacent_test_adapter(name: &str) -> PathBuf {
    let exe_dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    #[cfg(windows)]
    {
        let path = exe_dir.join(format!("{name}.cmd"));
        fs::write(&path, "@echo off\r\necho adapter-ok\r\nexit /b 7\r\n").unwrap();
        path
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let path = exe_dir.join(name);
        fs::write(&path, "#!/bin/sh\necho adapter-ok\nexit 7\n").unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();
        path
    }
}

fn write_sdk_tool(sdk_root: &Path, name: &str, stdout: &str, exit_code: i32) -> PathBuf {
    let bin_dir = sdk_root.join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    #[cfg(windows)]
    {
        let path = bin_dir.join(format!("{name}.cmd"));
        fs::write(
            &path,
            format!("@echo off\r\necho {stdout}\r\nexit /b {exit_code}\r\n"),
        )
        .unwrap();
        path
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let path = bin_dir.join(name);
        fs::write(
            &path,
            format!("#!/bin/sh\necho {stdout}\nexit {exit_code}\n"),
        )
        .unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();
        path
    }
}

fn restore_env(name: &str, original: Option<std::ffi::OsString>) {
    if let Some(value) = original {
        std::env::set_var(name, value);
    } else {
        std::env::remove_var(name);
    }
}

#[test]
fn runtime_executes_driver_output_and_cpu_validates_matches() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    let sdk = GpuSdk::OpenCl {
        sdk: "test OpenCL".to_owned(),
    };
    let runner = FixtureDriverRunner {
        output: DriverRunOutput {
            exit_code: 0,
            reported_matches: vec![3, 4],
            stdout: "device completed".to_owned(),
            stderr: String::new(),
        },
    };

    let report = AcceleratorRuntime::execute_with_driver(
        &program,
        SearchDomain::new(0, 64),
        &sdk,
        &token,
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert_eq!(report.matches, vec![3]);
    assert!(report.telemetry.rationale.contains("driver exit 0"));
    assert_eq!(report.telemetry.rejected_device_matches, 1);
}

#[test]
fn runtime_splits_driver_launches_that_exceed_single_dispatch_capacity() {
    let program = SearchProgram::new(
        64,
        vec![SearchOp::ChecksumEq {
            modulus: 1,
            target: 0,
        }],
    )
    .unwrap();
    let token = CancellationToken::new();
    let sdk = GpuSdk::Vulkan {
        sdk: "test Vulkan".to_owned(),
    };
    let runner = RecordingPlanRunner::new();
    let single_dispatch_capacity = u64::from(u32::MAX) * 256;

    let report = AcceleratorRuntime::execute_with_driver(
        &program,
        SearchDomain::new(0, single_dispatch_capacity + 2),
        &sdk,
        &token,
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    let launch_domains = runner.launch_domains.borrow();
    assert!(launch_domains.len() > 2);
    assert_eq!(
        launch_domains.first(),
        Some(&SearchDomain::new(0, u64::from(u32::MAX)))
    );
    assert_eq!(
        launch_domains.last(),
        Some(&SearchDomain::new(
            single_dispatch_capacity,
            single_dispatch_capacity + 2
        ))
    );
    assert!(launch_domains
        .iter()
        .all(|domain| u32::try_from(domain.end - domain.start).is_ok()));
    assert_eq!(report.matches[0], 0);
}

#[test]
fn runtime_splits_driver_launches_before_device_match_counter_can_wrap() {
    let program = SearchProgram::new(
        64,
        vec![SearchOp::ChecksumEq {
            modulus: 1,
            target: 0,
        }],
    )
    .unwrap();
    let token = CancellationToken::new();
    let sdk = GpuSdk::OpenCl {
        sdk: "test OpenCL".to_owned(),
    };
    let runner = RecordingPlanRunner::new();
    let counter_capacity = u64::from(u32::MAX);

    let report = AcceleratorRuntime::execute_with_driver(
        &program,
        SearchDomain::new(0, counter_capacity + 2),
        &sdk,
        &token,
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert_eq!(
        runner.launch_domains.borrow().as_slice(),
        &[
            SearchDomain::new(0, counter_capacity),
            SearchDomain::new(counter_capacity, counter_capacity + 2),
        ]
    );
    assert_eq!(report.matches, vec![0, counter_capacity]);
}

#[test]
fn runtime_telemetry_reports_multi_launch_driver_execution() {
    let program = SearchProgram::new(
        64,
        vec![SearchOp::ChecksumEq {
            modulus: 1,
            target: 0,
        }],
    )
    .unwrap();
    let token = CancellationToken::new();
    let sdk = GpuSdk::OpenCl {
        sdk: "test OpenCL".to_owned(),
    };
    let runner = RecordingPlanRunner::new();
    let counter_capacity = u64::from(u32::MAX);

    let report = AcceleratorRuntime::execute_with_driver(
        &program,
        SearchDomain::new(0, counter_capacity + 2),
        &sdk,
        &token,
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert!(report.telemetry.rationale.contains("driver launches 2"));
}

#[test]
fn runtime_enforces_launch_output_capacity_after_cpu_validation() {
    let program = SearchProgram::new(
        16,
        vec![SearchOp::ChecksumEq {
            modulus: 1,
            target: 0,
        }],
    )
    .unwrap();
    let token = CancellationToken::new();
    let sdk = GpuSdk::OpenCl {
        sdk: "test OpenCL".to_owned(),
    };
    let runner = FixtureDriverRunner {
        output: DriverRunOutput {
            exit_code: 0,
            reported_matches: (0..1_500).collect(),
            stdout: "device completed".to_owned(),
            stderr: String::new(),
        },
    };

    let report = AcceleratorRuntime::execute_with_driver(
        &program,
        SearchDomain::new(0, 2_000),
        &sdk,
        &token,
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert_eq!(report.matches.len(), report.telemetry.launch.max_matches);
    assert_eq!(report.matches[0], 0);
    assert_eq!(report.matches[1023], 1023);
    assert_eq!(report.telemetry.rejected_device_matches, 476);
}

#[test]
fn runtime_falls_back_when_driver_reports_only_invalid_matches() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    let sdk = GpuSdk::OpenCl {
        sdk: "test OpenCL".to_owned(),
    };
    let runner = FixtureDriverRunner {
        output: DriverRunOutput {
            exit_code: 0,
            reported_matches: vec![4],
            stdout: "device completed".to_owned(),
            stderr: String::new(),
        },
    };

    let report = AcceleratorRuntime::execute_with_driver(
        &program,
        SearchDomain::new(0, 64),
        &sdk,
        &token,
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::CpuFallback);
    assert_eq!(
        report.matches,
        NativeSearcher::search(&program, SearchDomain::new(0, 64), &token)
    );
    assert_eq!(report.telemetry.rejected_device_matches, 1);
    assert!(report
        .telemetry
        .rationale
        .contains("no valid device matches"));
}

#[test]
fn runtime_selects_detected_sdk_and_executes_driver_runner() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    let sdks = [GpuSdk::OpenCl {
        sdk: "test OpenCL".to_owned(),
    }];
    let runner = FixtureDriverRunner {
        output: DriverRunOutput {
            exit_code: 0,
            reported_matches: vec![3, 4],
            stdout: "device completed".to_owned(),
            stderr: String::new(),
        },
    };

    let report = AcceleratorRuntime::execute_with_detected_driver(
        &program,
        SearchDomain::new(0, 1_000_000),
        &sdks,
        &token,
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert_eq!(report.matches, vec![3]);
    assert!(report.telemetry.rationale.contains("OpenCL"));
    assert!(report.telemetry.rationale.contains("driver exit 0"));
}

#[test]
fn runtime_tries_next_compatible_gpu_sdk_when_selected_driver_fails() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    let sdks = [
        GpuSdk::OpenCl {
            sdk: "test OpenCL".to_owned(),
        },
        GpuSdk::Vulkan {
            sdk: "test Vulkan".to_owned(),
        },
    ];
    let runner = FailingThenSuccessfulSdkRunner::new();

    let report = AcceleratorRuntime::execute_with_detected_driver_and_policy(
        &program,
        SearchDomain::new(0, 1_000_000),
        &sdks,
        &token,
        RuntimePolicy { force_gpu: true },
        &[],
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert_eq!(report.matches, vec![3]);
    assert_eq!(
        runner.attempted_sdks.borrow().as_slice(),
        ["OpenCL", "Vulkan"]
    );
    assert!(report.telemetry.rationale.contains("Vulkan"));
    assert!(report.telemetry.rationale.contains("driver exit 0"));
}

#[test]
fn runtime_telemetry_includes_driver_stderr_on_gpu_launch_failure() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    let sdk = GpuSdk::Cuda {
        sdk: "test CUDA".to_owned(),
    };
    let runner = FixtureDriverRunner {
        output: DriverRunOutput {
            exit_code: 1,
            reported_matches: Vec::new(),
            stdout: String::new(),
            stderr: "failed to load CUDA driver library nvcuda.dll".to_owned(),
        },
    };

    let report = AcceleratorRuntime::execute_with_driver(
        &program,
        SearchDomain::new(0, 1_000_000),
        &sdk,
        &token,
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::CpuFallback);
    assert!(report.telemetry.rationale.contains("CUDA"));
    assert!(report.telemetry.rationale.contains("driver exit 1"));
    assert!(report.telemetry.rationale.contains("nvcuda.dll"));
}

#[test]
fn runtime_honors_cancellation_before_launching_gpu_driver() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    token.cancel();
    let sdks = [GpuSdk::OpenCl {
        sdk: "test OpenCL".to_owned(),
    }];
    let runner = CountingDriverRunner {
        calls: RefCell::new(0),
        output: DriverRunOutput {
            exit_code: 0,
            reported_matches: vec![3],
            stdout: "device completed".to_owned(),
            stderr: String::new(),
        },
    };

    let report = AcceleratorRuntime::execute_with_detected_driver(
        &program,
        SearchDomain::new(0, 1_000_000),
        &sdks,
        &token,
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::CpuFallback);
    assert_eq!(*runner.calls.borrow(), 0);
    assert!(report.telemetry.rationale.contains("cancelled"));
}

#[test]
fn runtime_uses_scalar_for_tiny_workloads_without_launching_gpu_driver() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    let sdks = [GpuSdk::OpenCl {
        sdk: "test OpenCL".to_owned(),
    }];
    let runner = CountingDriverRunner {
        calls: RefCell::new(0),
        output: DriverRunOutput {
            exit_code: 0,
            reported_matches: vec![3],
            stdout: "device completed".to_owned(),
            stderr: String::new(),
        },
    };

    let report = AcceleratorRuntime::execute_with_detected_driver(
        &program,
        SearchDomain::new(0, 64),
        &sdks,
        &token,
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::CpuFallback);
    assert_eq!(*runner.calls.borrow(), 0);
    assert!(report.telemetry.rationale.contains("Scalar"));
}

#[test]
fn runtime_force_gpu_policy_launches_driver_for_tiny_workloads() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    let sdks = [GpuSdk::OpenCl {
        sdk: "test OpenCL".to_owned(),
    }];
    let runner = CountingDriverRunner {
        calls: RefCell::new(0),
        output: DriverRunOutput {
            exit_code: 0,
            reported_matches: vec![3],
            stdout: "device completed".to_owned(),
            stderr: String::new(),
        },
    };

    let report = AcceleratorRuntime::execute_with_detected_driver_and_policy(
        &program,
        SearchDomain::new(0, 64),
        &sdks,
        &token,
        RuntimePolicy { force_gpu: true },
        &[],
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert_eq!(*runner.calls.borrow(), 1);
    assert_eq!(report.matches, vec![3]);
}

#[test]
fn runtime_skips_plain_vulkan_for_64_bit_programs() {
    let program = SearchProgram::new(
        64,
        vec![SearchOp::ChecksumEq {
            modulus: 1,
            target: 0,
        }],
    )
    .unwrap();
    let token = CancellationToken::new();
    let sdks = [
        GpuSdk::Vulkan {
            sdk: "plain Vulkan runtime".to_owned(),
        },
        GpuSdk::Cuda {
            sdk: "CUDA runtime int64".to_owned(),
        },
    ];
    let runner = FixtureDriverRunner {
        output: DriverRunOutput {
            exit_code: 0,
            reported_matches: vec![0],
            stdout: "device completed".to_owned(),
            stderr: String::new(),
        },
    };

    let report = AcceleratorRuntime::execute_with_detected_driver_and_policy(
        &program,
        SearchDomain::new(0, 1_000_000),
        &sdks,
        &token,
        RuntimePolicy { force_gpu: false },
        &[],
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert!(report.telemetry.rationale.contains("CUDA"));
    assert!(!report.telemetry.rationale.contains("Vulkan"));
}

#[test]
fn runtime_skips_opencl_without_int64_for_64_bit_programs() {
    let program = SearchProgram::new(64, vec![SearchOp::XorEq { mask: 0, target: 0 }]).unwrap();
    let token = CancellationToken::new();
    let sdks = [
        GpuSdk::OpenCl {
            sdk: "OpenCL runtime".to_owned(),
        },
        GpuSdk::Hip {
            sdk: "HIP runtime int64".to_owned(),
        },
    ];
    let runner = FixtureDriverRunner {
        output: DriverRunOutput {
            exit_code: 0,
            reported_matches: vec![0],
            stdout: "device completed".to_owned(),
            stderr: String::new(),
        },
    };

    let report = AcceleratorRuntime::execute_with_detected_driver_and_policy(
        &program,
        SearchDomain::new(0, 1_000_000),
        &sdks,
        &token,
        RuntimePolicy { force_gpu: false },
        &[],
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert!(report.telemetry.rationale.contains("HIP"));
    assert!(!report.telemetry.rationale.contains("OpenCL"));
}

#[test]
fn runtime_does_not_treat_negated_int64_text_as_capability() {
    let program = SearchProgram::new(64, vec![SearchOp::XorEq { mask: 0, target: 0 }]).unwrap();
    let token = CancellationToken::new();
    let sdks = [
        GpuSdk::OpenCl {
            sdk: "OpenCL runtime no-int64".to_owned(),
        },
        GpuSdk::Hip {
            sdk: "HIP runtime int64".to_owned(),
        },
    ];
    let runner = FixtureDriverRunner {
        output: DriverRunOutput {
            exit_code: 0,
            reported_matches: vec![0],
            stdout: "device completed".to_owned(),
            stderr: String::new(),
        },
    };

    let report = AcceleratorRuntime::execute_with_detected_driver_and_policy(
        &program,
        SearchDomain::new(0, 1_000_000),
        &sdks,
        &token,
        RuntimePolicy { force_gpu: false },
        &[],
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert!(report.telemetry.rationale.contains("HIP"));
    assert!(!report.telemetry.rationale.contains("OpenCL"));
}

#[test]
fn runtime_force_gpu_rejects_plain_vulkan_for_64_bit_programs() {
    let program = SearchProgram::new(64, vec![SearchOp::XorEq { mask: 0, target: 0 }]).unwrap();
    let token = CancellationToken::new();
    let sdks = [GpuSdk::Vulkan {
        sdk: "plain Vulkan runtime".to_owned(),
    }];
    let runner = CountingDriverRunner {
        calls: RefCell::new(0),
        output: DriverRunOutput {
            exit_code: 0,
            reported_matches: vec![0],
            stdout: "device completed".to_owned(),
            stderr: String::new(),
        },
    };

    let report = AcceleratorRuntime::execute_with_detected_driver_and_policy(
        &program,
        SearchDomain::new(0, 64),
        &sdks,
        &token,
        RuntimePolicy { force_gpu: true },
        &[],
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::CpuFallback);
    assert_eq!(*runner.calls.borrow(), 0);
    assert!(report
        .telemetry
        .rationale
        .contains("no compatible GPU SDK detected for search program"));
}

#[test]
fn runtime_force_gpu_uses_compatible_backend_for_64_bit_programs() {
    let program = SearchProgram::new(64, vec![SearchOp::XorEq { mask: 0, target: 0 }]).unwrap();
    let token = CancellationToken::new();
    let sdks = [
        GpuSdk::Vulkan {
            sdk: "plain Vulkan runtime".to_owned(),
        },
        GpuSdk::Cuda {
            sdk: "CUDA runtime int64".to_owned(),
        },
    ];
    let runner = FixtureDriverRunner {
        output: DriverRunOutput {
            exit_code: 0,
            reported_matches: vec![0],
            stdout: "device completed".to_owned(),
            stderr: String::new(),
        },
    };

    let report = AcceleratorRuntime::execute_with_detected_driver_and_policy(
        &program,
        SearchDomain::new(0, 64),
        &sdks,
        &token,
        RuntimePolicy { force_gpu: true },
        &[],
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert!(report.telemetry.rationale.contains("CUDA"));
    assert!(!report.telemetry.rationale.contains("Vulkan"));
}

#[test]
fn runtime_uses_gpu_cache_hit_threshold_for_warmed_kernel() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    let domain = SearchDomain::new(0, 100_000);
    let sdk = GpuSdk::OpenCl {
        sdk: "test OpenCL runtime".to_owned(),
    };
    let launch = AcceleratorRuntime::plan_launch(domain, 256, 1024);
    let cached_plan =
        DriverCommandPlan::for_launch(&sdk, &program, domain, launch, "target/atlas-gpu");
    let sdks = [sdk];
    let runner = CountingDriverRunner {
        calls: RefCell::new(0),
        output: DriverRunOutput {
            exit_code: 0,
            reported_matches: vec![3],
            stdout: "device completed".to_owned(),
            stderr: String::new(),
        },
    };

    let report = AcceleratorRuntime::execute_with_detected_driver_and_kernel_cache(
        &program,
        domain,
        &sdks,
        &token,
        &[cached_plan.cache_key],
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert_eq!(*runner.calls.borrow(), 1);
    assert!(report.telemetry.rationale.contains("driver exit 0"));
}

#[test]
fn host_runtime_uses_persisted_kernel_cache_for_warmed_gpu_threshold() {
    struct Cleanup {
        tool_dir: PathBuf,
        adapter_path: PathBuf,
        artifact_root: PathBuf,
        original_path: Option<std::ffi::OsString>,
    }

    impl Drop for Cleanup {
        fn drop(&mut self) {
            restore_env("PATH", self.original_path.clone());
            let _ = fs::remove_dir_all(&self.tool_dir);
            let _ = fs::remove_file(&self.adapter_path);
            let _ = fs::remove_dir_all(&self.artifact_root);
        }
    }

    let _env_guard = env_lock();
    let tool_dir =
        std::env::temp_dir().join(format!("atlas-gpu-host-cache-{}", std::process::id()));
    fs::create_dir_all(&tool_dir).unwrap();
    fs::write(tool_dir.join("clinfo.exe"), "").unwrap();
    let exe_dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let adapter_path = exe_dir.join(if cfg!(windows) {
        "atlas-gpu-opencl-run.cmd"
    } else {
        "atlas-gpu-opencl-run"
    });
    let adapter_fixture = if cfg!(windows) {
        "@echo off\r\necho match=3\r\nexit /b 0\r\n"
    } else {
        "#!/bin/sh\necho match=3\nexit 0\n"
    };
    if adapter_path.exists() {
        let existing = fs::read_to_string(&adapter_path).unwrap_or_default();
        if existing == adapter_fixture {
            fs::remove_file(&adapter_path).unwrap();
        }
    }
    assert!(
        !adapter_path.exists(),
        "test would overwrite existing adapter: {}",
        adapter_path.display()
    );
    fs::write(&adapter_path, adapter_fixture).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(&adapter_path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&adapter_path, permissions).unwrap();
    }

    let program = SearchProgram::try_from_fixture("add").unwrap();
    let domain = SearchDomain::new(0, 100_000);
    let sdk = GpuSdk::OpenCl {
        sdk: "Khronos OpenCL-compatible toolchain".to_owned(),
    };
    let launch = AcceleratorRuntime::plan_launch(domain, 256, 1024);
    let cached_plan =
        DriverCommandPlan::for_launch(&sdk, &program, domain, launch, "target/atlas-gpu");
    fs::create_dir_all(Path::new(&cached_plan.artifact_file).parent().unwrap()).unwrap();
    fs::write(&cached_plan.artifact_file, [0xCA, 0xFE]).unwrap();
    let artifact_root = Path::new(&cached_plan.artifact_file)
        .parent()
        .unwrap()
        .to_path_buf();

    let original_path = std::env::var_os("PATH");
    let joined_path = std::env::join_paths(std::iter::once(tool_dir.clone())).unwrap();
    std::env::set_var("PATH", &joined_path);
    let _cleanup = Cleanup {
        tool_dir,
        adapter_path,
        artifact_root,
        original_path,
    };
    let token = CancellationToken::new();

    let report = AcceleratorRuntime::execute_with_host_driver(&program, domain, &token);

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert_eq!(report.matches, vec![3]);
    assert!(report.telemetry.rationale.contains("driver exit 0"));
}

#[test]
fn public_execute_uses_placement_before_process_gpu_launch() {
    let _env_guard = env_lock();
    let tool_dir =
        std::env::temp_dir().join(format!("atlas-gpu-public-path-{}", std::process::id()));
    fs::create_dir_all(&tool_dir).unwrap();
    let adapter_path = tool_dir.join(if cfg!(windows) {
        "atlas-gpu-opencl-run.bat"
    } else {
        "atlas-gpu-opencl-run"
    });
    fs::write(&adapter_path, "exit 42").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(&adapter_path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&adapter_path, permissions).unwrap();
    }
    let original_path = std::env::var_os("PATH").unwrap_or_default();
    let joined_path = std::env::join_paths(
        std::iter::once(tool_dir.clone()).chain(std::env::split_paths(&original_path)),
    )
    .unwrap();
    std::env::set_var("PATH", &joined_path);

    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    let report = AcceleratorRuntime::execute(
        &program,
        SearchDomain::new(0, 64),
        &[GpuSdk::OpenCl {
            sdk: "test OpenCL".to_owned(),
        }],
        &token,
        &[],
    );

    std::env::set_var("PATH", original_path);
    let _ = fs::remove_dir_all(tool_dir);
    assert_eq!(report.mode, RuntimeMode::CpuFallback);
    assert!(report.telemetry.rationale.contains("Scalar"));
    assert!(!report.telemetry.rationale.contains("driver exit 42"));
}

#[test]
fn runtime_discovers_host_path_sdks_and_executes_driver_runner() {
    let tool_dir =
        std::env::temp_dir().join(format!("atlas-gpu-runtime-tools-{}", std::process::id()));
    fs::create_dir_all(&tool_dir).unwrap();
    fs::write(tool_dir.join("clinfo.exe"), "").unwrap();
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    let runner = FixtureDriverRunner {
        output: DriverRunOutput {
            exit_code: 0,
            reported_matches: vec![3],
            stdout: "device completed".to_owned(),
            stderr: String::new(),
        },
    };

    let report = AcceleratorRuntime::execute_with_path_detected_driver(
        &program,
        SearchDomain::new(0, 1_000_000),
        [tool_dir.clone()],
        &token,
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::DeviceValidated);
    assert_eq!(report.matches, vec![3]);
    assert!(report.telemetry.rationale.contains("OpenCL"));
    let _ = fs::remove_dir_all(tool_dir);
}

#[test]
fn runtime_falls_back_when_driver_execution_fails() {
    let program = SearchProgram::try_from_fixture("add").unwrap();
    let token = CancellationToken::new();
    let sdk = GpuSdk::Vulkan {
        sdk: "test Vulkan".to_owned(),
    };
    let runner = FixtureDriverRunner {
        output: DriverRunOutput {
            exit_code: 42,
            reported_matches: Vec::new(),
            stdout: String::new(),
            stderr: "shader compile failed".to_owned(),
        },
    };

    let report = AcceleratorRuntime::execute_with_driver(
        &program,
        SearchDomain::new(0, 64),
        &sdk,
        &token,
        &runner,
    );

    assert_eq!(report.mode, RuntimeMode::CpuFallback);
    assert_eq!(
        report.matches,
        NativeSearcher::search(&program, SearchDomain::new(0, 64), &token)
    );
    assert!(report.telemetry.rationale.contains("driver exit 42"));
}

#[test]
fn driver_output_parses_decimal_and_hex_device_matches_from_stdout() {
    assert_eq!(
        DriverRunOutput::parse_reported_matches("match=3\n0x04\nignored\n5\n"),
        vec![3, 4, 5]
    );
}
