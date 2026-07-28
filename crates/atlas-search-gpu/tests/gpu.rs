//! GPU boundary and differential tests.

use atlas_scheduler::CancellationToken;
use atlas_search_gpu::{
    AcceleratorRuntime, CommandRunner, DriverCommandPlan, DriverRunOutput, DriverRunner, GpuSdk,
    GpuSdkDetector, GpuSdkPlan, GpuSearcher, KernelCacheKey, ProcessDriverRunner, RuntimeMode,
};
use atlas_search_ir::{SearchDomain, SearchOp, SearchProgram};
use atlas_search_native::NativeSearcher;
use std::cell::RefCell;
use std::fs;
use std::path::Path;

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

    let cuda = GpuSearcher::compile_cuda(&program);

    assert!(cuda.contains("unsigned long long raw_candidate = start + gid"));
    assert!(cuda.contains("unsigned long long candidate = raw_candidate & mask"));
    assert!(cuda.contains("((candidate ^ 170ULL) & mask) == 255ULL"));
    assert!(cuda.contains("((candidate + 1ULL) & mask) == 4ULL"));
    assert!(cuda.contains("(candidate % 17ULL) == 3ULL"));
    assert!(cuda.contains("((candidate * 65537ULL + 4919ULL) & mask) == 12648430ULL"));
    assert!(cuda.contains("rotate_left_width(candidate, 7U, 24U)"));
    assert!(cuda.contains("((candidate >> 8U) & 255ULL) == 84ULL"));
    assert!(cuda.contains("atomicAdd(out_len, 1U)"));
    assert!(cuda.contains("out[slot] = raw_candidate"));
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
    assert_eq!(plan.source_file, "target/atlas-gpu/atlas_search.hip");
    assert!(plan.kernel_source.contains("#include <hip/hip_runtime.h>"));
    assert!(plan.kernel_source.contains("__global__ void atlas_search"));
    assert!(plan
        .kernel_source
        .contains("((candidate ^ 170ULL) & mask) == 255ULL"));
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

    assert_eq!(plan.artifact_file, "target/atlas-gpu/atlas_search.hsaco");
    assert_eq!(plan.compile_command[0], "hipcc");
    assert!(plan.compile_command.iter().any(|arg| arg == "--genco"));
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

    assert!(glsl.contains("uint64_t raw_candidate = params.start + gid"));
    assert!(glsl.contains("uint64_t candidate = raw_candidate & mask"));
    assert!(glsl.contains("((candidate ^ 170UL) & mask) == 255UL"));
    assert!(glsl.contains("((candidate + 1UL) & mask) == 4UL"));
    assert!(glsl.contains("(candidate % 17UL) == 3UL"));
    assert!(glsl.contains("((candidate * 65537UL + 4919UL) & mask) == 12648430UL"));
    assert!(glsl.contains("rotate_left_width(candidate, 7U, 24U)"));
    assert!(glsl.contains("((candidate >> 8U) & 255UL) == 84UL"));
    assert!(glsl.contains("atomicAdd(matches.out_len, 1U)"));
    assert!(glsl.contains("matches.out_values[slot] = raw_candidate"));
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

    assert!(opencl.contains("candidate = start + gid"));
    assert!(opencl.contains("ulong candidate = raw_candidate & mask"));
    assert!(opencl.contains("((candidate ^ 170UL) & mask) == 255UL"));
    assert!(opencl.contains("((candidate + 1UL) & mask) == 4UL"));
    assert!(opencl.contains("(candidate % 17UL) == 3UL"));
    assert!(opencl.contains("((candidate * 65537UL + 4919UL) & mask) == 12648430UL"));
    assert!(opencl.contains("rotate_left_width(candidate, 7U, 24U)"));
    assert!(opencl.contains("((candidate >> 8U) & 255UL) == 84UL"));
    assert!(opencl.contains("atomic_inc(out_len)"));
}

#[test]
fn opencl_codegen_preserves_full_candidate_when_outputting_matches() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();

    let opencl = GpuSearcher::compile_opencl(&program);

    assert!(opencl.contains("ulong raw_candidate = start + gid"));
    assert!(opencl.contains("ulong candidate = raw_candidate & mask"));
    assert!(opencl.contains("out[slot] = raw_candidate"));
}

#[test]
fn kernel_cache_key_changes_across_compiler_device_and_options() {
    let base = KernelCacheKey::new("p", "nvcc-1", "sm_90", "-O2");

    assert_ne!(base, KernelCacheKey::new("p", "nvcc-2", "sm_90", "-O2"));
    assert_ne!(base, KernelCacheKey::new("p", "nvcc-1", "sm_80", "-O2"));
    assert_ne!(base, KernelCacheKey::new("p", "nvcc-1", "sm_90", "-G"));
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
fn launch_config_bounds_workgroups_and_output_transfer_capacity() {
    let config = AcceleratorRuntime::plan_launch(SearchDomain::new(0, 1_000_000), 256, 1024);

    assert_eq!(config.local_size, 256);
    assert_eq!(config.global_size % config.local_size, 0);
    assert!(config.global_size >= 1_000_000);
    assert_eq!(config.max_matches, 1024);
    assert_eq!(config.output_buffer_bytes, 1024 * 8);
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
    assert_eq!(opencl.source_file, "target/atlas-gpu/atlas_search.cl");
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
    assert_eq!(
        plan.compile_command[2],
        "target/atlas-gpu/atlas_search.comp"
    );
    assert_eq!(plan.launch_command[1], "target/atlas-gpu/atlas_search.comp");
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
    assert_eq!(plan.compile_command[2], "target/atlas-gpu/atlas_search.cu");
    assert_eq!(plan.launch_command[1], "target/atlas-gpu/atlas_search.cu");
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

    assert_eq!(plan.source_file, "target/atlas-gpu/atlas_search.cl");
    assert!(plan.kernel_source.contains("candidate = start + gid"));
    assert!(plan
        .kernel_source
        .contains("((candidate ^ 170UL) & mask) == 255UL"));
    assert_eq!(plan.compile_command[0], "atlas-gpu-opencl-run");
    assert!(plan
        .compile_command
        .iter()
        .any(|arg| arg == "--compile-check"));
    assert!(plan
        .compile_command
        .iter()
        .any(|arg| arg == "target/atlas-gpu/atlas_search.cl"));
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
    assert!(written_source.contains("candidate = start + gid"));
    assert_eq!(runner.commands.borrow().len(), 2);
    let _ = fs::remove_dir_all(output_dir);
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
    assert_eq!(plan.launch_command[1], "target/atlas-gpu/atlas_search.cl");
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
fn public_execute_uses_placement_before_process_gpu_launch() {
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
