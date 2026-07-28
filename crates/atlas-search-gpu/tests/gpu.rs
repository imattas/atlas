//! GPU boundary and differential tests.

use atlas_scheduler::CancellationToken;
use atlas_search_gpu::{
    AcceleratorRuntime, DriverCommandPlan, DriverRunOutput, DriverRunner, GpuSdk, GpuSdkDetector,
    GpuSdkPlan, GpuSearcher, KernelCacheKey, RuntimeMode,
};
use atlas_search_ir::{SearchDomain, SearchProgram};
use atlas_search_native::NativeSearcher;
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

    assert_eq!(opencl.source_file, "gpu/opencl/atlas_search.cl");
    assert_eq!(opencl.compile_command[0], "opencl-clang");
    assert_eq!(vulkan.source_file, "gpu/vulkan/atlas_search.comp");
    assert_eq!(vulkan.compile_command[0], "glslc");
    assert_eq!(cuda.source_file, "gpu/cuda/atlas_search.cu");
    assert_eq!(cuda.compile_command[0], "nvcc");
    assert_ne!(opencl.cache_key, vulkan.cache_key);
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
            .join(&plan.source_file);

        assert!(
            workspace_path.exists(),
            "missing planned kernel source {}",
            plan.source_file
        );
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
