//! GPU boundary and differential tests.

use atlas_scheduler::CancellationToken;
use atlas_search_gpu::{
    AcceleratorRuntime, GpuSdk, GpuSdkDetector, GpuSdkPlan, GpuSearcher, KernelCacheKey,
    RuntimeMode,
};
use atlas_search_ir::{SearchDomain, SearchProgram};
use atlas_search_native::NativeSearcher;

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
