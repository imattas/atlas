//! GPU boundary and differential tests.

use atlas_scheduler::CancellationToken;
use atlas_search_gpu::{GpuSdk, GpuSdkDetector, GpuSdkPlan, GpuSearcher, KernelCacheKey};
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
