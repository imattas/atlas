//! Vulkan adapter CLI tests.

#[cfg(windows)]
use atlas_gpu_vulkan_adapter::vulkan_loader_candidates_from_host_roots;
use atlas_gpu_vulkan_adapter::{
    run_cli, vulkan_loader_candidates_from_roots, AdapterCommand, FeatureReport, LaunchArgs,
    LaunchOutput, Launcher, VulkanLaunchAbi, VulkanSpirvLauncher,
};
use atlas_search_gpu::GpuSearcher;
use atlas_search_ir::{SearchOp, SearchProgram};
use std::cell::RefCell;
use std::fs;
#[cfg(windows)]
use std::sync::{Mutex, OnceLock};

#[cfg(windows)]
fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(windows)]
fn restore_env(name: &str, original: Option<std::ffi::OsString>) {
    if let Some(value) = original {
        std::env::set_var(name, value);
    } else {
        std::env::remove_var(name);
    }
}

#[derive(Debug, Clone, Copy)]
struct FixtureLauncher;

impl Launcher for FixtureLauncher {
    fn features(&self) -> Result<FeatureReport, String> {
        Ok(FeatureReport {
            hardware: "AMD Radeon RX 7900 XTX via Vulkan".to_owned(),
            features: vec!["shaderInt64".to_owned()],
        })
    }

    fn compile_check(&self, _input: &str, _output: Option<&str>) -> Result<(), String> {
        Ok(())
    }

    fn launch(&self, args: &LaunchArgs) -> Result<LaunchOutput, String> {
        assert_eq!(args.artifact, "target/atlas-gpu/atlas_search.spv");
        assert_eq!(args.start, 10);
        assert_eq!(args.end, 20);
        assert_eq!(args.max_matches, 3);
        assert_eq!(args.global_size, 256);
        assert_eq!(args.local_size, 256);
        Ok(LaunchOutput {
            matches: vec![11, 13, 17],
            match_count: 5,
        })
    }
}

#[derive(Debug)]
struct RecordingLauncher {
    compile_checked: RefCell<Vec<String>>,
}

impl RecordingLauncher {
    fn new() -> Self {
        Self {
            compile_checked: RefCell::new(Vec::new()),
        }
    }
}

impl Launcher for RecordingLauncher {
    fn features(&self) -> Result<FeatureReport, String> {
        Ok(FeatureReport {
            hardware: "Fixture Vulkan device".to_owned(),
            features: Vec::new(),
        })
    }

    fn compile_check(&self, input: &str, output: Option<&str>) -> Result<(), String> {
        self.compile_checked
            .borrow_mut()
            .push(format!("{input}->{}", output.unwrap_or("")));
        Ok(())
    }

    fn launch(&self, _args: &LaunchArgs) -> Result<LaunchOutput, String> {
        Ok(LaunchOutput {
            matches: Vec::new(),
            match_count: 0,
        })
    }
}

#[test]
fn parses_launch_protocol_arguments() {
    let args = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.spv".to_owned(),
        "--start".to_owned(),
        "10".to_owned(),
        "--end".to_owned(),
        "20".to_owned(),
        "--max-matches".to_owned(),
        "3".to_owned(),
        "--global-size".to_owned(),
        "256".to_owned(),
        "--local-size".to_owned(),
        "256".to_owned(),
    ])
    .unwrap();

    assert_eq!(args.artifact, "target/atlas-gpu/atlas_search.spv");
    assert_eq!(args.start, 10);
    assert_eq!(args.end, 20);
    assert_eq!(args.max_matches, 3);
    assert_eq!(args.global_size, 256);
    assert_eq!(args.local_size, 256);
}

#[test]
fn parses_compile_check_command() {
    let command = AdapterCommand::parse(&[
        "--compile-check".to_owned(),
        "target/atlas-gpu/atlas_search.spv".to_owned(),
    ])
    .unwrap();

    assert_eq!(
        command,
        AdapterCommand::CompileCheck {
            input: "target/atlas-gpu/atlas_search.spv".to_owned(),
            output: None,
        }
    );
}

#[test]
fn parses_features_command() {
    let command = AdapterCommand::parse(&["--features".to_owned()]).unwrap();

    assert_eq!(command, AdapterCommand::Features);
}

#[test]
fn cli_features_emits_launcher_capabilities() {
    let output = run_cli(&["--features".to_owned()], &FixtureLauncher).unwrap();

    assert_eq!(
        output,
        "hardware=AMD Radeon RX 7900 XTX via Vulkan\nfeature=shaderInt64\nfeature=launchAbiU32\nfeature=launchAbiU64\n"
    );
}

#[test]
fn cli_compile_check_invokes_launcher_backend() {
    let launcher = RecordingLauncher::new();

    let output = run_cli(
        &[
            "--compile-check".to_owned(),
            "target/atlas-gpu/atlas_search.spv".to_owned(),
        ],
        &launcher,
    )
    .unwrap();

    assert_eq!(output, "");
    assert_eq!(
        launcher.compile_checked.borrow().as_slice(),
        ["target/atlas-gpu/atlas_search.spv->"]
    );
}

#[test]
fn rejects_malformed_or_unbounded_launch_ranges() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.spv".to_owned(),
        "--start".to_owned(),
        "20".to_owned(),
        "--end".to_owned(),
        "10".to_owned(),
        "--max-matches".to_owned(),
        "3".to_owned(),
        "--global-size".to_owned(),
        "256".to_owned(),
        "--local-size".to_owned(),
        "256".to_owned(),
    ])
    .unwrap_err();

    assert!(error.contains("end must be greater than start"));
}

#[test]
fn rejects_global_size_smaller_than_launch_domain() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.spv".to_owned(),
        "--start".to_owned(),
        "10".to_owned(),
        "--end".to_owned(),
        "20".to_owned(),
        "--max-matches".to_owned(),
        "3".to_owned(),
        "--global-size".to_owned(),
        "9".to_owned(),
        "--local-size".to_owned(),
        "256".to_owned(),
    ])
    .unwrap_err();

    assert!(error.contains("global-size must cover launch domain"));
}

#[test]
fn rejects_local_size_that_does_not_match_generated_shader() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.spv".to_owned(),
        "--start".to_owned(),
        "10".to_owned(),
        "--end".to_owned(),
        "20".to_owned(),
        "--max-matches".to_owned(),
        "3".to_owned(),
        "--global-size".to_owned(),
        "256".to_owned(),
        "--local-size".to_owned(),
        "64".to_owned(),
    ])
    .unwrap_err();

    assert!(error.contains("Vulkan shader local-size must be 256"));
}

#[test]
fn rejects_max_matches_that_exceeds_kernel_uint() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.spv".to_owned(),
        "--start".to_owned(),
        "10".to_owned(),
        "--end".to_owned(),
        "20".to_owned(),
        "--max-matches".to_owned(),
        "4294967296".to_owned(),
        "--global-size".to_owned(),
        "256".to_owned(),
        "--local-size".to_owned(),
        "256".to_owned(),
    ])
    .unwrap_err();

    assert!(error.contains("max-matches exceeds Vulkan uint"));
}

#[test]
fn rejects_dispatch_group_count_that_exceeds_vulkan_uint() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.spv".to_owned(),
        "--start".to_owned(),
        "0".to_owned(),
        "--end".to_owned(),
        "1".to_owned(),
        "--max-matches".to_owned(),
        "1".to_owned(),
        "--global-size".to_owned(),
        "1099511627776".to_owned(),
        "--local-size".to_owned(),
        "256".to_owned(),
    ])
    .unwrap_err();

    assert!(error.contains("dispatch group count exceeds Vulkan uint"));
}

#[test]
fn parses_explicit_u32_launch_abi() {
    let args = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.spv".to_owned(),
        "--start".to_owned(),
        "10".to_owned(),
        "--end".to_owned(),
        "20".to_owned(),
        "--max-matches".to_owned(),
        "3".to_owned(),
        "--global-size".to_owned(),
        "256".to_owned(),
        "--local-size".to_owned(),
        "256".to_owned(),
        "--abi".to_owned(),
        "u32".to_owned(),
    ])
    .unwrap();

    assert_eq!(args.launch_abi, Some(VulkanLaunchAbi::U32));
}

#[test]
fn rejects_missing_explicit_launch_abi_value() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.spv".to_owned(),
        "--start".to_owned(),
        "10".to_owned(),
        "--end".to_owned(),
        "20".to_owned(),
        "--max-matches".to_owned(),
        "3".to_owned(),
        "--global-size".to_owned(),
        "256".to_owned(),
        "--local-size".to_owned(),
        "256".to_owned(),
        "--abi".to_owned(),
    ])
    .unwrap_err();

    assert!(error.contains("missing --abi value"));
}

#[test]
fn cli_emits_match_lines_from_launcher() {
    let output = run_cli(
        &[
            "target/atlas-gpu/atlas_search.spv".to_owned(),
            "--start".to_owned(),
            "10".to_owned(),
            "--end".to_owned(),
            "20".to_owned(),
            "--max-matches".to_owned(),
            "3".to_owned(),
            "--global-size".to_owned(),
            "256".to_owned(),
            "--local-size".to_owned(),
            "256".to_owned(),
        ],
        &FixtureLauncher,
    )
    .unwrap();

    assert_eq!(output, "match_count=5\nmatch=11\nmatch=13\nmatch=17\n");
}

#[test]
fn compile_check_rejects_non_spirv_artifact() {
    let output_dir = std::env::temp_dir().join(format!("atlas-vulkan-spv-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let spv_path = output_dir.join("bad.spv");
    fs::write(&spv_path, b"not spirv").unwrap();

    let error = VulkanSpirvLauncher
        .compile_check(&spv_path.to_string_lossy(), None)
        .unwrap_err();

    assert!(error.contains("invalid SPIR-V magic"));
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
fn compile_check_accepts_generated_glsl_source_without_external_glslc() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let source = GpuSearcher::compile_vulkan_glsl(&program);
    let output_dir =
        std::env::temp_dir().join(format!("atlas-vulkan-source-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let source_path = output_dir.join("atlas_search.comp");
    fs::write(&source_path, source).unwrap();

    VulkanSpirvLauncher
        .compile_check(&source_path.to_string_lossy(), None)
        .unwrap();

    let _ = fs::remove_dir_all(output_dir);
}

#[test]
fn compile_check_writes_spirv_artifact_when_output_is_requested() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let source = GpuSearcher::compile_vulkan_glsl(&program);
    let output_dir =
        std::env::temp_dir().join(format!("atlas-vulkan-cache-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let source_path = output_dir.join("atlas_search.comp");
    let spirv_path = output_dir.join("atlas_search.spv");
    fs::write(&source_path, source).unwrap();

    run_cli(
        &[
            "--compile-check".to_owned(),
            source_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            spirv_path.to_string_lossy().into_owned(),
        ],
        &VulkanSpirvLauncher,
    )
    .unwrap();

    let bytes = fs::read(&spirv_path).unwrap();
    assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07]);
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
fn vulkan_loader_candidates_include_sdk_root_loader_directories() {
    let root = std::env::temp_dir().join(format!("atlas-vulkan-sdk-{}", std::process::id()));
    let candidates = vulkan_loader_candidates_from_roots([root.clone()]);

    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.starts_with(root.join("Bin"))),
        "expected candidates to include Vulkan SDK Bin directory, got {candidates:?}"
    );
    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.starts_with(root.join("lib"))),
        "expected candidates to include Vulkan SDK lib directory, got {candidates:?}"
    );
    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.starts_with(root.join("lib64"))),
        "expected candidates to include Vulkan SDK lib64 directory, got {candidates:?}"
    );
}

#[cfg(windows)]
#[test]
fn vulkan_loader_candidates_include_standard_systemdrive_sdk_layout() {
    let _env_guard = env_lock();
    let root = std::env::temp_dir().join(format!("atlas-vulkan-standard-{}", std::process::id()));
    let vulkan_sdk = root.join("VulkanSDK").join("1.3.290.0");
    fs::create_dir_all(vulkan_sdk.join("Bin")).unwrap();
    let loader = vulkan_sdk.join("Bin").join("vulkan-1.dll");
    fs::write(&loader, []).unwrap();
    let original_system_drive = std::env::var_os("SystemDrive");
    std::env::set_var("SystemDrive", &root);

    let candidates = vulkan_loader_candidates_from_host_roots();

    restore_env("SystemDrive", original_system_drive);
    let _ = fs::remove_dir_all(root);
    assert!(
        candidates.contains(&loader),
        "expected candidates to include standard Vulkan SDK loader, got {candidates:?}"
    );
}

#[cfg(windows)]
#[test]
fn vulkan_loader_candidates_prefer_newest_standard_sdk_loader() {
    let _env_guard = env_lock();
    let root = std::env::temp_dir().join(format!(
        "atlas-vulkan-standard-order-{}",
        std::process::id()
    ));
    let vulkan_base = root.join("VulkanSDK");
    let base_loader = vulkan_base.join("Bin").join("vulkan-1.dll");
    let old_loader = vulkan_base
        .join("1.2.198.0")
        .join("Bin")
        .join("vulkan-1.dll");
    let new_loader = vulkan_base
        .join("1.3.290.0")
        .join("Bin")
        .join("vulkan-1.dll");
    for loader in [&base_loader, &old_loader, &new_loader] {
        fs::create_dir_all(loader.parent().unwrap()).unwrap();
        fs::write(loader, []).unwrap();
    }
    let original_system_drive = std::env::var_os("SystemDrive");
    std::env::set_var("SystemDrive", &root);

    let candidates = vulkan_loader_candidates_from_host_roots();

    restore_env("SystemDrive", original_system_drive);
    let _ = fs::remove_dir_all(root);
    assert_eq!(
        candidates.first(),
        Some(&new_loader),
        "expected newest Vulkan SDK loader first, got {candidates:?}"
    );
}

#[test]
#[ignore = "requires Vulkan runtime and a Vulkan compute-capable device"]
fn generated_vulkan_kernel_runs_on_device_and_preserves_full_candidates() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let source = GpuSearcher::compile_vulkan_glsl(&program);
    let output_dir = std::env::temp_dir().join(format!("atlas-vulkan-e2e-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let source_path = output_dir.join("atlas_search.comp");
    fs::write(&source_path, source).unwrap();

    let args = LaunchArgs {
        artifact: source_path.to_string_lossy().into_owned(),
        start: 0x50,
        end: 0x160,
        max_matches: 8,
        global_size: 512,
        local_size: 256,
        launch_abi: Some(VulkanLaunchAbi::U32),
    };

    let output = VulkanSpirvLauncher.launch(&args).unwrap();

    assert_eq!(output.matches, vec![0x55, 0x155]);
    assert_eq!(output.match_count, 2);
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
#[ignore = "requires Vulkan runtime and a Vulkan compute-capable device"]
fn generated_vulkan_dense_kernel_retains_full_device_buffer() {
    let program = SearchProgram::try_from_fixture("dense").unwrap();
    let source = GpuSearcher::compile_vulkan_glsl(&program);
    let output_dir =
        std::env::temp_dir().join(format!("atlas-vulkan-dense-e2e-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let source_path = output_dir.join("atlas_search.comp");
    fs::write(&source_path, source).unwrap();

    let args = LaunchArgs {
        artifact: source_path.to_string_lossy().into_owned(),
        start: 0,
        end: 1500,
        max_matches: 1500,
        global_size: 1536,
        local_size: 256,
        launch_abi: Some(VulkanLaunchAbi::U32),
    };

    let output = VulkanSpirvLauncher.launch(&args).unwrap();

    let expected = (0..1500).collect::<Vec<_>>();
    assert_eq!(output.matches, expected);
    assert_eq!(output.match_count, 1500);
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
#[ignore = "requires Vulkan runtime with shaderInt64 support and a Vulkan compute-capable device"]
fn generated_vulkan_64_bit_kernel_runs_on_device() {
    let program = SearchProgram::new(
        64,
        vec![SearchOp::XorEq {
            mask: 1,
            target: 0x8000_0000_0000_0001,
        }],
    )
    .unwrap();
    let source = GpuSearcher::compile_vulkan_glsl(&program);
    let output_dir =
        std::env::temp_dir().join(format!("atlas-vulkan-64-e2e-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let source_path = output_dir.join("atlas_search.comp");
    fs::write(&source_path, source).unwrap();

    let args = LaunchArgs {
        artifact: source_path.to_string_lossy().into_owned(),
        start: 0x8000_0000_0000_0000,
        end: 0x8000_0000_0000_0002,
        max_matches: 2,
        global_size: 256,
        local_size: 256,
        launch_abi: Some(VulkanLaunchAbi::U64),
    };

    let output = VulkanSpirvLauncher.launch(&args).unwrap();

    assert_eq!(output.matches, vec![0x8000_0000_0000_0000]);
    assert_eq!(output.match_count, 1);
    let _ = fs::remove_dir_all(output_dir);
}
