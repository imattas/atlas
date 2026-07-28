//! Vulkan adapter CLI tests.

use atlas_gpu_vulkan_adapter::{
    run_cli, AdapterCommand, LaunchArgs, Launcher, VulkanSpirvLauncher,
};
use atlas_search_gpu::GpuSearcher;
use atlas_search_ir::SearchProgram;
use std::cell::RefCell;
use std::fs;

#[derive(Debug, Clone, Copy)]
struct FixtureLauncher;

impl Launcher for FixtureLauncher {
    fn compile_check(&self, _spirv: &str) -> Result<(), String> {
        Ok(())
    }

    fn launch(&self, args: &LaunchArgs) -> Result<Vec<u64>, String> {
        assert_eq!(args.artifact, "target/atlas-gpu/atlas_search.spv");
        assert_eq!(args.start, 10);
        assert_eq!(args.end, 20);
        assert_eq!(args.max_matches, 3);
        assert_eq!(args.global_size, 256);
        assert_eq!(args.local_size, 256);
        Ok(vec![11, 13, 17])
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
    fn compile_check(&self, spirv: &str) -> Result<(), String> {
        self.compile_checked.borrow_mut().push(spirv.to_owned());
        Ok(())
    }

    fn launch(&self, _args: &LaunchArgs) -> Result<Vec<u64>, String> {
        Ok(Vec::new())
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
            spirv: "target/atlas-gpu/atlas_search.spv".to_owned(),
        }
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
        ["target/atlas-gpu/atlas_search.spv"]
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

    assert_eq!(output, "match=11\nmatch=13\nmatch=17\n");
}

#[test]
fn compile_check_rejects_non_spirv_artifact() {
    let output_dir = std::env::temp_dir().join(format!("atlas-vulkan-spv-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let spv_path = output_dir.join("bad.spv");
    fs::write(&spv_path, b"not spirv").unwrap();

    let error = VulkanSpirvLauncher
        .compile_check(&spv_path.to_string_lossy())
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
        .compile_check(&source_path.to_string_lossy())
        .unwrap();

    let _ = fs::remove_dir_all(output_dir);
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
    };

    let matches = VulkanSpirvLauncher.launch(&args).unwrap();

    assert_eq!(matches, vec![0x55, 0x155]);
    let _ = fs::remove_dir_all(output_dir);
}
