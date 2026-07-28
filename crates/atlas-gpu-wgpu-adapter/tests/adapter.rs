//! WGPU adapter CLI tests.

use atlas_gpu_wgpu_adapter::{
    run_cli, validate_wgsl_launch_shape, AdapterCommand, FeatureReport, LaunchArgs, LaunchOutput,
    Launcher,
};
use atlas_search_gpu::GpuSearcher;
use atlas_search_ir::SearchProgram;
use std::cell::RefCell;
use std::fs;

#[derive(Debug, Clone, Copy)]
struct FixtureLauncher;

impl Launcher for FixtureLauncher {
    fn features(&self) -> Result<FeatureReport, String> {
        Ok(FeatureReport {
            hardware: "NVIDIA RTX 4090 via WGPU".to_owned(),
            features: vec!["launchAbiU32".to_owned()],
        })
    }

    fn compile_check(&self, _source: &str, _output: Option<&str>) -> Result<(), String> {
        Ok(())
    }

    fn launch(&self, args: &LaunchArgs) -> Result<LaunchOutput, String> {
        assert_eq!(args.artifact, "target/atlas-gpu/atlas_search.wgsl");
        assert_eq!(args.start, 10);
        assert_eq!(args.end, 20);
        assert_eq!(args.max_matches, 3);
        assert_eq!(args.global_size, 256);
        assert_eq!(args.local_size, 64);
        Ok(LaunchOutput {
            matches: vec![11, 13, 17],
            match_count: 5,
        })
    }
}

#[derive(Debug)]
struct RecordingLauncher {
    compile_checked: RefCell<Vec<(String, Option<String>)>>,
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
            hardware: "Recording WGPU adapter".to_owned(),
            features: Vec::new(),
        })
    }

    fn compile_check(&self, source: &str, output: Option<&str>) -> Result<(), String> {
        self.compile_checked
            .borrow_mut()
            .push((source.to_owned(), output.map(str::to_owned)));
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
        "target/atlas-gpu/atlas_search.wgsl".to_owned(),
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
    .unwrap();

    assert_eq!(args.artifact, "target/atlas-gpu/atlas_search.wgsl");
    assert_eq!(args.start, 10);
    assert_eq!(args.end, 20);
    assert_eq!(args.max_matches, 3);
    assert_eq!(args.global_size, 256);
    assert_eq!(args.local_size, 64);
}

#[test]
fn parses_compile_check_source_and_output_command() {
    let command = AdapterCommand::parse(&[
        "--compile-check".to_owned(),
        "target/atlas-gpu/atlas_search.wgsl".to_owned(),
        "-o".to_owned(),
        "target/atlas-gpu/checked.wgsl".to_owned(),
    ])
    .unwrap();

    assert_eq!(
        command,
        AdapterCommand::CompileCheck {
            source: "target/atlas-gpu/atlas_search.wgsl".to_owned(),
            output: Some("target/atlas-gpu/checked.wgsl".to_owned()),
        }
    );
}

#[test]
fn cli_features_emits_launcher_capabilities() {
    let output = run_cli(&["--features".to_owned()], &FixtureLauncher).unwrap();

    assert_eq!(
        output,
        "hardware=NVIDIA RTX 4090 via WGPU\nfeature=launchAbiU32\n"
    );
}

#[test]
fn cli_compile_check_invokes_launcher_backend() {
    let launcher = RecordingLauncher::new();

    let output = run_cli(
        &[
            "--compile-check".to_owned(),
            "target/atlas-gpu/atlas_search.wgsl".to_owned(),
            "-o".to_owned(),
            "target/atlas-gpu/checked.wgsl".to_owned(),
        ],
        &launcher,
    )
    .unwrap();

    assert_eq!(output, "");
    assert_eq!(
        launcher.compile_checked.borrow().as_slice(),
        &[(
            "target/atlas-gpu/atlas_search.wgsl".to_owned(),
            Some("target/atlas-gpu/checked.wgsl".to_owned())
        )]
    );
}

#[test]
fn cli_emits_match_lines_from_launcher() {
    let output = run_cli(
        &[
            "target/atlas-gpu/atlas_search.wgsl".to_owned(),
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
        ],
        &FixtureLauncher,
    )
    .unwrap();

    assert_eq!(output, "match_count=5\nmatch=11\nmatch=13\nmatch=17\n");
}

#[test]
fn compile_check_accepts_generated_wgsl_and_writes_checked_artifact() {
    let root = std::env::temp_dir().join(format!("atlas-wgpu-artifact-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let source = root.join("atlas_search.wgsl");
    let checked = root.join("checked.wgsl");
    let source_text = GpuSearcher::compile_wgsl(&SearchProgram::try_from_fixture("xor").unwrap());
    fs::write(&source, &source_text).unwrap();

    atlas_gpu_wgpu_adapter::WgpuLauncher
        .compile_check(&source.to_string_lossy(), Some(&checked.to_string_lossy()))
        .unwrap();

    let written = fs::read_to_string(&checked).unwrap();
    let _ = fs::remove_dir_all(root);
    assert_eq!(written, source_text);
}

#[test]
fn compile_check_rejects_invalid_wgsl() {
    let root = std::env::temp_dir().join(format!("atlas-wgpu-invalid-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let source = root.join("bad.wgsl");
    fs::write(&source, "not valid wgsl").unwrap();

    let error = atlas_gpu_wgpu_adapter::WgpuLauncher
        .compile_check(&source.to_string_lossy(), None)
        .unwrap_err();

    let _ = fs::remove_dir_all(root);
    assert!(error.contains("invalid WGSL"), "{error}");
}

#[test]
fn launch_shape_validation_rejects_local_size_that_does_not_match_wgsl() {
    let source_text = GpuSearcher::compile_wgsl(&SearchProgram::try_from_fixture("xor").unwrap());

    validate_wgsl_launch_shape(&source_text, 256).unwrap();
    let error = validate_wgsl_launch_shape(&source_text, 128).unwrap_err();

    assert!(error.contains("WGPU local-size mismatch"), "{error}");
}

#[test]
#[ignore = "requires a wgpu-compatible adapter"]
fn generated_wgpu_kernel_runs_on_device_and_preserves_full_candidates() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let source = GpuSearcher::compile_wgsl(&program);
    let output_dir = std::env::temp_dir().join(format!("atlas-wgpu-e2e-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let source_path = output_dir.join("atlas_search.wgsl");
    fs::write(&source_path, source).unwrap();
    let args = LaunchArgs {
        artifact: source_path.to_string_lossy().into_owned(),
        start: 0x50,
        end: 0x160,
        max_matches: 8,
        global_size: 512,
        local_size: 256,
    };

    let output = atlas_gpu_wgpu_adapter::WgpuLauncher.launch(&args).unwrap();

    assert_eq!(output.matches, vec![0x55, 0x155]);
    assert_eq!(output.match_count, 2);
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
#[ignore = "requires a wgpu-compatible adapter"]
fn generated_wgpu_dense_kernel_retains_full_device_buffer() {
    let program = SearchProgram::try_from_fixture("dense").unwrap();
    let source = GpuSearcher::compile_wgsl(&program);
    let output_dir =
        std::env::temp_dir().join(format!("atlas-wgpu-dense-e2e-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let source_path = output_dir.join("atlas_search.wgsl");
    fs::write(&source_path, source).unwrap();
    let args = LaunchArgs {
        artifact: source_path.to_string_lossy().into_owned(),
        start: 0,
        end: 1500,
        max_matches: 1500,
        global_size: 1536,
        local_size: 256,
    };

    let output = atlas_gpu_wgpu_adapter::WgpuLauncher.launch(&args).unwrap();

    let expected = (0..1500).collect::<Vec<_>>();
    assert_eq!(output.matches, expected);
    assert_eq!(output.match_count, 1500);
    let _ = fs::remove_dir_all(output_dir);
}
