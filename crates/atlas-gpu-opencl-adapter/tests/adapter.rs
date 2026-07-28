//! OpenCL adapter CLI tests.

#[cfg(windows)]
use atlas_gpu_opencl_adapter::opencl_loader_candidates_from_host_roots;
use atlas_gpu_opencl_adapter::{
    opencl_loader_candidates_from_roots, run_cli, AdapterCommand, FeatureReport, LaunchArgs,
    LaunchOutput, Launcher, OpenClLaunchAbi,
};
use atlas_search_gpu::GpuSearcher;
use atlas_search_ir::{SearchOp, SearchProgram};
use std::cell::RefCell;
use std::fs;

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
            hardware: "Intel Arc A770 via OpenCL".to_owned(),
            features: vec!["int64".to_owned()],
        })
    }

    fn compile_check(&self, _source: &str, _output: Option<&str>) -> Result<(), String> {
        Ok(())
    }

    fn launch(&self, args: &LaunchArgs) -> Result<LaunchOutput, String> {
        assert_eq!(args.artifact, "target/atlas-gpu/atlas_search.cl");
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
            hardware: "Fixture OpenCL device".to_owned(),
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
        "target/atlas-gpu/atlas_search.cl".to_owned(),
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

    assert_eq!(args.artifact, "target/atlas-gpu/atlas_search.cl");
    assert_eq!(args.start, 10);
    assert_eq!(args.end, 20);
    assert_eq!(args.max_matches, 3);
    assert_eq!(args.global_size, 256);
    assert_eq!(args.local_size, 64);
}

#[test]
fn parses_compile_check_command() {
    let command = AdapterCommand::parse(&[
        "--compile-check".to_owned(),
        "target/atlas-gpu/atlas_search.cl".to_owned(),
    ])
    .unwrap();

    assert_eq!(
        command,
        AdapterCommand::CompileCheck {
            source: "target/atlas-gpu/atlas_search.cl".to_owned(),
            output: None,
        }
    );
}

#[test]
fn parses_compile_check_source_and_output_command() {
    let command = AdapterCommand::parse(&[
        "--compile-check".to_owned(),
        "target/atlas-gpu/atlas_search.cl".to_owned(),
        "-o".to_owned(),
        "target/atlas-gpu/atlas_search.opencl.bin".to_owned(),
    ])
    .unwrap();

    assert_eq!(
        command,
        AdapterCommand::CompileCheck {
            source: "target/atlas-gpu/atlas_search.cl".to_owned(),
            output: Some("target/atlas-gpu/atlas_search.opencl.bin".to_owned()),
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
        "hardware=Intel Arc A770 via OpenCL\nfeature=int64\nfeature=launchAbiU32\nfeature=launchAbiU64\n"
    );
}

#[test]
fn cli_compile_check_invokes_launcher_backend() {
    let launcher = RecordingLauncher::new();

    let output = run_cli(
        &[
            "--compile-check".to_owned(),
            "target/atlas-gpu/atlas_search.cl".to_owned(),
        ],
        &launcher,
    )
    .unwrap();

    assert_eq!(output, "");
    assert_eq!(
        launcher.compile_checked.borrow().as_slice(),
        &[("target/atlas-gpu/atlas_search.cl".to_owned(), None)]
    );
}

#[test]
#[ignore = "requires a local OpenCL runtime and device"]
fn compile_check_writes_checked_source_artifact() {
    let root = std::env::temp_dir().join(format!("atlas-opencl-artifact-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let source = root.join("atlas_search.cl");
    let artifact = root.join("atlas_search.opencl.bin");
    let source_text = GpuSearcher::compile_opencl(&SearchProgram::try_from_fixture("xor").unwrap());
    fs::write(&source, &source_text).unwrap();
    let source = source.to_string_lossy().into_owned();
    let artifact_text = artifact.to_string_lossy().into_owned();

    atlas_gpu_opencl_adapter::OpenClLauncher
        .compile_check(&source, Some(&artifact_text))
        .unwrap();

    let written = fs::read_to_string(&artifact).unwrap();
    let _ = fs::remove_dir_all(root);
    assert_eq!(written, source_text);
}

#[test]
fn rejects_malformed_or_unbounded_launch_ranges() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.cl".to_owned(),
        "--start".to_owned(),
        "20".to_owned(),
        "--end".to_owned(),
        "10".to_owned(),
        "--max-matches".to_owned(),
        "3".to_owned(),
        "--global-size".to_owned(),
        "256".to_owned(),
        "--local-size".to_owned(),
        "64".to_owned(),
    ])
    .unwrap_err();

    assert!(error.contains("end must be greater than start"));
}

#[test]
fn rejects_global_size_smaller_than_launch_domain() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.cl".to_owned(),
        "--start".to_owned(),
        "10".to_owned(),
        "--end".to_owned(),
        "20".to_owned(),
        "--max-matches".to_owned(),
        "3".to_owned(),
        "--global-size".to_owned(),
        "9".to_owned(),
        "--local-size".to_owned(),
        "1".to_owned(),
    ])
    .unwrap_err();

    assert!(error.contains("global-size must cover launch domain"));
}

#[test]
fn rejects_global_size_that_is_not_multiple_of_local_size() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.cl".to_owned(),
        "--start".to_owned(),
        "10".to_owned(),
        "--end".to_owned(),
        "20".to_owned(),
        "--max-matches".to_owned(),
        "3".to_owned(),
        "--global-size".to_owned(),
        "130".to_owned(),
        "--local-size".to_owned(),
        "64".to_owned(),
    ])
    .unwrap_err();

    assert!(error.contains("global-size must be a multiple of local-size"));
}

#[test]
fn rejects_max_matches_that_exceeds_kernel_uint() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.cl".to_owned(),
        "--start".to_owned(),
        "10".to_owned(),
        "--end".to_owned(),
        "20".to_owned(),
        "--max-matches".to_owned(),
        "4294967296".to_owned(),
        "--global-size".to_owned(),
        "256".to_owned(),
        "--local-size".to_owned(),
        "64".to_owned(),
    ])
    .unwrap_err();

    assert!(error.contains("max-matches exceeds OpenCL uint"));
}

#[test]
fn parses_explicit_u32_launch_abi() {
    let args = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.cl".to_owned(),
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
        "--abi".to_owned(),
        "u32".to_owned(),
    ])
    .unwrap();

    assert_eq!(args.launch_abi, Some(OpenClLaunchAbi::U32));
}

#[test]
fn rejects_missing_explicit_launch_abi_value() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.cl".to_owned(),
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
        "--abi".to_owned(),
    ])
    .unwrap_err();

    assert!(error.contains("missing --abi value"));
}

#[test]
fn cli_emits_match_lines_from_launcher() {
    let output = run_cli(
        &[
            "target/atlas-gpu/atlas_search.cl".to_owned(),
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
fn opencl_loader_candidates_include_sdk_root_loader_directories() {
    let root = std::env::temp_dir().join(format!("atlas-opencl-sdk-{}", std::process::id()));
    let candidates = opencl_loader_candidates_from_roots([root.clone()]);

    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.starts_with(root.join("bin"))),
        "expected candidates to include OpenCL SDK bin directory, got {candidates:?}"
    );
    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.starts_with(root.join("lib"))),
        "expected candidates to include OpenCL SDK lib directory, got {candidates:?}"
    );
    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.starts_with(root.join("lib64"))),
        "expected candidates to include OpenCL SDK lib64 directory, got {candidates:?}"
    );
}

#[cfg(windows)]
#[test]
fn opencl_loader_candidates_include_standard_khronos_sdk_layout() {
    let root = std::env::temp_dir().join(format!("atlas-opencl-standard-{}", std::process::id()));
    let opencl_sdk = root.join("Khronos").join("OpenCL-SDK");
    fs::create_dir_all(opencl_sdk.join("bin")).unwrap();
    let loader = opencl_sdk.join("bin").join("OpenCL.dll");
    fs::write(&loader, []).unwrap();
    let original_program_files = std::env::var_os("ProgramFiles");
    let original_program_files_x86 = std::env::var_os("ProgramFiles(x86)");
    std::env::set_var("ProgramFiles", &root);
    std::env::remove_var("ProgramFiles(x86)");

    let candidates = opencl_loader_candidates_from_host_roots();

    restore_env("ProgramFiles", original_program_files);
    restore_env("ProgramFiles(x86)", original_program_files_x86);
    let _ = fs::remove_dir_all(root);
    assert!(
        candidates.contains(&loader),
        "expected candidates to include standard OpenCL SDK loader, got {candidates:?}"
    );
}

#[test]
#[ignore = "requires a local OpenCL runtime and device"]
fn generated_opencl_kernel_runs_on_device_and_preserves_full_candidates() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let source = GpuSearcher::compile_opencl(&program);
    let output_dir = std::env::temp_dir().join(format!("atlas-opencl-e2e-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let source_path = output_dir.join("atlas_search.cl");
    fs::write(&source_path, source).unwrap();
    let args = LaunchArgs {
        artifact: source_path.to_string_lossy().into_owned(),
        start: 0x50,
        end: 0x160,
        max_matches: 8,
        global_size: 512,
        local_size: 1,
        launch_abi: Some(OpenClLaunchAbi::U32),
    };

    let output = atlas_gpu_opencl_adapter::OpenClLauncher
        .launch(&args)
        .unwrap();

    assert_eq!(output.matches, vec![0x55, 0x155]);
    assert_eq!(output.match_count, 2);
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
#[ignore = "requires a local OpenCL runtime and device"]
fn generated_opencl_dense_kernel_retains_full_device_buffer() {
    let program = SearchProgram::try_from_fixture("dense").unwrap();
    let source = GpuSearcher::compile_opencl(&program);
    let output_dir =
        std::env::temp_dir().join(format!("atlas-opencl-dense-e2e-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let source_path = output_dir.join("atlas_search.cl");
    fs::write(&source_path, source).unwrap();
    let args = LaunchArgs {
        artifact: source_path.to_string_lossy().into_owned(),
        start: 0,
        end: 1500,
        max_matches: 1500,
        global_size: 1536,
        local_size: 1,
        launch_abi: Some(OpenClLaunchAbi::U32),
    };

    let output = atlas_gpu_opencl_adapter::OpenClLauncher
        .launch(&args)
        .unwrap();

    let expected = (0..1500).collect::<Vec<_>>();
    assert_eq!(output.matches, expected);
    assert_eq!(output.match_count, 1500);
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
#[ignore = "requires a local OpenCL runtime and device with int64 support"]
fn generated_opencl_64_bit_kernel_runs_on_device() {
    let program = SearchProgram::new(
        64,
        vec![SearchOp::XorEq {
            mask: 1,
            target: 0x8000_0000_0000_0001,
        }],
    )
    .unwrap();
    let source = GpuSearcher::compile_opencl(&program);
    let output_dir =
        std::env::temp_dir().join(format!("atlas-opencl-64-e2e-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let source_path = output_dir.join("atlas_search.cl");
    fs::write(&source_path, source).unwrap();
    let args = LaunchArgs {
        artifact: source_path.to_string_lossy().into_owned(),
        start: 0x8000_0000_0000_0000,
        end: 0x8000_0000_0000_0002,
        max_matches: 2,
        global_size: 256,
        local_size: 1,
        launch_abi: Some(OpenClLaunchAbi::U64),
    };

    let output = atlas_gpu_opencl_adapter::OpenClLauncher
        .launch(&args)
        .unwrap();

    assert_eq!(output.matches, vec![0x8000_0000_0000_0000]);
    assert_eq!(output.match_count, 1);
    let _ = fs::remove_dir_all(output_dir);
}
