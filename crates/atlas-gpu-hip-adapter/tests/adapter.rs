//! HIP adapter CLI tests.

use atlas_gpu_hip_adapter::{
    hip_runtime_library_candidates_from_roots, run_cli, AdapterCommand, HipModuleLauncher,
    LaunchArgs, Launcher,
};
use atlas_search_gpu::GpuSearcher;
use atlas_search_ir::SearchProgram;
use std::cell::RefCell;
use std::fs;
use std::process::Command;

#[derive(Debug, Clone, Copy)]
struct FixtureLauncher;

impl Launcher for FixtureLauncher {
    fn compile_check(&self, _artifact: &str) -> Result<(), String> {
        Ok(())
    }

    fn launch(&self, args: &LaunchArgs) -> Result<Vec<u64>, String> {
        assert_eq!(args.artifact, "target/atlas-gpu/atlas_search.hsaco");
        assert_eq!(args.start, 10);
        assert_eq!(args.end, 20);
        assert_eq!(args.max_matches, 3);
        assert_eq!(args.global_size, 256);
        assert_eq!(args.local_size, 64);
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
    fn compile_check(&self, artifact: &str) -> Result<(), String> {
        self.compile_checked.borrow_mut().push(artifact.to_owned());
        Ok(())
    }

    fn launch(&self, _args: &LaunchArgs) -> Result<Vec<u64>, String> {
        Ok(Vec::new())
    }
}

#[test]
fn parses_launch_protocol_arguments() {
    let args = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.hsaco".to_owned(),
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

    assert_eq!(args.artifact, "target/atlas-gpu/atlas_search.hsaco");
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
        "target/atlas-gpu/atlas_search.hsaco".to_owned(),
    ])
    .unwrap();

    assert_eq!(
        command,
        AdapterCommand::CompileCheck {
            artifact: "target/atlas-gpu/atlas_search.hsaco".to_owned(),
        }
    );
}

#[test]
fn cli_compile_check_invokes_launcher_backend() {
    let launcher = RecordingLauncher::new();

    let output = run_cli(
        &[
            "--compile-check".to_owned(),
            "target/atlas-gpu/atlas_search.hsaco".to_owned(),
        ],
        &launcher,
    )
    .unwrap();

    assert_eq!(output, "");
    assert_eq!(
        launcher.compile_checked.borrow().as_slice(),
        ["target/atlas-gpu/atlas_search.hsaco"]
    );
}

#[test]
fn rejects_malformed_or_unbounded_launch_ranges() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.hsaco".to_owned(),
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
fn cli_emits_match_lines_from_launcher() {
    let output = run_cli(
        &[
            "target/atlas-gpu/atlas_search.hsaco".to_owned(),
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

    assert_eq!(output, "match=11\nmatch=13\nmatch=17\n");
}

#[test]
fn compile_check_rejects_missing_code_object_artifact() {
    let missing = std::env::temp_dir()
        .join(format!("atlas-missing-hip-{}.hsaco", std::process::id()))
        .to_string_lossy()
        .into_owned();

    let error = HipModuleLauncher.compile_check(&missing).unwrap_err();

    assert!(error.contains("cannot read HIP code object"));
}

#[test]
fn hip_runtime_library_candidates_include_sdk_root_library_directories() {
    let hip_root =
        std::env::temp_dir().join(format!("atlas-hip-runtime-root-{}", std::process::id()));
    let candidates = hip_runtime_library_candidates_from_roots([hip_root.clone()]);

    assert!(candidates
        .iter()
        .any(|path| path.starts_with(hip_root.join("bin"))));
    assert!(candidates
        .iter()
        .any(|path| path.starts_with(hip_root.join("lib"))));
    assert!(candidates
        .iter()
        .any(|path| path.starts_with(hip_root.join("lib64"))));
}

#[test]
#[ignore = "requires hipcc, HIP runtime, and an AMD HIP-capable device"]
fn generated_hip_kernel_runs_on_device_and_preserves_full_candidates() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let source = GpuSearcher::compile_hip(&program);
    let output_dir = std::env::temp_dir().join(format!("atlas-hip-e2e-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let source_path = output_dir.join("atlas_search.hip");
    let code_object_path = output_dir.join("atlas_search.hsaco");
    fs::write(&source_path, source).unwrap();
    let arch = detect_hip_arch().unwrap_or_else(|| "gfx1100".to_owned());

    let compile = Command::new("hipcc")
        .arg("--genco")
        .arg("-O2")
        .arg(format!("--offload-arch={arch}"))
        .arg(&source_path)
        .arg("-o")
        .arg(&code_object_path)
        .output()
        .unwrap();
    assert!(
        compile.status.success() && code_object_path.exists(),
        "hipcc failed: stdout={} stderr={}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );
    let args = LaunchArgs {
        artifact: code_object_path.to_string_lossy().into_owned(),
        start: 0x50,
        end: 0x160,
        max_matches: 8,
        global_size: 512,
        local_size: 64,
    };

    let matches = HipModuleLauncher.launch(&args).unwrap();

    assert_eq!(matches, vec![0x55, 0x155]);
    let _ = fs::remove_dir_all(output_dir);
}

fn detect_hip_arch() -> Option<String> {
    let output = Command::new("hipInfo").output().ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().find_map(|line| {
        line.split_once("gcnArchName:")
            .map(|(_, arch)| arch.trim().to_owned())
            .filter(|arch| !arch.is_empty())
    })
}
