//! HIP adapter CLI tests.

use atlas_gpu_hip_adapter::{
    hip_runtime_library_candidates_from_host_roots, hip_runtime_library_candidates_from_roots,
    run_cli, AdapterCommand, HipModuleLauncher, LaunchArgs, Launcher,
};
use atlas_search_gpu::GpuSearcher;
use atlas_search_ir::SearchProgram;
use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(windows)]
static WINDOWS_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    fn compile_check(&self, _artifact: &str, _output: Option<&str>) -> Result<(), String> {
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
    fn compile_check(&self, artifact: &str, output: Option<&str>) -> Result<(), String> {
        self.compile_checked
            .borrow_mut()
            .push((artifact.to_owned(), output.map(str::to_owned)));
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
            input: "target/atlas-gpu/atlas_search.hsaco".to_owned(),
            output: None,
        }
    );
}

#[test]
fn parses_compile_check_source_and_output_command() {
    let command = AdapterCommand::parse(&[
        "--compile-check".to_owned(),
        "target/atlas-gpu/atlas_search.hip".to_owned(),
        "-o".to_owned(),
        "target/atlas-gpu/atlas_search.hsaco".to_owned(),
    ])
    .unwrap();

    assert_eq!(
        command,
        AdapterCommand::CompileCheck {
            input: "target/atlas-gpu/atlas_search.hip".to_owned(),
            output: Some("target/atlas-gpu/atlas_search.hsaco".to_owned()),
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
        &[("target/atlas-gpu/atlas_search.hsaco".to_owned(), None)]
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

    let error = HipModuleLauncher.compile_check(&missing, None).unwrap_err();

    assert!(error.contains("cannot read HIP code object"));
}

#[test]
fn compile_check_writes_code_object_artifact_from_hip_source() {
    #[cfg(windows)]
    let _env_lock = WINDOWS_ENV_LOCK.lock().unwrap();
    let root = std::env::temp_dir().join(format!("atlas-hip-fake-hipcc-{}", std::process::id()));
    let bin_dir = root.join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    let hipcc = write_fake_hipcc(&bin_dir);
    let source = root.join("atlas_search.hip");
    let artifact = root.join("atlas_search.hsaco");
    fs::write(&source, "extern \"C\" __global__ void atlas_search() {}\n").unwrap();
    let original_path = std::env::var_os("PATH");
    std::env::set_var("PATH", &bin_dir);
    let source = source.to_string_lossy().into_owned();
    let artifact_text = artifact.to_string_lossy().into_owned();

    HipModuleLauncher
        .compile_check(&source, Some(&artifact_text))
        .unwrap();

    assert!(artifact.exists());
    restore_env("PATH", original_path);
    let _ = fs::remove_file(hipcc);
    let _ = fs::remove_dir_all(root);
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

#[cfg(windows)]
#[test]
fn hip_runtime_library_candidates_include_standard_rocm_layout() {
    let _env_lock = WINDOWS_ENV_LOCK.lock().unwrap();
    let root = std::env::temp_dir().join(format!("atlas-hip-standard-{}", std::process::id()));
    let rocm_root = root.join("AMD").join("ROCm").join("6.1");
    fs::create_dir_all(rocm_root.join("bin")).unwrap();
    let runtime = rocm_root.join("bin").join("amdhip64.dll");
    fs::write(&runtime, []).unwrap();
    let original_program_files = std::env::var_os("ProgramFiles");
    let original_program_files_x86 = std::env::var_os("ProgramFiles(x86)");
    std::env::set_var("ProgramFiles", &root);
    std::env::remove_var("ProgramFiles(x86)");

    let candidates = hip_runtime_library_candidates_from_host_roots();

    restore_env("ProgramFiles", original_program_files);
    restore_env("ProgramFiles(x86)", original_program_files_x86);
    let _ = fs::remove_dir_all(root);
    assert!(
        candidates.contains(&runtime),
        "expected candidates to include standard ROCm HIP runtime, got {candidates:?}"
    );
}

#[cfg(windows)]
#[test]
fn hip_runtime_library_candidates_prefer_newest_standard_rocm_runtime() {
    let _env_lock = WINDOWS_ENV_LOCK.lock().unwrap();
    let root =
        std::env::temp_dir().join(format!("atlas-hip-standard-order-{}", std::process::id()));
    let rocm_base = root.join("AMD").join("ROCm");
    let base_runtime = rocm_base.join("bin").join("amdhip64.dll");
    let old_runtime = rocm_base.join("5.0").join("bin").join("amdhip64.dll");
    let new_runtime = rocm_base.join("6.1").join("bin").join("amdhip64.dll");
    for runtime in [&base_runtime, &old_runtime, &new_runtime] {
        fs::create_dir_all(runtime.parent().unwrap()).unwrap();
        fs::write(runtime, []).unwrap();
    }
    let original_program_files = std::env::var_os("ProgramFiles");
    let original_program_files_x86 = std::env::var_os("ProgramFiles(x86)");
    std::env::set_var("ProgramFiles", &root);
    std::env::remove_var("ProgramFiles(x86)");

    let candidates = hip_runtime_library_candidates_from_host_roots();

    restore_env("ProgramFiles", original_program_files);
    restore_env("ProgramFiles(x86)", original_program_files_x86);
    let _ = fs::remove_dir_all(root);
    assert_eq!(
        candidates.first(),
        Some(&new_runtime),
        "expected newest ROCm runtime first, got {candidates:?}"
    );
}

#[cfg(windows)]
#[test]
fn hip_runtime_library_candidates_do_not_emit_duplicate_bin_segments() {
    let _env_lock = WINDOWS_ENV_LOCK.lock().unwrap();
    let root =
        std::env::temp_dir().join(format!("atlas-hip-standard-clean-{}", std::process::id()));
    let rocm_root = root.join("AMD").join("ROCm").join("6.1");
    fs::create_dir_all(rocm_root.join("bin")).unwrap();
    let original_program_files = std::env::var_os("ProgramFiles");
    let original_program_files_x86 = std::env::var_os("ProgramFiles(x86)");
    std::env::set_var("ProgramFiles", &root);
    std::env::remove_var("ProgramFiles(x86)");

    let candidates = hip_runtime_library_candidates_from_host_roots();

    restore_env("ProgramFiles", original_program_files);
    restore_env("ProgramFiles(x86)", original_program_files_x86);
    let _ = fs::remove_dir_all(root);
    assert!(
        !candidates.iter().any(|candidate| candidate
            .components()
            .collect::<Vec<_>>()
            .windows(2)
            .any(|components| components[0].as_os_str() == "bin"
                && components[1].as_os_str() == "bin")),
        "expected HIP runtime candidates without duplicate bin segments, got {candidates:?}"
    );
}

#[cfg(windows)]
#[test]
fn hip_runtime_library_candidates_do_not_emit_duplicate_hip_segments() {
    let _env_lock = WINDOWS_ENV_LOCK.lock().unwrap();
    let root =
        std::env::temp_dir().join(format!("atlas-hip-standard-clean-{}", std::process::id()));
    let rocm_root = root.join("AMD").join("ROCm").join("6.1");
    fs::create_dir_all(rocm_root.join("hip").join("bin")).unwrap();
    let original_program_files = std::env::var_os("ProgramFiles");
    let original_program_files_x86 = std::env::var_os("ProgramFiles(x86)");
    std::env::set_var("ProgramFiles", &root);
    std::env::remove_var("ProgramFiles(x86)");

    let candidates = hip_runtime_library_candidates_from_host_roots();

    restore_env("ProgramFiles", original_program_files);
    restore_env("ProgramFiles(x86)", original_program_files_x86);
    let _ = fs::remove_dir_all(root);
    assert!(
        !candidates.iter().any(|candidate| candidate
            .components()
            .collect::<Vec<_>>()
            .windows(2)
            .any(|components| components[0].as_os_str() == "hip"
                && components[1].as_os_str() == "hip")),
        "expected HIP runtime candidates without duplicate hip segments, got {candidates:?}"
    );
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

fn write_fake_hipcc(bin_dir: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        let path = bin_dir.join("hipcc.cmd");
        fs::write(
            &path,
            "@echo off\r\necho fake-hsaco>\"%~5\"\r\nexit /b 0\r\n",
        )
        .unwrap();
        path
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let path = bin_dir.join("hipcc");
        fs::write(
            &path,
            "#!/bin/sh\nwhile [ \"$#\" -gt 0 ]; do\n  if [ \"$1\" = \"-o\" ]; then shift; echo fake-hsaco > \"$1\"; exit 0; fi\n  shift\ndone\nexit 0\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();
        path
    }
}
