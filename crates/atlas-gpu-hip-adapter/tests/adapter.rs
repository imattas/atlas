//! HIP adapter CLI tests.

#[cfg(windows)]
use atlas_gpu_hip_adapter::hip_runtime_library_candidates_from_host_roots;
use atlas_gpu_hip_adapter::{
    hip_runtime_library_candidates_from_roots, run_cli, AdapterCommand, FeatureReport,
    HipLaunchAbi, HipModuleLauncher, LaunchArgs, LaunchOutput, Launcher,
};
use atlas_search_gpu::GpuSearcher;
use atlas_search_ir::{SearchOp, SearchProgram};
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
    fn features(&self) -> Result<FeatureReport, String> {
        Ok(FeatureReport {
            hardware: "AMD Radeon RX 7900 XTX via HIP".to_owned(),
            features: vec!["int64".to_owned()],
        })
    }

    fn compile_check(&self, _artifact: &str, _output: Option<&str>) -> Result<(), String> {
        Ok(())
    }

    fn launch(&self, args: &LaunchArgs) -> Result<LaunchOutput, String> {
        assert_eq!(args.artifact, "target/atlas-gpu/atlas_search.hsaco");
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
            hardware: "Fixture HIP device".to_owned(),
            features: Vec::new(),
        })
    }

    fn compile_check(&self, artifact: &str, output: Option<&str>) -> Result<(), String> {
        self.compile_checked
            .borrow_mut()
            .push((artifact.to_owned(), output.map(str::to_owned)));
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
fn parses_features_command() {
    let command = AdapterCommand::parse(&["--features".to_owned()]).unwrap();

    assert_eq!(command, AdapterCommand::Features);
}

#[test]
fn cli_features_emits_launcher_capabilities() {
    let output = run_cli(&["--features".to_owned()], &FixtureLauncher).unwrap();

    assert_eq!(
        output,
        "hardware=AMD Radeon RX 7900 XTX via HIP\nfeature=int64\nfeature=launchAbiU32\nfeature=launchAbiU64\n"
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
fn rejects_global_size_smaller_than_launch_domain() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.hsaco".to_owned(),
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
fn rejects_max_matches_that_exceeds_kernel_uint() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.hsaco".to_owned(),
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

    assert!(error.contains("max-matches exceeds HIP uint"));
}

#[test]
fn rejects_grid_size_that_exceeds_hip_uint() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.hsaco".to_owned(),
        "--start".to_owned(),
        "0".to_owned(),
        "--end".to_owned(),
        "1".to_owned(),
        "--max-matches".to_owned(),
        "1".to_owned(),
        "--global-size".to_owned(),
        "4294967296".to_owned(),
        "--local-size".to_owned(),
        "1".to_owned(),
    ])
    .unwrap_err();

    assert!(error.contains("grid size exceeds HIP uint"));
}

#[test]
fn rejects_local_size_that_exceeds_hip_block_limit() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.hsaco".to_owned(),
        "--start".to_owned(),
        "0".to_owned(),
        "--end".to_owned(),
        "1025".to_owned(),
        "--max-matches".to_owned(),
        "1".to_owned(),
        "--global-size".to_owned(),
        "1025".to_owned(),
        "--local-size".to_owned(),
        "1025".to_owned(),
    ])
    .unwrap_err();

    assert!(error.contains("local-size exceeds HIP block limit"));
}

#[test]
fn parses_explicit_u32_launch_abi() {
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
        "--abi".to_owned(),
        "u32".to_owned(),
    ])
    .unwrap();

    assert_eq!(args.launch_abi, Some(HipLaunchAbi::U32));
}

#[test]
fn rejects_missing_explicit_launch_abi_value() {
    let error = LaunchArgs::parse(&[
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
        "--abi".to_owned(),
    ])
    .unwrap_err();

    assert!(error.contains("missing --abi value"));
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

    assert_eq!(output, "match_count=5\nmatch=11\nmatch=13\nmatch=17\n");
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
    let original_hip_path = std::env::var_os("HIP_PATH");
    let original_rocm_path = std::env::var_os("ROCM_PATH");
    let original_rocm_home = std::env::var_os("ROCM_HOME");
    #[cfg(windows)]
    let original_program_files = std::env::var_os("ProgramFiles");
    #[cfg(windows)]
    let original_program_files_x86 = std::env::var_os("ProgramFiles(x86)");
    std::env::set_var("PATH", &bin_dir);
    std::env::set_var("HIP_PATH", &root);
    std::env::set_var("ROCM_PATH", &root);
    std::env::set_var("ROCM_HOME", &root);
    #[cfg(windows)]
    {
        std::env::set_var("ProgramFiles", &root);
        std::env::remove_var("ProgramFiles(x86)");
    }
    let source = source.to_string_lossy().into_owned();
    let artifact_text = artifact.to_string_lossy().into_owned();

    HipModuleLauncher
        .compile_check(&source, Some(&artifact_text))
        .unwrap();

    assert!(artifact.exists());
    restore_env("PATH", original_path);
    restore_env("HIP_PATH", original_hip_path);
    restore_env("ROCM_PATH", original_rocm_path);
    restore_env("ROCM_HOME", original_rocm_home);
    #[cfg(windows)]
    {
        restore_env("ProgramFiles", original_program_files);
        restore_env("ProgramFiles(x86)", original_program_files_x86);
    }
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

    let source_path_text = source_path.to_string_lossy().into_owned();
    let code_object_path_text = code_object_path.to_string_lossy().into_owned();
    HipModuleLauncher
        .compile_check(&source_path_text, Some(&code_object_path_text))
        .unwrap_or_else(|error| {
            panic!("HIP source compile failed for detected arch {arch}: {error}")
        });
    let args = LaunchArgs {
        artifact: code_object_path.to_string_lossy().into_owned(),
        start: 0x50,
        end: 0x160,
        max_matches: 8,
        global_size: 512,
        local_size: 64,
        launch_abi: Some(HipLaunchAbi::U32),
    };

    let output = HipModuleLauncher.launch(&args).unwrap();

    assert_eq!(output.matches, vec![0x55, 0x155]);
    assert_eq!(output.match_count, 2);
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
#[ignore = "requires hipcc, HIP runtime, and an AMD HIP-capable device"]
fn generated_hip_dense_kernel_retains_full_device_buffer() {
    let program = SearchProgram::try_from_fixture("dense").unwrap();
    let source = GpuSearcher::compile_hip(&program);
    let output_dir =
        std::env::temp_dir().join(format!("atlas-hip-dense-e2e-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let source_path = output_dir.join("atlas_search.hip");
    let code_object_path = output_dir.join("atlas_search.hsaco");
    fs::write(&source_path, source).unwrap();
    let arch = detect_hip_arch().unwrap_or_else(|| "gfx1100".to_owned());

    let source_path_text = source_path.to_string_lossy().into_owned();
    let code_object_path_text = code_object_path.to_string_lossy().into_owned();
    HipModuleLauncher
        .compile_check(&source_path_text, Some(&code_object_path_text))
        .unwrap_or_else(|error| {
            panic!("HIP dense source compile failed for detected arch {arch}: {error}")
        });
    let args = LaunchArgs {
        artifact: code_object_path.to_string_lossy().into_owned(),
        start: 0,
        end: 1500,
        max_matches: 1500,
        global_size: 1536,
        local_size: 64,
        launch_abi: Some(HipLaunchAbi::U32),
    };

    let output = HipModuleLauncher.launch(&args).unwrap();

    let expected = (0..1500).collect::<Vec<_>>();
    assert_eq!(output.matches, expected);
    assert_eq!(output.match_count, 1500);
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
#[ignore = "requires hipcc, HIP runtime, and an AMD HIP-capable device with int64 support"]
fn generated_hip_64_bit_kernel_runs_on_device() {
    let program = SearchProgram::new(
        64,
        vec![SearchOp::XorEq {
            mask: 1,
            target: 0x8000_0000_0000_0001,
        }],
    )
    .unwrap();
    let source = GpuSearcher::compile_hip(&program);
    let output_dir = std::env::temp_dir().join(format!("atlas-hip-64-e2e-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let source_path = output_dir.join("atlas_search.hip");
    let code_object_path = output_dir.join("atlas_search.hsaco");
    fs::write(&source_path, source).unwrap();
    let arch = detect_hip_arch().unwrap_or_else(|| "gfx1100".to_owned());

    let source_path_text = source_path.to_string_lossy().into_owned();
    let code_object_path_text = code_object_path.to_string_lossy().into_owned();
    HipModuleLauncher
        .compile_check(&source_path_text, Some(&code_object_path_text))
        .unwrap_or_else(|error| {
            panic!("HIP 64-bit source compile failed for detected arch {arch}: {error}")
        });
    let args = LaunchArgs {
        artifact: code_object_path.to_string_lossy().into_owned(),
        start: 0x8000_0000_0000_0000,
        end: 0x8000_0000_0000_0002,
        max_matches: 2,
        global_size: 256,
        local_size: 64,
        launch_abi: Some(HipLaunchAbi::U64),
    };

    let output = HipModuleLauncher.launch(&args).unwrap();

    assert_eq!(output.matches, vec![0x8000_0000_0000_0000]);
    assert_eq!(output.match_count, 1);
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
            "@echo off\r\nset args=%*\r\nif \"%args:-nogpuinc=%\"==\"%args%\" exit /b 3\r\nif \"%args:-nogpulib=%\"==\"%args%\" exit /b 4\r\nif not \"%~6\"==\"-o\" exit /b 5\r\necho fake-hsaco>\"%~7\"\r\nexit /b 0\r\n",
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
            "#!/bin/sh\nprintf '%s\\n' \"$@\" | grep -- '-nogpuinc' >/dev/null || exit 3\nprintf '%s\\n' \"$@\" | grep -- '-nogpulib' >/dev/null || exit 4\nwhile [ \"$#\" -gt 0 ]; do\n  if [ \"$1\" = \"-o\" ]; then shift; echo fake-hsaco > \"$1\"; exit 0; fi\n  shift\ndone\nexit 5\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();
        path
    }
}
