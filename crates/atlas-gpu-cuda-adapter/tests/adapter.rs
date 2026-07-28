//! CUDA adapter CLI tests.

use atlas_gpu_cuda_adapter::{
    cuda_driver_library_candidates_from_roots, cuda_sdk_root_candidates_from_bases,
    nvcc_command_candidates_from_roots, nvcc_ptx_output_path_for_source,
    nvrtc_library_candidates_from_roots, run_cli, AdapterCommand, CudaPtxLauncher, LaunchAbi,
    LaunchArgs, LaunchOutput, Launcher,
};
use atlas_search_gpu::GpuSearcher;
use atlas_search_ir::SearchProgram;
use std::cell::RefCell;
use std::fs;
use std::sync::{Mutex, OnceLock};

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

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
    fn features(&self) -> Result<Vec<String>, String> {
        Ok(vec!["int64".to_owned()])
    }

    fn compile_check(&self, _input: &str, _output: Option<&str>) -> Result<(), String> {
        Ok(())
    }

    fn launch(&self, args: &LaunchArgs) -> Result<LaunchOutput, String> {
        assert_eq!(args.artifact, "target/atlas-gpu/atlas_search.ptx");
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
    fn features(&self) -> Result<Vec<String>, String> {
        Ok(Vec::new())
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
        "target/atlas-gpu/atlas_search.ptx".to_owned(),
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

    assert_eq!(args.artifact, "target/atlas-gpu/atlas_search.ptx");
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
        "target/atlas-gpu/atlas_search.ptx".to_owned(),
    ])
    .unwrap();

    assert_eq!(
        command,
        AdapterCommand::CompileCheck {
            input: "target/atlas-gpu/atlas_search.ptx".to_owned(),
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

    assert_eq!(output, "feature=int64\n");
}

#[test]
fn cli_compile_check_invokes_launcher_backend() {
    let launcher = RecordingLauncher::new();

    let output = run_cli(
        &[
            "--compile-check".to_owned(),
            "target/atlas-gpu/atlas_search.ptx".to_owned(),
        ],
        &launcher,
    )
    .unwrap();

    assert_eq!(output, "");
    assert_eq!(
        launcher.compile_checked.borrow().as_slice(),
        ["target/atlas-gpu/atlas_search.ptx->"]
    );
}

#[test]
fn rejects_malformed_or_unbounded_launch_ranges() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.ptx".to_owned(),
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
        "target/atlas-gpu/atlas_search.ptx".to_owned(),
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
        "target/atlas-gpu/atlas_search.ptx".to_owned(),
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

    assert!(error.contains("max-matches exceeds CUDA uint"));
}

#[test]
fn rejects_grid_size_that_exceeds_cuda_uint() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.ptx".to_owned(),
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

    assert!(error.contains("grid size exceeds CUDA uint"));
}

#[test]
fn rejects_local_size_that_exceeds_cuda_block_limit() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.ptx".to_owned(),
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

    assert!(error.contains("local-size exceeds CUDA block limit"));
}

#[test]
fn parses_explicit_u32_launch_abi() {
    let args = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.ptx".to_owned(),
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

    assert_eq!(args.launch_abi, Some(LaunchAbi::U32));
}

#[test]
fn rejects_missing_explicit_launch_abi_value() {
    let error = LaunchArgs::parse(&[
        "target/atlas-gpu/atlas_search.ptx".to_owned(),
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
            "target/atlas-gpu/atlas_search.ptx".to_owned(),
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
fn compile_check_rejects_ptx_without_atlas_kernel_entry() {
    let output_dir = std::env::temp_dir().join(format!("atlas-cuda-ptx-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let ptx_path = output_dir.join("bad.ptx");
    fs::write(&ptx_path, ".version 8.0\n.target sm_52\n").unwrap();

    let error = CudaPtxLauncher
        .compile_check(&ptx_path.to_string_lossy(), None)
        .unwrap_err();

    assert!(error.contains("missing atlas_search kernel entry"));
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
fn compile_check_routes_cuda_source_to_runtime_compiler_not_ptx_parser() {
    let _env_guard = env_lock();
    let output_dir = std::env::temp_dir().join(format!("atlas-cuda-source-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let source_path = output_dir.join("atlas_search.cu");
    fs::write(&source_path, "this is not valid cuda").unwrap();

    let error = CudaPtxLauncher
        .compile_check(&source_path.to_string_lossy(), None)
        .unwrap_err();

    assert!(
        error.contains("NVRTC")
            || error.contains("nvrtc")
            || error.contains("CUDA source compile failed"),
        "{error}"
    );
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
fn compile_check_writes_ptx_artifact_when_output_is_requested() {
    let output_dir = std::env::temp_dir().join(format!("atlas-cuda-cache-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let input_path = output_dir.join("input.ptx");
    let output_path = output_dir.join("atlas_search.ptx");
    fs::write(
        &input_path,
        ".version 8.0\n.target sm_52\n.visible .entry atlas_search() { ret; }\n",
    )
    .unwrap();

    run_cli(
        &[
            "--compile-check".to_owned(),
            input_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            output_path.to_string_lossy().into_owned(),
        ],
        &CudaPtxLauncher,
    )
    .unwrap();

    let ptx = fs::read_to_string(&output_path).unwrap();
    assert!(ptx.contains(".entry atlas_search"));
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
fn compile_check_writes_ptx_with_clang_when_nvrtc_and_nvcc_are_absent() {
    let _env_guard = env_lock();
    let root = std::env::temp_dir().join(format!("atlas-cuda-fake-clang-{}", std::process::id()));
    let bin_dir = root.join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    write_fake_clang_cuda_compiler(&bin_dir);
    let source_path = root.join("atlas_search.cu");
    let output_path = root.join("atlas_search.ptx");
    fs::write(
        &source_path,
        "extern \"C\" __global__ void atlas_search() {}\n",
    )
    .unwrap();
    let original_path = std::env::var_os("PATH");
    let original_cuda_path = std::env::var_os("CUDA_PATH");
    let original_cuda_home = std::env::var_os("CUDA_HOME");
    let original_cuda_root = std::env::var_os("CUDA_ROOT");
    #[cfg(windows)]
    let original_program_files = std::env::var_os("ProgramFiles");
    #[cfg(windows)]
    let original_program_files_x86 = std::env::var_os("ProgramFiles(x86)");
    std::env::set_var("PATH", &bin_dir);
    std::env::set_var("CUDA_PATH", &root);
    std::env::set_var("CUDA_HOME", &root);
    std::env::set_var("CUDA_ROOT", &root);
    #[cfg(windows)]
    {
        std::env::set_var("ProgramFiles", &root);
        std::env::remove_var("ProgramFiles(x86)");
    }

    CudaPtxLauncher
        .compile_check(
            &source_path.to_string_lossy(),
            Some(&output_path.to_string_lossy()),
        )
        .unwrap();

    let ptx = fs::read_to_string(&output_path).unwrap();
    assert!(ptx.contains(".entry atlas_search"));
    restore_env("PATH", original_path);
    restore_env("CUDA_PATH", original_cuda_path);
    restore_env("CUDA_HOME", original_cuda_home);
    restore_env("CUDA_ROOT", original_cuda_root);
    #[cfg(windows)]
    {
        restore_env("ProgramFiles", original_program_files);
        restore_env("ProgramFiles(x86)", original_program_files_x86);
    }
    let _ = fs::remove_dir_all(root);
}

#[test]
fn nvrtc_library_candidates_include_cuda_sdk_root_library_directories() {
    let cuda_root =
        std::env::temp_dir().join(format!("atlas-cuda-sdk-root-{}", std::process::id()));
    let candidates = nvrtc_library_candidates_from_roots([cuda_root.clone()]);

    assert!(candidates
        .iter()
        .any(|path| path.starts_with(cuda_root.join("bin"))));
    assert!(candidates
        .iter()
        .any(|path| path.starts_with(cuda_root.join("lib64"))));
    assert!(candidates
        .iter()
        .any(|path| path.starts_with(cuda_root.join("lib").join("x64"))));
}

#[test]
fn cuda_driver_library_candidates_include_cuda_root_driver_compat_directories() {
    let cuda_root =
        std::env::temp_dir().join(format!("atlas-cuda-driver-root-{}", std::process::id()));
    let candidates = cuda_driver_library_candidates_from_roots([cuda_root.clone()]);

    assert!(candidates
        .iter()
        .any(|path| path.starts_with(cuda_root.join("compat"))));
    assert!(candidates
        .iter()
        .any(|path| path.starts_with(cuda_root.join("lib64").join("stubs"))));
    assert!(candidates
        .iter()
        .any(|path| path.starts_with(cuda_root.join("lib64"))));
}

#[test]
fn cuda_sdk_root_candidates_include_versioned_standard_install_dirs() {
    let program_files =
        std::env::temp_dir().join(format!("atlas-cuda-program-files-{}", std::process::id()));
    let cuda_base = program_files
        .join("NVIDIA GPU Computing Toolkit")
        .join("CUDA");
    let v12 = cuda_base.join("v12.4");
    let v13 = cuda_base.join("v13.0");
    fs::create_dir_all(&v12).unwrap();
    fs::create_dir_all(&v13).unwrap();

    let candidates = cuda_sdk_root_candidates_from_bases([program_files.clone()]);

    assert!(candidates.contains(&v13));
    assert!(candidates.contains(&v12));
    assert!(candidates.iter().any(|path| {
        path.starts_with(
            program_files
                .join("NVIDIA GPU Computing Toolkit")
                .join("CUDA"),
        )
    }));
    let _ = fs::remove_dir_all(program_files);
}

#[test]
fn cuda_sdk_root_candidates_order_versioned_installs_newest_first() {
    let program_files = std::env::temp_dir().join(format!(
        "atlas-cuda-program-files-order-{}",
        std::process::id()
    ));
    let cuda_base = program_files
        .join("NVIDIA GPU Computing Toolkit")
        .join("CUDA");
    let v9 = cuda_base.join("v9.0");
    let v12 = cuda_base.join("v12.4");
    let v13 = cuda_base.join("v13.0");
    fs::create_dir_all(&v9).unwrap();
    fs::create_dir_all(&v12).unwrap();
    fs::create_dir_all(&v13).unwrap();

    let candidates = cuda_sdk_root_candidates_from_bases([program_files.clone()]);
    let versioned = candidates
        .iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with('v'))
        })
        .cloned()
        .collect::<Vec<_>>();

    assert_eq!(versioned, vec![v13, v12, v9]);
    let _ = fs::remove_dir_all(program_files);
}

#[test]
fn nvcc_command_candidates_include_cuda_sdk_bins() {
    let cuda_root = std::env::temp_dir().join(format!("atlas-cuda-nvcc-{}", std::process::id()));
    let candidates = nvcc_command_candidates_from_roots([cuda_root.clone()]);

    assert!(candidates
        .iter()
        .any(|path| path.starts_with(cuda_root.join("bin"))));
    assert!(candidates.iter().any(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("nvcc"))
    }));
}

#[test]
fn nvcc_fallback_uses_distinct_ptx_outputs_for_distinct_sources() {
    let first = nvcc_ptx_output_path_for_source("target/atlas-gpu/first/atlas_search.cu");
    let second = nvcc_ptx_output_path_for_source("target/atlas-gpu/second/atlas_search.cu");

    assert_ne!(first, second);
    assert_eq!(
        first.extension().and_then(|value| value.to_str()),
        Some("ptx")
    );
    assert_eq!(
        second.extension().and_then(|value| value.to_str()),
        Some("ptx")
    );
}

#[test]
#[ignore = "requires NVRTC, CUDA driver runtime, and an NVIDIA CUDA device"]
fn generated_cuda_kernel_runs_on_device_and_preserves_full_candidates() {
    let program = SearchProgram::try_from_fixture("xor").unwrap();
    let source = GpuSearcher::compile_cuda(&program);
    let output_dir = std::env::temp_dir().join(format!("atlas-cuda-e2e-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let source_path = output_dir.join("atlas_search.cu");
    fs::write(&source_path, source).unwrap();

    let args = LaunchArgs {
        artifact: source_path.to_string_lossy().into_owned(),
        start: 0x50,
        end: 0x160,
        max_matches: 8,
        global_size: 512,
        local_size: 64,
        launch_abi: Some(LaunchAbi::U32),
    };

    let output = CudaPtxLauncher.launch(&args).unwrap_or_else(|error| {
        panic!("CUDA NVRTC/driver/device e2e prerequisites failed: {error}")
    });

    assert_eq!(output.matches, vec![0x55, 0x155]);
    assert_eq!(output.match_count, 2);
    let _ = fs::remove_dir_all(output_dir);
}

fn write_fake_clang_cuda_compiler(bin_dir: &std::path::Path) {
    #[cfg(windows)]
    {
        for name in ["clang.cmd", "clang++.cmd"] {
            fs::write(
                bin_dir.join(name),
                "@echo off\r\nif not \"%~1\"==\"--cuda-device-only\" exit /b 3\r\nif not \"%~3\"==\"-nocudainc\" exit /b 4\r\nif not \"%~4\"==\"-nocudalib\" exit /b 5\r\n:loop\r\nif \"%~1\"==\"\" exit /b 6\r\nif \"%~1\"==\"-o\" goto output\r\nshift\r\ngoto loop\r\n:output\r\nshift\r\necho .version 8.0>\"%~1\"\r\necho .target sm_52>>\"%~1\"\r\necho .visible .entry atlas_search() { ret; }>>\"%~1\"\r\nexit /b 0\r\n",
            )
            .unwrap();
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for name in ["clang", "clang++"] {
            let path = bin_dir.join(name);
            fs::write(
                &path,
                "#!/bin/sh\nprintf '%s\\n' \"$@\" | grep -- '--cuda-device-only' >/dev/null || exit 3\nprintf '%s\\n' \"$@\" | grep -- '-nocudainc' >/dev/null || exit 4\nprintf '%s\\n' \"$@\" | grep -- '-nocudalib' >/dev/null || exit 5\nwhile [ \"$#\" -gt 0 ]; do\n  if [ \"$1\" = \"-o\" ]; then shift; printf '.version 8.0\\n.target sm_52\\n.visible .entry atlas_search() { ret; }\\n' > \"$1\"; exit 0; fi\n  shift\ndone\nexit 6\n",
            )
            .unwrap();
            let mut permissions = fs::metadata(&path).unwrap().permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&path, permissions).unwrap();
        }
    }
}
