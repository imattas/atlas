//! CLI command tests.

use atlas_cli::run;
use std::fs;
use std::sync::{Mutex, OnceLock};

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

#[test]
fn cli_supports_required_commands() {
    for command in ["solve", "inspect", "benchmark", "worker", "doctor"] {
        let output = run(&[command.to_owned()]).unwrap();
        assert!(!output.is_empty());
    }
}

#[test]
fn solve_emits_machine_readable_json() {
    let output = run(&["solve".to_owned()]).unwrap();

    assert!(output.starts_with('{'));
    assert!(output.contains("\"schema_major\":1"));
}

#[test]
fn solve_executes_search_runtime_and_reports_matches() {
    let output = run(&[
        "solve".to_owned(),
        "--fixture".to_owned(),
        "xor".to_owned(),
        "--start".to_owned(),
        "0x50".to_owned(),
        "--end".to_owned(),
        "0x60".to_owned(),
    ])
    .unwrap();

    assert!(output.contains("\"result_level\":\"ModelOnly\""));
    assert!(output.contains("matches=[85]"));
    assert!(output.contains("mode="));
    assert!(!output.contains("no backend result yet"));
}

#[test]
fn solve_force_gpu_launches_adapter_for_tiny_domain() {
    let _env_guard = env_lock();
    let tool_dir =
        std::env::temp_dir().join(format!("atlas-cli-force-gpu-tools-{}", std::process::id()));
    fs::create_dir_all(&tool_dir).unwrap();
    let adapter_path = tool_dir.join(if cfg!(windows) {
        "atlas-gpu-opencl-run.bat"
    } else {
        "atlas-gpu-opencl-run"
    });
    fs::write(
        &adapter_path,
        if cfg!(windows) {
            "@echo off\r\necho match=85\r\nexit /b 0\r\n"
        } else {
            "#!/bin/sh\necho match=85\nexit 0\n"
        },
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(&adapter_path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&adapter_path, permissions).unwrap();
    }
    fs::write(tool_dir.join("clinfo.exe"), "").unwrap();
    let original_path = std::env::var_os("PATH").unwrap_or_default();
    let joined_path = std::env::join_paths(
        std::iter::once(tool_dir.clone()).chain(std::env::split_paths(&original_path)),
    )
    .unwrap();
    std::env::set_var("PATH", &joined_path);

    let output = run(&[
        "solve".to_owned(),
        "--fixture".to_owned(),
        "xor".to_owned(),
        "--start".to_owned(),
        "0x50".to_owned(),
        "--end".to_owned(),
        "0x60".to_owned(),
        "--force-gpu".to_owned(),
    ])
    .unwrap();

    std::env::set_var("PATH", original_path);
    let _ = fs::remove_dir_all(tool_dir);
    assert!(output.contains("mode=DeviceValidated"));
    assert!(output.contains("matches=[85]"));
}

#[test]
fn solve_force_gpu_honors_explicit_gpu_sdk_selection() {
    let _env_guard = env_lock();
    let tool_dir =
        std::env::temp_dir().join(format!("atlas-cli-selected-gpu-sdk-{}", std::process::id()));
    fs::create_dir_all(&tool_dir).unwrap();
    fs::write(
        tool_dir.join(if cfg!(windows) {
            "atlas-gpu-opencl-run.bat"
        } else {
            "atlas-gpu-opencl-run"
        }),
        if cfg!(windows) {
            "@echo off\r\nexit /b 42\r\n"
        } else {
            "#!/bin/sh\nexit 42\n"
        },
    )
    .unwrap();
    fs::write(
        tool_dir.join(if cfg!(windows) {
            "atlas-gpu-vulkan-run.bat"
        } else {
            "atlas-gpu-vulkan-run"
        }),
        if cfg!(windows) {
            "@echo off\r\necho match=85\r\nexit /b 0\r\n"
        } else {
            "#!/bin/sh\necho match=85\nexit 0\n"
        },
    )
    .unwrap();
    fs::write(tool_dir.join("clinfo.exe"), "").unwrap();
    fs::write(tool_dir.join("vulkaninfo.exe"), "").unwrap();
    let original_path = std::env::var_os("PATH").unwrap_or_default();
    let joined_path = std::env::join_paths(
        std::iter::once(tool_dir.clone()).chain(std::env::split_paths(&original_path)),
    )
    .unwrap();
    std::env::set_var("PATH", &joined_path);

    let output = run(&[
        "solve".to_owned(),
        "--fixture".to_owned(),
        "xor".to_owned(),
        "--start".to_owned(),
        "0x50".to_owned(),
        "--end".to_owned(),
        "0x60".to_owned(),
        "--force-gpu".to_owned(),
        "--gpu-sdk".to_owned(),
        "vulkan".to_owned(),
    ])
    .unwrap();

    std::env::set_var("PATH", original_path);
    let _ = fs::remove_dir_all(tool_dir);
    assert!(output.contains("mode=DeviceValidated"));
    assert!(output.contains("matches=[85]"));
    assert!(output.contains("Vulkan"));
    assert!(!output.contains("driver exit 42"));
}

#[test]
fn solve_rejects_unknown_gpu_sdk_selection() {
    let error = run(&[
        "solve".to_owned(),
        "--gpu-sdk".to_owned(),
        "metal".to_owned(),
    ])
    .unwrap_err();

    assert!(error.contains("unsupported --gpu-sdk"));
    assert!(error.contains("opencl"));
    assert!(error.contains("vulkan"));
    assert!(error.contains("cuda"));
    assert!(error.contains("hip"));
}

#[test]
fn benchmark_reports_native_and_forced_gpu_runtime() {
    let _env_guard = env_lock();
    let tool_dir = std::env::temp_dir().join(format!(
        "atlas-cli-benchmark-gpu-tools-{}",
        std::process::id()
    ));
    fs::create_dir_all(&tool_dir).unwrap();
    let adapter_path = tool_dir.join(if cfg!(windows) {
        "atlas-gpu-opencl-run.bat"
    } else {
        "atlas-gpu-opencl-run"
    });
    fs::write(
        &adapter_path,
        if cfg!(windows) {
            "@echo off\r\necho match=85\r\nexit /b 0\r\n"
        } else {
            "#!/bin/sh\necho match=85\nexit 0\n"
        },
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(&adapter_path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&adapter_path, permissions).unwrap();
    }
    fs::write(tool_dir.join("clinfo.exe"), "").unwrap();
    let original_path = std::env::var_os("PATH").unwrap_or_default();
    let joined_path = std::env::join_paths(
        std::iter::once(tool_dir.clone()).chain(std::env::split_paths(&original_path)),
    )
    .unwrap();
    std::env::set_var("PATH", &joined_path);

    let output = run(&[
        "benchmark".to_owned(),
        "--fixture".to_owned(),
        "xor".to_owned(),
        "--start".to_owned(),
        "0x50".to_owned(),
        "--end".to_owned(),
        "0x60".to_owned(),
        "--force-gpu".to_owned(),
    ])
    .unwrap();

    std::env::set_var("PATH", original_path);
    let _ = fs::remove_dir_all(tool_dir);
    assert!(output.contains("\"kind\":\"benchmark\""));
    assert!(output.contains("\"native\""));
    assert!(output.contains("\"accelerator\""));
    assert!(output.contains("\"mode\":\"DeviceValidated\""));
    assert!(output.contains("\"matches\":[85]"));
    assert!(output.contains("\"launch\""));
    assert!(output.contains("\"global_size\":256"));
    assert!(output.contains("\"local_size\":256"));
    assert!(output.contains("\"max_matches\":1024"));
    assert!(output.contains("\"output_buffer_bytes\":8192"));
    assert!(!output.contains("\"kind\":\"benchmark\"}\n"));
}

#[test]
fn doctor_reports_detected_gpu_sdk_families() {
    let _env_guard = env_lock();
    let tool_dir =
        std::env::temp_dir().join(format!("atlas-cli-doctor-gpu-tools-{}", std::process::id()));
    fs::create_dir_all(&tool_dir).unwrap();
    fs::write(tool_dir.join("clinfo.exe"), "").unwrap();
    fs::write(tool_dir.join("vulkaninfo.exe"), "").unwrap();
    fs::write(tool_dir.join("hipcc.exe"), "").unwrap();
    let original_path = std::env::var_os("PATH").unwrap_or_default();
    let joined_path = std::env::join_paths(
        std::iter::once(tool_dir.clone()).chain(std::env::split_paths(&original_path)),
    )
    .unwrap();
    std::env::set_var("PATH", &joined_path);

    let output = run(&["doctor".to_owned()]).unwrap();

    std::env::set_var("PATH", original_path);
    let _ = fs::remove_dir_all(tool_dir);
    assert!(output.contains("\"kind\":\"doctor\""));
    assert!(output.contains("\"gpu_sdks\""));
    assert!(output.contains("\"OpenCL\""));
    assert!(output.contains("\"Vulkan\""));
    assert!(output.contains("\"HIP\""));
}

#[test]
fn doctor_reports_gpu_adapter_binary_availability() {
    let _env_guard = env_lock();
    let tool_dir =
        std::env::temp_dir().join(format!("atlas-cli-doctor-adapters-{}", std::process::id()));
    fs::create_dir_all(&tool_dir).unwrap();
    fs::write(
        tool_dir.join(if cfg!(windows) {
            "atlas-gpu-opencl-run.bat"
        } else {
            "atlas-gpu-opencl-run"
        }),
        if cfg!(windows) {
            "@echo off\r\nexit /b 0\r\n"
        } else {
            "#!/bin/sh\nexit 0\n"
        },
    )
    .unwrap();
    let original_path = std::env::var_os("PATH").unwrap_or_default();
    let joined_path = std::env::join_paths(std::iter::once(tool_dir.clone())).unwrap();
    std::env::set_var("PATH", &joined_path);

    let output = run(&["doctor".to_owned()]).unwrap();

    std::env::set_var("PATH", original_path);
    let _ = fs::remove_dir_all(tool_dir);
    assert!(output.contains("\"adapter_binaries\""));
    assert!(output.contains("\"name\":\"OpenCL\""));
    assert!(output.contains("\"command\":\"atlas-gpu-opencl-run\""));
    assert!(output.contains("\"available\":true"));
    assert!(output.contains("\"name\":\"CUDA\""));
    assert!(output.contains("\"command\":\"atlas-gpu-cuda-run\""));
    assert!(output.contains("\"available\":false"));
}

#[test]
fn doctor_reports_adjacent_gpu_adapter_binary_availability() {
    struct Cleanup {
        adapter_path: std::path::PathBuf,
        empty_path_dir: std::path::PathBuf,
        original_path: std::ffi::OsString,
    }

    impl Drop for Cleanup {
        fn drop(&mut self) {
            std::env::set_var("PATH", &self.original_path);
            let _ = fs::remove_file(&self.adapter_path);
            let _ = fs::remove_dir_all(&self.empty_path_dir);
        }
    }

    let _env_guard = env_lock();
    let adjacent_dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let adapter_path = adjacent_dir.join(if cfg!(windows) {
        "atlas-gpu-cuda-run.bat"
    } else {
        "atlas-gpu-cuda-run"
    });
    assert!(
        !adapter_path.exists(),
        "test would overwrite existing adapter: {}",
        adapter_path.display()
    );
    fs::write(
        &adapter_path,
        if cfg!(windows) {
            "@echo off\r\nexit /b 0\r\n"
        } else {
            "#!/bin/sh\nexit 0\n"
        },
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(&adapter_path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&adapter_path, permissions).unwrap();
    }
    let empty_path_dir =
        std::env::temp_dir().join(format!("atlas-cli-empty-path-{}", std::process::id()));
    fs::create_dir_all(&empty_path_dir).unwrap();
    let original_path = std::env::var_os("PATH").unwrap_or_default();
    let joined_path = std::env::join_paths(std::iter::once(empty_path_dir.clone())).unwrap();
    std::env::set_var("PATH", &joined_path);
    let _cleanup = Cleanup {
        adapter_path,
        empty_path_dir,
        original_path,
    };

    let output = run(&["doctor".to_owned()]).unwrap();

    assert!(output.contains("\"name\":\"CUDA\""));
    assert!(output.contains("\"command\":\"atlas-gpu-cuda-run\""));
    assert!(output.contains("\"available\":true"));
}
