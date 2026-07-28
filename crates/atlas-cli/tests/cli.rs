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
