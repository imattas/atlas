//! CUDA adapter CLI tests.

use atlas_gpu_cuda_adapter::{
    cuda_driver_library_candidates_from_roots, nvrtc_library_candidates_from_roots, run_cli,
    AdapterCommand, CudaPtxLauncher, LaunchArgs, Launcher,
};
use atlas_search_gpu::GpuSearcher;
use atlas_search_ir::SearchProgram;
use std::cell::RefCell;
use std::fs;

#[derive(Debug, Clone, Copy)]
struct FixtureLauncher;

impl Launcher for FixtureLauncher {
    fn compile_check(&self, _input: &str, _output: Option<&str>) -> Result<(), String> {
        Ok(())
    }

    fn launch(&self, args: &LaunchArgs) -> Result<Vec<u64>, String> {
        assert_eq!(args.artifact, "target/atlas-gpu/atlas_search.ptx");
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
    fn compile_check(&self, input: &str, output: Option<&str>) -> Result<(), String> {
        self.compile_checked
            .borrow_mut()
            .push(format!("{input}->{}", output.unwrap_or("")));
        Ok(())
    }

    fn launch(&self, _args: &LaunchArgs) -> Result<Vec<u64>, String> {
        Ok(Vec::new())
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

    assert_eq!(output, "match=11\nmatch=13\nmatch=17\n");
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
    };

    let matches = CudaPtxLauncher.launch(&args).unwrap_or_else(|error| {
        panic!("CUDA NVRTC/driver/device e2e prerequisites failed: {error}")
    });

    assert_eq!(matches, vec![0x55, 0x155]);
    let _ = fs::remove_dir_all(output_dir);
}
