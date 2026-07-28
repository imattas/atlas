//! Command-line interface helpers.

use atlas_report::SolveReportV1;
use atlas_scheduler::CancellationToken;
use atlas_search_gpu::{
    AcceleratorRuntime, GpuSdk, GpuSdkDetector, ProcessDriverRunner, RuntimeMode, RuntimePolicy,
};
use atlas_search_ir::{SearchDomain, SearchProgram};
use atlas_search_native::NativeSearcher;
use atlas_validator::ResultLevel;
use std::path::PathBuf;
use std::time::Instant;

/// Runs the CLI with explicit args and returns stdout.
///
/// # Errors
///
/// Returns an error when args are invalid.
pub fn run(args: &[String]) -> Result<String, String> {
    let Some(command) = args.first().map(String::as_str) else {
        return Err("missing command".to_owned());
    };
    match command {
        "doctor" => Ok(doctor()),
        "inspect" => Ok("{\"schema_major\":1,\"kind\":\"inspect\"}\n".to_owned()),
        "benchmark" => benchmark(&args[1..]),
        "worker" => Ok("{\"schema_major\":1,\"kind\":\"worker\"}\n".to_owned()),
        "solve" => solve(&args[1..]),
        other => Err(format!("unknown command '{other}'")),
    }
}

fn doctor() -> String {
    let detected_sdks = GpuSdkDetector::detect_from_host_path();
    let sdk_names = detected_sdks
        .iter()
        .map(gpu_sdk_name)
        .map(|name| format!("\"{}\"", json_escape(name)))
        .collect::<Vec<_>>()
        .join(",");
    let adapter_binaries = gpu_adapter_commands()
        .iter()
        .map(|adapter| {
            format!(
                "{{\"name\":\"{}\",\"command\":\"{}\",\"available\":{}}}",
                adapter.name,
                adapter.command,
                command_available(adapter.command)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"schema_major\":1,\"kind\":\"doctor\",\"gpu_sdks\":[{sdk_names}],\"adapter_binaries\":[{adapter_binaries}]}}\n"
    )
}

fn solve(args: &[String]) -> Result<String, String> {
    let request = SolveRequest::parse(args)?;
    let token = CancellationToken::new();
    let report = execute_accelerator(
        &request.program,
        request.domain,
        request.force_gpu,
        request.gpu_sdk,
        &token,
    );
    let result_level = if report.matches.is_empty() {
        ResultLevel::Unknown
    } else {
        ResultLevel::ModelOnly
    };
    let mode = mode_name(report.mode);
    let explanation = format!(
        "fixture={}; domain={}..{}; mode={mode}; matches={:?}; telemetry={}",
        request.fixture,
        request.domain.start,
        request.domain.end,
        report.matches,
        report.telemetry.rationale
    );
    let solve_report = SolveReportV1::new(
        result_level,
        "local-input",
        reproduction(&request),
        explanation,
    );
    Ok(format!("{}\n", solve_report.to_json()))
}

fn benchmark(args: &[String]) -> Result<String, String> {
    let request = SolveRequest::parse(args)?;
    let token = CancellationToken::new();
    let native_start = Instant::now();
    let native_matches = NativeSearcher::search(&request.program, request.domain, &token);
    let native_elapsed_ns = native_start.elapsed().as_nanos();
    let accelerator_start = Instant::now();
    let accelerator = execute_accelerator(
        &request.program,
        request.domain,
        request.force_gpu,
        request.gpu_sdk,
        &token,
    );
    let accelerator_elapsed_ns = accelerator_start.elapsed().as_nanos();
    Ok(format!(
        "{{\"schema_major\":1,\"kind\":\"benchmark\",\"fixture\":\"{}\",\"domain\":{{\"start\":{},\"end\":{}}},\"native\":{{\"elapsed_ns\":{},\"matches\":{}}},\"accelerator\":{{\"elapsed_ns\":{},\"mode\":\"{}\",\"matches\":{},\"launch\":{{\"global_size\":{},\"local_size\":{},\"max_matches\":{},\"output_buffer_bytes\":{}}},\"telemetry\":\"{}\"}}}}\n",
        json_escape(&request.fixture),
        request.domain.start,
        request.domain.end,
        native_elapsed_ns,
        format_matches(&native_matches),
        accelerator_elapsed_ns,
        mode_name(accelerator.mode),
        format_matches(&accelerator.matches),
        accelerator.telemetry.launch.global_size,
        accelerator.telemetry.launch.local_size,
        accelerator.telemetry.launch.max_matches,
        accelerator.telemetry.launch.output_buffer_bytes,
        json_escape(&accelerator.telemetry.rationale)
    ))
}

struct SolveRequest {
    fixture: String,
    program: SearchProgram,
    domain: SearchDomain,
    force_gpu: bool,
    gpu_sdk: Option<GpuSdkChoice>,
}

impl SolveRequest {
    fn parse(args: &[String]) -> Result<Self, String> {
        let fixture = optional_flag(args, "--fixture").unwrap_or("xor");
        let start = optional_flag(args, "--start")
            .map(parse_u64)
            .transpose()?
            .unwrap_or(0);
        let end = optional_flag(args, "--end")
            .map(parse_u64)
            .transpose()?
            .unwrap_or(256);
        if end <= start {
            return Err("--end must be greater than --start".to_owned());
        }
        let program =
            SearchProgram::try_from_fixture(fixture).map_err(|error| format!("{error:?}"))?;
        Ok(Self {
            fixture: fixture.to_owned(),
            program,
            domain: SearchDomain::new(start, end),
            force_gpu: has_flag(args, "--force-gpu"),
            gpu_sdk: optional_flag(args, "--gpu-sdk")
                .map(parse_gpu_sdk_choice)
                .transpose()?,
        })
    }
}

fn execute_accelerator(
    program: &SearchProgram,
    domain: SearchDomain,
    force_gpu: bool,
    gpu_sdk: Option<GpuSdkChoice>,
    token: &CancellationToken,
) -> atlas_search_gpu::AcceleratorReport {
    if force_gpu || gpu_sdk.is_some() {
        let detected_sdks = GpuSdkDetector::detect_from_host_path();
        let selected_sdks = filter_detected_sdks(detected_sdks, gpu_sdk);
        AcceleratorRuntime::execute_with_detected_driver_and_policy(
            program,
            domain,
            &selected_sdks,
            token,
            RuntimePolicy { force_gpu },
            &[],
            &ProcessDriverRunner,
        )
    } else {
        AcceleratorRuntime::execute_with_host_driver(program, domain, token)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuSdkChoice {
    OpenCl,
    Vulkan,
    Cuda,
    Hip,
}

fn parse_gpu_sdk_choice(value: &str) -> Result<GpuSdkChoice, String> {
    match value.to_ascii_lowercase().as_str() {
        "opencl" => Ok(GpuSdkChoice::OpenCl),
        "vulkan" => Ok(GpuSdkChoice::Vulkan),
        "cuda" => Ok(GpuSdkChoice::Cuda),
        "hip" => Ok(GpuSdkChoice::Hip),
        _ => Err(format!(
            "unsupported --gpu-sdk '{value}'; expected opencl, vulkan, cuda, or hip"
        )),
    }
}

fn filter_detected_sdks(detected_sdks: Vec<GpuSdk>, choice: Option<GpuSdkChoice>) -> Vec<GpuSdk> {
    let Some(choice) = choice else {
        return detected_sdks;
    };
    detected_sdks
        .into_iter()
        .filter(|sdk| gpu_sdk_matches_choice(sdk, choice))
        .collect()
}

fn gpu_sdk_matches_choice(sdk: &GpuSdk, choice: GpuSdkChoice) -> bool {
    matches!(
        (sdk, choice),
        (GpuSdk::OpenCl { .. }, GpuSdkChoice::OpenCl)
            | (GpuSdk::Vulkan { .. }, GpuSdkChoice::Vulkan)
            | (GpuSdk::Cuda { .. }, GpuSdkChoice::Cuda)
            | (GpuSdk::Hip { .. }, GpuSdkChoice::Hip)
    )
}

fn gpu_sdk_choice_name(choice: GpuSdkChoice) -> &'static str {
    match choice {
        GpuSdkChoice::OpenCl => "opencl",
        GpuSdkChoice::Vulkan => "vulkan",
        GpuSdkChoice::Cuda => "cuda",
        GpuSdkChoice::Hip => "hip",
    }
}

fn mode_name(mode: RuntimeMode) -> &'static str {
    match mode {
        RuntimeMode::CpuFallback => "CpuFallback",
        RuntimeMode::DeviceValidated => "DeviceValidated",
    }
}

fn gpu_sdk_name(sdk: &GpuSdk) -> &'static str {
    match sdk {
        GpuSdk::OpenCl { .. } => "OpenCL",
        GpuSdk::Vulkan { .. } => "Vulkan",
        GpuSdk::Cuda { .. } => "CUDA",
        GpuSdk::Hip { .. } => "HIP",
    }
}

struct GpuAdapterCommand {
    name: &'static str,
    command: &'static str,
}

fn gpu_adapter_commands() -> [GpuAdapterCommand; 4] {
    [
        GpuAdapterCommand {
            name: "OpenCL",
            command: "atlas-gpu-opencl-run",
        },
        GpuAdapterCommand {
            name: "Vulkan",
            command: "atlas-gpu-vulkan-run",
        },
        GpuAdapterCommand {
            name: "CUDA",
            command: "atlas-gpu-cuda-run",
        },
        GpuAdapterCommand {
            name: "HIP",
            command: "atlas-gpu-hip-run",
        },
    ]
}

fn command_available(command: &str) -> bool {
    adapter_search_dirs().into_iter().any(|dir| {
        command_candidates(&dir, command)
            .into_iter()
            .any(|path| path.is_file())
    })
}

fn adapter_search_dirs() -> Vec<PathBuf> {
    let mut dirs = std::env::var_os("PATH")
        .into_iter()
        .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            dirs.push(parent.to_path_buf());
        }
    }
    dirs
}

fn command_candidates(dir: &std::path::Path, command: &str) -> Vec<PathBuf> {
    let mut candidates = vec![dir.join(command)];
    if cfg!(windows) {
        candidates.push(dir.join(format!("{command}.exe")));
        candidates.push(dir.join(format!("{command}.bat")));
        candidates.push(dir.join(format!("{command}.cmd")));
    }
    candidates
}

fn optional_flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2)
        .find_map(|window| (window[0] == name).then_some(window[1].as_str()))
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn parse_u64(value: &str) -> Result<u64, String> {
    if let Some(hex) = value.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).map_err(|_| format!("invalid integer '{value}'"))
    } else {
        value
            .parse()
            .map_err(|_| format!("invalid integer '{value}'"))
    }
}

fn format_matches(matches: &[u64]) -> String {
    let values = matches.iter().map(u64::to_string).collect::<Vec<_>>();
    format!("[{}]", values.join(","))
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn reproduction(request: &SolveRequest) -> String {
    let force = if request.force_gpu {
        " --force-gpu"
    } else {
        ""
    };
    let gpu_sdk = request
        .gpu_sdk
        .map(|choice| format!(" --gpu-sdk {}", gpu_sdk_choice_name(choice)))
        .unwrap_or_default();
    format!(
        "atlas solve --fixture {} --start {} --end {}{}{}",
        request.fixture, request.domain.start, request.domain.end, force, gpu_sdk
    )
}
