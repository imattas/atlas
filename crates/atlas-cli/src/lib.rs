//! Command-line interface helpers.

use atlas_report::SolveReportV1;
use atlas_scheduler::CancellationToken;
use atlas_search_gpu::{
    AcceleratorReport, AcceleratorRuntime, GpuSdk, GpuSdkDetector, ProcessDriverRunner,
    RuntimeMode, RuntimePolicy, RuntimeTelemetry,
};
use atlas_search_ir::{SearchDomain, SearchProgram};
use atlas_search_native::NativeSearcher;
use atlas_search_simd::SimdSearcher;
use atlas_validator::ResultLevel;
use std::path::PathBuf;
use std::process::Command;
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
    let feature_probes = gpu_adapter_commands()
        .iter()
        .map(|adapter| {
            let probe = adapter_runtime_feature_probe(adapter.command);
            (adapter.name, probe)
        })
        .collect::<Vec<_>>();
    let gpu_features = feature_probes
        .iter()
        .map(|(name, probe)| {
            let features = probe
                .features
                .iter()
                .map(|feature| format!("\"{}\"", json_escape(feature)))
                .collect::<Vec<_>>()
                .join(",");
            format!("{{\"name\":\"{name}\",\"features\":[{features}]}}")
        })
        .collect::<Vec<_>>()
        .join(",");
    let gpu_feature_probes = feature_probes
        .iter()
        .map(|(name, probe)| {
            let features = probe
                .features
                .iter()
                .map(|feature| format!("\"{}\"", json_escape(feature)))
                .collect::<Vec<_>>()
                .join(",");
            let stderr = probe
                .stderr
                .as_deref()
                .map(json_escape)
                .map_or_else(|| "null".to_owned(), |value| format!("\"{value}\""));
            let exit_code = probe
                .exit_code
                .map_or_else(|| "null".to_owned(), |code| code.to_string());
            format!(
                "{{\"name\":\"{}\",\"ok\":{},\"exit_code\":{},\"stderr\":{},\"features\":[{}]}}",
                name, probe.ok, exit_code, stderr, features
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"schema_major\":1,\"kind\":\"doctor\",\"gpu_sdks\":[{sdk_names}],\"adapter_binaries\":[{adapter_binaries}],\"gpu_features\":[{gpu_features}],\"gpu_feature_probes\":[{gpu_feature_probes}]}}\n"
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
        "fixture={}; domain={}..{}; mode={mode}; matches={:?}; launch=global_size:{},local_size:{},max_matches:{},output_buffer_bytes:{}; telemetry={}",
        request.fixture,
        request.domain.start,
        request.domain.end,
        report.matches,
        report.telemetry.launch.global_size,
        report.telemetry.launch.local_size,
        report.telemetry.launch.max_matches,
        report.telemetry.launch.output_buffer_bytes,
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
    let mut native_samples_ns = Vec::with_capacity(request.samples);
    let mut simd_samples_ns = Vec::with_capacity(request.samples);
    let mut accelerator_samples_ns = Vec::with_capacity(request.samples);
    let mut native_matches = Vec::new();
    let mut simd_matches = Vec::new();
    let mut accelerator = None;
    for _ in 0..request.samples {
        let native_start = Instant::now();
        native_matches = NativeSearcher::search(&request.program, request.domain, &token);
        native_samples_ns.push(native_start.elapsed().as_nanos());
        let simd_start = Instant::now();
        simd_matches = SimdSearcher::search(&request.program, request.domain, &token, 4);
        simd_samples_ns.push(simd_start.elapsed().as_nanos());
        let accelerator_start = Instant::now();
        accelerator = Some(execute_accelerator(
            &request.program,
            request.domain,
            request.force_gpu,
            request.gpu_sdk,
            &token,
        ));
        accelerator_samples_ns.push(accelerator_start.elapsed().as_nanos());
    }
    let native_elapsed_ns = *native_samples_ns.iter().min().unwrap_or(&0);
    let simd_elapsed_ns = *simd_samples_ns.iter().min().unwrap_or(&0);
    let accelerator_elapsed_ns = *accelerator_samples_ns.iter().min().unwrap_or(&0);
    let accelerator = accelerator.expect("benchmark samples must be nonzero");
    let speedup_ratio = format_speedup_ratio(native_elapsed_ns, accelerator_elapsed_ns);
    let requested_gpu_sdk = requested_gpu_sdk_json(request.gpu_sdk);
    let actual_gpu_sdk = optional_string_json(accelerator.telemetry.selected_gpu_sdk.as_deref());
    let hardware = optional_string_json(accelerator.telemetry.selected_gpu_runtime.as_deref());
    Ok(format!(
        "{{\"schema_major\":1,\"kind\":\"benchmark\",\"fixture\":\"{}\",\"domain\":{{\"start\":{},\"end\":{}}},\"sample_count\":{},\"native_samples_ns\":{},\"simd_samples_ns\":{},\"accelerator_samples_ns\":{},\"native\":{{\"elapsed_ns\":{},\"matches\":{}}},\"simd\":{{\"elapsed_ns\":{},\"matches\":{}}},\"accelerator\":{{\"elapsed_ns\":{},\"requested_gpu_sdk\":{},\"actual_gpu_sdk\":{},\"hardware\":{},\"speedup_ratio\":{},\"mode\":\"{}\",\"matches\":{},\"launch\":{{\"global_size\":{},\"local_size\":{},\"max_matches\":{},\"output_buffer_bytes\":{}}},\"telemetry\":\"{}\"}}}}\n",
        json_escape(&request.fixture),
        request.domain.start,
        request.domain.end,
        request.samples,
        format_duration_samples(&native_samples_ns),
        format_duration_samples(&simd_samples_ns),
        format_duration_samples(&accelerator_samples_ns),
        native_elapsed_ns,
        format_matches(&native_matches),
        simd_elapsed_ns,
        format_matches(&simd_matches),
        accelerator_elapsed_ns,
        requested_gpu_sdk,
        actual_gpu_sdk,
        hardware,
        speedup_ratio,
        mode_name(accelerator.mode),
        format_matches(&accelerator.matches),
        accelerator.telemetry.launch.global_size,
        accelerator.telemetry.launch.local_size,
        accelerator.telemetry.launch.max_matches,
        accelerator.telemetry.launch.output_buffer_bytes,
        json_escape(&accelerator.telemetry.rationale)
    ))
}

fn requested_gpu_sdk_json(gpu_sdk: Option<GpuSdkChoice>) -> String {
    gpu_sdk
        .map(gpu_sdk_choice_name)
        .map_or_else(|| "null".to_owned(), |name| format!("\"{name}\""))
}

fn optional_string_json(value: Option<&str>) -> String {
    value.map_or_else(
        || "null".to_owned(),
        |value| format!("\"{}\"", json_escape(value)),
    )
}

fn format_speedup_ratio(native_elapsed_ns: u128, accelerator_elapsed_ns: u128) -> String {
    if accelerator_elapsed_ns == 0 {
        return "null".to_owned();
    }
    let scaled = native_elapsed_ns.saturating_mul(1_000_000) / accelerator_elapsed_ns;
    let whole = scaled / 1_000_000;
    let fractional = scaled % 1_000_000;
    format!("{whole}.{fractional:06}")
}

struct SolveRequest {
    fixture: String,
    program: SearchProgram,
    domain: SearchDomain,
    force_gpu: bool,
    gpu_sdk: Option<GpuSdkChoice>,
    samples: usize,
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
        let samples = optional_flag(args, "--samples")
            .map(parse_positive_usize)
            .transpose()?
            .unwrap_or(1);
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
            samples,
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
        let detected_sdks =
            GpuSdkDetector::detect_from_host_path_with_adapter_features(&ProcessDriverRunner);
        let selected_sdks = filter_detected_sdks(detected_sdks, gpu_sdk);
        if selected_sdks.is_empty() {
            if let Some(choice) = gpu_sdk {
                return missing_requested_gpu_sdk_report(program, domain, choice, token);
            }
        }
        let force_accelerator = force_gpu || gpu_sdk.is_some();
        AcceleratorRuntime::execute_with_detected_driver_and_policy(
            program,
            domain,
            &selected_sdks,
            token,
            RuntimePolicy {
                force_gpu: force_accelerator,
            },
            &[],
            &ProcessDriverRunner,
        )
    } else {
        AcceleratorRuntime::execute_with_host_driver(program, domain, token)
    }
}

fn missing_requested_gpu_sdk_report(
    program: &SearchProgram,
    domain: SearchDomain,
    choice: GpuSdkChoice,
    token: &CancellationToken,
) -> AcceleratorReport {
    AcceleratorReport {
        mode: RuntimeMode::CpuFallback,
        matches: NativeSearcher::search(program, domain, token),
        telemetry: RuntimeTelemetry {
            launch: AcceleratorRuntime::plan_launch(domain, 256, 1024),
            selected_gpu_sdk: None,
            selected_gpu_runtime: None,
            rationale: format!(
                "requested GPU SDK {} not detected; CPU fallback used",
                gpu_sdk_choice_display_name(choice)
            ),
            cpu_validated: true,
            rejected_device_matches: 0,
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuSdkChoice {
    OpenCl,
    Vulkan,
    Wgpu,
    Cuda,
    Hip,
}

fn parse_gpu_sdk_choice(value: &str) -> Result<GpuSdkChoice, String> {
    match value.to_ascii_lowercase().as_str() {
        "opencl" => Ok(GpuSdkChoice::OpenCl),
        "vulkan" => Ok(GpuSdkChoice::Vulkan),
        "wgpu" | "webgpu" => Ok(GpuSdkChoice::Wgpu),
        "cuda" => Ok(GpuSdkChoice::Cuda),
        "hip" => Ok(GpuSdkChoice::Hip),
        _ => Err(format!(
            "unsupported --gpu-sdk '{value}'; expected opencl, vulkan, wgpu, cuda, or hip"
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
            | (GpuSdk::Wgpu { .. }, GpuSdkChoice::Wgpu)
            | (GpuSdk::Cuda { .. }, GpuSdkChoice::Cuda)
            | (GpuSdk::Hip { .. }, GpuSdkChoice::Hip)
    )
}

fn gpu_sdk_choice_name(choice: GpuSdkChoice) -> &'static str {
    match choice {
        GpuSdkChoice::OpenCl => "opencl",
        GpuSdkChoice::Vulkan => "vulkan",
        GpuSdkChoice::Wgpu => "wgpu",
        GpuSdkChoice::Cuda => "cuda",
        GpuSdkChoice::Hip => "hip",
    }
}

fn gpu_sdk_choice_display_name(choice: GpuSdkChoice) -> &'static str {
    match choice {
        GpuSdkChoice::OpenCl => "OpenCL",
        GpuSdkChoice::Vulkan => "Vulkan",
        GpuSdkChoice::Wgpu => "WGPU",
        GpuSdkChoice::Cuda => "CUDA",
        GpuSdkChoice::Hip => "HIP",
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
        GpuSdk::Wgpu { .. } => "WGPU",
        GpuSdk::Cuda { .. } => "CUDA",
        GpuSdk::Hip { .. } => "HIP",
    }
}

struct GpuAdapterCommand {
    name: &'static str,
    command: &'static str,
}

fn gpu_adapter_commands() -> [GpuAdapterCommand; 5] {
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
            name: "WGPU",
            command: "atlas-gpu-wgpu-run",
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
    adapter_command_path(command).is_some()
}

struct AdapterRuntimeFeatureProbe {
    ok: bool,
    exit_code: Option<i32>,
    stderr: Option<String>,
    features: Vec<String>,
}

fn adapter_runtime_feature_probe(command: &str) -> AdapterRuntimeFeatureProbe {
    let Some(path) = adapter_command_path(command) else {
        return AdapterRuntimeFeatureProbe {
            ok: false,
            exit_code: None,
            stderr: Some("adapter binary not found".to_owned()),
            features: Vec::new(),
        };
    };
    let Ok(output) = Command::new(path).arg("--features").output() else {
        return AdapterRuntimeFeatureProbe {
            ok: false,
            exit_code: None,
            stderr: Some("failed to execute adapter feature probe".to_owned()),
            features: Vec::new(),
        };
    };
    if !output.status.success() {
        let stderr = trimmed_utf8(&output.stderr);
        return AdapterRuntimeFeatureProbe {
            ok: false,
            exit_code: output.status.code(),
            stderr,
            features: Vec::new(),
        };
    }
    let features = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.trim().strip_prefix("feature=").map(str::to_owned))
        .collect();
    AdapterRuntimeFeatureProbe {
        ok: true,
        exit_code: output.status.code(),
        stderr: trimmed_utf8(&output.stderr),
        features,
    }
}

fn trimmed_utf8(bytes: &[u8]) -> Option<String> {
    let trimmed = String::from_utf8_lossy(bytes).trim().to_owned();
    (!trimmed.is_empty()).then_some(trimmed)
}

fn adapter_command_path(command: &str) -> Option<PathBuf> {
    adapter_search_dirs().into_iter().find_map(|dir| {
        command_candidates(&dir, command)
            .into_iter()
            .find(|path| path.is_file())
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

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    let samples = value
        .parse()
        .map_err(|_| format!("invalid sample count '{value}'"))?;
    if samples == 0 {
        return Err("--samples must be greater than zero".to_owned());
    }
    Ok(samples)
}

fn format_matches(matches: &[u64]) -> String {
    let values = matches.iter().map(u64::to_string).collect::<Vec<_>>();
    format!("[{}]", values.join(","))
}

fn format_duration_samples(samples: &[u128]) -> String {
    let values = samples.iter().map(u128::to_string).collect::<Vec<_>>();
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
