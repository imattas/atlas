//! Command-line interface helpers.

use atlas_report::SolveReportV1;
use atlas_scheduler::CancellationToken;
use atlas_search_gpu::{
    AcceleratorRuntime, GpuSdkDetector, ProcessDriverRunner, RuntimeMode, RuntimePolicy,
};
use atlas_search_ir::{SearchDomain, SearchProgram};
use atlas_search_native::NativeSearcher;
use atlas_validator::ResultLevel;
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
        "doctor" => Ok("AtlasCTF doctor: ok\n".to_owned()),
        "inspect" => Ok("{\"schema_major\":1,\"kind\":\"inspect\"}\n".to_owned()),
        "benchmark" => benchmark(&args[1..]),
        "worker" => Ok("{\"schema_major\":1,\"kind\":\"worker\"}\n".to_owned()),
        "solve" => solve(&args[1..]),
        other => Err(format!("unknown command '{other}'")),
    }
}

fn solve(args: &[String]) -> Result<String, String> {
    let request = SolveRequest::parse(args)?;
    let token = CancellationToken::new();
    let report = execute_accelerator(&request.program, request.domain, request.force_gpu, &token);
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
    let accelerator =
        execute_accelerator(&request.program, request.domain, request.force_gpu, &token);
    let accelerator_elapsed_ns = accelerator_start.elapsed().as_nanos();
    Ok(format!(
        "{{\"schema_major\":1,\"kind\":\"benchmark\",\"fixture\":\"{}\",\"domain\":{{\"start\":{},\"end\":{}}},\"native\":{{\"elapsed_ns\":{},\"matches\":{}}},\"accelerator\":{{\"elapsed_ns\":{},\"mode\":\"{}\",\"matches\":{},\"telemetry\":\"{}\"}}}}\n",
        json_escape(&request.fixture),
        request.domain.start,
        request.domain.end,
        native_elapsed_ns,
        format_matches(&native_matches),
        accelerator_elapsed_ns,
        mode_name(accelerator.mode),
        format_matches(&accelerator.matches),
        json_escape(&accelerator.telemetry.rationale)
    ))
}

struct SolveRequest {
    fixture: String,
    program: SearchProgram,
    domain: SearchDomain,
    force_gpu: bool,
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
        })
    }
}

fn execute_accelerator(
    program: &SearchProgram,
    domain: SearchDomain,
    force_gpu: bool,
    token: &CancellationToken,
) -> atlas_search_gpu::AcceleratorReport {
    if force_gpu {
        let detected_sdks = GpuSdkDetector::detect_from_host_path();
        AcceleratorRuntime::execute_with_detected_driver_and_policy(
            program,
            domain,
            &detected_sdks,
            token,
            RuntimePolicy { force_gpu },
            &[],
            &ProcessDriverRunner,
        )
    } else {
        AcceleratorRuntime::execute_with_host_driver(program, domain, token)
    }
}

fn mode_name(mode: RuntimeMode) -> &'static str {
    match mode {
        RuntimeMode::CpuFallback => "CpuFallback",
        RuntimeMode::DeviceValidated => "DeviceValidated",
    }
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
    format!(
        "atlas solve --fixture {} --start {} --end {}{}",
        request.fixture, request.domain.start, request.domain.end, force
    )
}
