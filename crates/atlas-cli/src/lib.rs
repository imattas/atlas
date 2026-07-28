//! Command-line interface helpers.

use atlas_report::SolveReportV1;
use atlas_scheduler::CancellationToken;
use atlas_search_gpu::{
    AcceleratorRuntime, GpuSdkDetector, ProcessDriverRunner, RuntimeMode, RuntimePolicy,
};
use atlas_search_ir::{SearchDomain, SearchProgram};
use atlas_validator::ResultLevel;

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
        "benchmark" => Ok("{\"schema_major\":1,\"kind\":\"benchmark\"}\n".to_owned()),
        "worker" => Ok("{\"schema_major\":1,\"kind\":\"worker\"}\n".to_owned()),
        "solve" => solve(&args[1..]),
        other => Err(format!("unknown command '{other}'")),
    }
}

fn solve(args: &[String]) -> Result<String, String> {
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
    let force_gpu = has_flag(args, "--force-gpu");
    let program = SearchProgram::try_from_fixture(fixture).map_err(|error| format!("{error:?}"))?;
    let domain = SearchDomain::new(start, end);
    let token = CancellationToken::new();
    let report = if force_gpu {
        let detected_sdks = GpuSdkDetector::detect_from_host_path();
        AcceleratorRuntime::execute_with_detected_driver_and_policy(
            &program,
            domain,
            &detected_sdks,
            &token,
            RuntimePolicy { force_gpu },
            &[],
            &ProcessDriverRunner,
        )
    } else {
        AcceleratorRuntime::execute_with_host_driver(&program, domain, &token)
    };
    let result_level = if report.matches.is_empty() {
        ResultLevel::Unknown
    } else {
        ResultLevel::ModelOnly
    };
    let mode = match report.mode {
        RuntimeMode::CpuFallback => "CpuFallback",
        RuntimeMode::DeviceValidated => "DeviceValidated",
    };
    let explanation = format!(
        "fixture={fixture}; domain={start}..{end}; mode={mode}; matches={:?}; telemetry={}",
        report.matches, report.telemetry.rationale
    );
    let solve_report = SolveReportV1::new(
        result_level,
        "local-input",
        reproduction(fixture, start, end),
        explanation,
    );
    Ok(format!("{}\n", solve_report.to_json()))
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

fn reproduction(fixture: &str, start: u64, end: u64) -> String {
    format!("atlas solve --fixture {fixture} --start {start} --end {end}")
}
