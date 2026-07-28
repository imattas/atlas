//! Command-line interface helpers.

use atlas_report::SolveReportV1;
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
        "solve" => {
            let report = SolveReportV1::new(
                ResultLevel::Unknown,
                "local-input",
                "atlas solve",
                "no backend result yet",
            );
            Ok(format!("{}\n", report.to_json()))
        }
        other => Err(format!("unknown command '{other}'")),
    }
}
