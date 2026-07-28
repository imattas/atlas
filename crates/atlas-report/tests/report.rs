//! Report tests.

use atlas_report::SolveReportV1;
use atlas_validator::ResultLevel;

#[test]
fn report_is_versioned_deterministic_json() {
    let report = SolveReportV1::new(ResultLevel::ModelOnly, "abc", "atlas solve input", "ok");

    assert_eq!(
        report.to_json(),
        "{\"schema_major\":1,\"result_level\":\"ModelOnly\",\"input_hash\":\"abc\",\"reproduction\":\"atlas solve input\",\"explanation\":\"ok\"}"
    );
}

#[test]
fn report_redacts_secret_marker() {
    let report = SolveReportV1::new(ResultLevel::Partial, "abc", "cmd", "SECRET=value");

    assert!(report.to_json().contains("SECRET=<redacted>value"));
}
