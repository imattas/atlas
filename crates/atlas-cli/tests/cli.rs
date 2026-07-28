//! CLI command tests.

use atlas_cli::run;

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
