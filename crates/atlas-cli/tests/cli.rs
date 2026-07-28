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
