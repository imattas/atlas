//! Program analysis tests.

use atlas_program::{Architecture, Program};
use atlas_program_analysis::{
    infer_loop_bound, prune_constant_branches, slice_for_target, tainted_instructions, CallSummary,
    Instruction, SummaryRegistry, Target,
};

fn program() -> Program {
    Program {
        architecture: Architecture::RestrictedC,
        entry: "main".to_owned(),
        sources: vec!["checker.c".to_owned()],
    }
}

#[test]
fn backward_slice_removes_irrelevant_logging_and_initialization() {
    let instructions = vec![
        Instruction::new("init", Vec::<String>::new(), vec!["tmp".to_owned()], false),
        Instruction::new(
            "read",
            vec!["input".to_owned()],
            vec!["x".to_owned()],
            false,
        ),
        Instruction::new("log", vec!["tmp".to_owned()], Vec::<String>::new(), false),
        Instruction::new("accept", vec!["x".to_owned()], vec!["ok".to_owned()], true),
    ];

    let slice = slice_for_target(
        &program(),
        &instructions,
        &Target {
            variable: "ok".to_owned(),
        },
    );

    assert_eq!(slice.provenance, vec!["read", "accept"]);
}

#[test]
fn forward_taint_tracks_input_influence() {
    let instructions = vec![
        Instruction::new(
            "read",
            vec!["input".to_owned()],
            vec!["x".to_owned()],
            false,
        ),
        Instruction::new("derive", vec!["x".to_owned()], vec!["y".to_owned()], false),
        Instruction::new(
            "unrelated",
            vec!["z".to_owned()],
            vec!["w".to_owned()],
            false,
        ),
    ];

    assert_eq!(
        tainted_instructions(&instructions, &["input".to_owned()]),
        vec!["read", "derive"]
    );
}

#[test]
fn branch_pruning_keeps_only_reachable_constant_branch() {
    assert_eq!(
        prune_constant_branches(Some(true), "then", "else"),
        vec!["then"]
    );
    assert_eq!(
        prune_constant_branches(Some(false), "then", "else"),
        vec!["else"]
    );
    assert_eq!(
        prune_constant_branches(None, "then", "else"),
        vec!["then", "else"]
    );
}

#[test]
fn loop_bound_inference_handles_bounded_positive_loops() {
    assert_eq!(infer_loop_bound(0, 4, 1), Some(5));
    assert_eq!(infer_loop_bound(5, 0, 1), None);
    assert_eq!(infer_loop_bound(0, 4, 0), None);
}

#[test]
fn pure_library_summaries_are_versioned() {
    let mut registry = SummaryRegistry::default();
    registry.register(CallSummary {
        name: "strlen".to_owned(),
        effect: "returns bounded length".to_owned(),
        version: 1,
    });

    assert_eq!(registry.get("strlen").unwrap().version, 1);
}
