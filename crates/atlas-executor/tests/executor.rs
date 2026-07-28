//! Symbolic/concolic executor tests.

use atlas_executor::{
    merge_states, BranchConstraint, ConcolicSeed, CoverageQueue, ExecutionBudget, ExecutionEvent,
    Executor, Inputs, LoopPolicy, PathCandidate, SeedMutator, SymbolicState,
};
use atlas_program::{Architecture, Program};
use atlas_validator::ResultLevel;

fn program() -> Program {
    Program {
        architecture: Architecture::RestrictedPython,
        entry: "module".to_owned(),
        sources: vec!["checker.py".to_owned()],
    }
}

#[test]
fn symbolic_registers_memory_aliases_and_branch_constraints_are_recorded() {
    let mut state = SymbolicState::new();
    state.write_register("rax", "input0 + 1");
    state.memory.store(0x1000, "flag[0]");
    state.memory.alias("sp", 0x1000);
    state.assume("rax == 0x41");

    assert_eq!(state.register("rax"), Some("input0 + 1"));
    assert_eq!(state.memory.load_alias("sp"), Some("flag[0]"));
    assert_eq!(state.path_predicates(), &["rax == 0x41".to_owned()]);
}

#[test]
fn loop_policy_enforces_bounded_iterations() {
    let mut loops = LoopPolicy::new(2);

    assert!(loops.enter("loop0"));
    assert!(loops.enter("loop0"));
    assert!(!loops.enter("loop0"));
}

#[test]
fn concolic_seed_mutation_is_deterministic() {
    let seed = ConcolicSeed {
        bytes: vec![0b1010_0000],
    };

    assert_eq!(
        SeedMutator::flip(&seed, 0, 0b0000_1111).bytes,
        vec![0b1010_1111]
    );
    assert_eq!(SeedMutator::flip(&seed, 3, 0xff).bytes, seed.bytes);
}

#[test]
fn coverage_queue_prioritizes_highest_score_then_id() {
    let mut queue = CoverageQueue::new();
    queue.push(PathCandidate::new("b", Vec::new(), 2));
    queue.push(PathCandidate::new("a", Vec::new(), 2));
    queue.push(PathCandidate::new("c", Vec::new(), 1));

    assert_eq!(queue.pop().unwrap().id, "a");
    assert_eq!(queue.pop().unwrap().id, "b");
    assert_eq!(queue.pop().unwrap().id, "c");
}

#[test]
fn merge_preserves_path_predicates_from_both_states() {
    let mut left = SymbolicState::new();
    let mut right = SymbolicState::new();
    left.assume("left");
    right.assume("right");

    let merged = merge_states(&left, &right);

    assert_eq!(
        merged.path_predicates(),
        &["left".to_owned(), "merged:right".to_owned()]
    );
}

#[test]
fn exhausted_path_budget_returns_partial_never_unsat() {
    let events = Executor::explore(
        &program(),
        &Inputs {
            symbolic: vec!["a".to_owned(), "b".to_owned()],
            seed: Vec::new(),
        },
        ExecutionBudget {
            max_paths: 1,
            max_loop_iterations: 1,
            max_processes: 1,
        },
    );

    assert!(events.iter().any(|event| {
        matches!(
            event,
            ExecutionEvent::BudgetExhausted {
                result: ResultLevel::Partial,
                ..
            }
        )
    }));
    assert!(!events.iter().any(|event| {
        matches!(
            event,
            ExecutionEvent::Complete {
                result: ResultLevel::ProvenUnsat
            }
        )
    }));
}

#[test]
fn path_candidates_retain_explicit_branch_constraints() {
    let candidate = PathCandidate::new("p0", vec![BranchConstraint::new("x > 0", true)], 1);

    assert_eq!(candidate.constraints[0].predicate, "x > 0");
    assert!(candidate.constraints[0].taken);
}
