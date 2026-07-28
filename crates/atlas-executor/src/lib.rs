//! Symbolic and concolic execution primitives for `AtlasCTF`.

mod concolic;
mod coverage;
mod loops;
mod memory;
mod merge;
mod path;
mod state;
mod summaries;

pub use concolic::{ConcolicSeed, SeedMutator};
pub use coverage::CoverageQueue;
pub use loops::LoopPolicy;
pub use memory::SymbolicMemory;
pub use merge::merge_states;
pub use path::{BranchConstraint, PathCandidate};
pub use state::SymbolicState;
pub use summaries::{FunctionSummary, SummaryStore};

use atlas_program::Program;
use atlas_validator::ResultLevel;

/// Execution inputs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Inputs {
    /// Symbolic input names.
    pub symbolic: Vec<String>,
    /// Concrete seed bytes for concolic execution.
    pub seed: Vec<u8>,
}

/// Execution budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionBudget {
    /// Maximum explored paths.
    pub max_paths: usize,
    /// Maximum loop iterations per loop id.
    pub max_loop_iterations: usize,
    /// Maximum process count.
    pub max_processes: usize,
}

impl Default for ExecutionBudget {
    fn default() -> Self {
        Self {
            max_paths: 16,
            max_loop_iterations: 4,
            max_processes: 1,
        }
    }
}

/// Executor event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionEvent {
    /// A path became feasible.
    Path(PathCandidate),
    /// A path or process budget was exhausted.
    BudgetExhausted {
        /// Nonterminal result level.
        result: ResultLevel,
        /// Explanation.
        reason: String,
    },
    /// Exploration completed.
    Complete {
        /// Nonterminal or terminal result.
        result: ResultLevel,
    },
}

/// Event stream alias.
pub type EventStream = Vec<ExecutionEvent>;

/// Symbolic executor.
pub struct Executor;

impl Executor {
    /// Explores a lowered program deterministically.
    #[must_use]
    pub fn explore(_program: &Program, inputs: &Inputs, budget: ExecutionBudget) -> EventStream {
        if budget.max_processes == 0 || budget.max_paths == 0 {
            return vec![ExecutionEvent::BudgetExhausted {
                result: ResultLevel::Partial,
                reason: "path or process budget exhausted".to_owned(),
            }];
        }

        let mut queue = CoverageQueue::new();
        for (index, name) in inputs.symbolic.iter().enumerate() {
            queue.push(PathCandidate::new(
                format!("path-{index}"),
                vec![BranchConstraint::new(format!("{name} symbolic"), true)],
                index,
            ));
        }
        if queue.is_empty() {
            queue.push(PathCandidate::new("path-0", Vec::new(), 0));
        }

        let mut events = Vec::new();
        for _ in 0..budget.max_paths {
            let Some(candidate) = queue.pop() else {
                break;
            };
            events.push(ExecutionEvent::Path(candidate));
        }
        if queue.is_empty() {
            events.push(ExecutionEvent::Complete {
                result: ResultLevel::Partial,
            });
        } else {
            events.push(ExecutionEvent::BudgetExhausted {
                result: ResultLevel::Partial,
                reason: "path budget exhausted with preserved candidates".to_owned(),
            });
        }
        events
    }
}
