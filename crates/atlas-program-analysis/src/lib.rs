//! Program simplification and slicing for Track 2.

mod branches;
mod calls;
mod loops;
mod slice;
mod taint;

pub use branches::prune_constant_branches;
pub use calls::{CallSummary, SummaryRegistry};
pub use loops::infer_loop_bound;
pub use slice::{slice_for_target, Instruction, ProgramSlice, Target};
pub use taint::tainted_instructions;
