//! Target slicing.

use std::collections::BTreeSet;

use atlas_program::Program;

/// Simplified instruction model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    /// Stable instruction id.
    pub id: String,
    /// Variables read by the instruction.
    pub reads: Vec<String>,
    /// Variables written by the instruction.
    pub writes: Vec<String>,
    /// Whether the instruction has observable side effects relevant to validation.
    pub observable: bool,
}

impl Instruction {
    /// Creates an instruction.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        reads: impl Into<Vec<String>>,
        writes: impl Into<Vec<String>>,
        observable: bool,
    ) -> Self {
        Self {
            id: id.into(),
            reads: reads.into(),
            writes: writes.into(),
            observable,
        }
    }
}

/// Slice target variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    /// Target variable.
    pub variable: String,
}

/// Program slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramSlice {
    /// Retained instructions.
    pub instructions: Vec<Instruction>,
    /// Provenance mapping back to original instruction ids.
    pub provenance: Vec<String>,
}

/// Computes a backward slice for a target variable.
#[must_use]
pub fn slice_for_target(
    _program: &Program,
    instructions: &[Instruction],
    target: &Target,
) -> ProgramSlice {
    let mut needed = BTreeSet::from([target.variable.clone()]);
    let mut retained = Vec::new();
    for instruction in instructions.iter().rev() {
        let writes_needed = instruction
            .writes
            .iter()
            .any(|write| needed.contains(write));
        if writes_needed || instruction.observable {
            for read in &instruction.reads {
                needed.insert(read.clone());
            }
            retained.push(instruction.clone());
        }
    }
    retained.reverse();
    let provenance = retained
        .iter()
        .map(|instruction| instruction.id.clone())
        .collect();
    ProgramSlice {
        instructions: retained,
        provenance,
    }
}
