//! Forward taint tracking.

use std::collections::BTreeSet;

use crate::Instruction;

/// Returns ids of instructions influenced by input variables.
#[must_use]
pub fn tainted_instructions(instructions: &[Instruction], inputs: &[String]) -> Vec<String> {
    let mut tainted: BTreeSet<String> = inputs.iter().cloned().collect();
    let mut out = Vec::new();
    for instruction in instructions {
        if instruction.reads.iter().any(|read| tainted.contains(read)) {
            out.push(instruction.id.clone());
            for write in &instruction.writes {
                tainted.insert(write.clone());
            }
        }
    }
    out
}
