//! Loop policies.

use std::collections::BTreeMap;

/// Bounded loop execution policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopPolicy {
    max_iterations: usize,
    counts: BTreeMap<String, usize>,
}

impl LoopPolicy {
    /// Creates a policy.
    #[must_use]
    pub fn new(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            counts: BTreeMap::new(),
        }
    }

    /// Records one loop iteration and returns whether it is permitted.
    pub fn enter(&mut self, loop_id: impl Into<String>) -> bool {
        let count = self.counts.entry(loop_id.into()).or_default();
        if *count >= self.max_iterations {
            return false;
        }
        *count += 1;
        true
    }
}
