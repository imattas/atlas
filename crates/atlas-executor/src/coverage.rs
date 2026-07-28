//! Deterministic coverage-prioritized queue.

use std::collections::VecDeque;

use crate::PathCandidate;

/// Coverage queue that pops highest score first with deterministic tie ordering.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CoverageQueue {
    queue: VecDeque<PathCandidate>,
}

impl CoverageQueue {
    /// Creates an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pushes a path candidate.
    pub fn push(&mut self, candidate: PathCandidate) {
        self.queue.push_back(candidate);
        self.queue.make_contiguous().sort_by(|left, right| {
            right
                .coverage_score
                .cmp(&left.coverage_score)
                .then(left.id.cmp(&right.id))
        });
    }

    /// Pops the highest-priority candidate.
    pub fn pop(&mut self) -> Option<PathCandidate> {
        self.queue.pop_front()
    }

    /// Returns whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}
