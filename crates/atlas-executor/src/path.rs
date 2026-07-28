//! Path constraints and candidates.

/// Explicit branch constraint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchConstraint {
    /// Predicate expression.
    pub predicate: String,
    /// Whether the branch condition is taken.
    pub taken: bool,
}

impl BranchConstraint {
    /// Creates a branch constraint.
    #[must_use]
    pub fn new(predicate: impl Into<String>, taken: bool) -> Self {
        Self {
            predicate: predicate.into(),
            taken,
        }
    }
}

/// Candidate path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathCandidate {
    /// Stable path id.
    pub id: String,
    /// Constraints accumulated along the path.
    pub constraints: Vec<BranchConstraint>,
    /// Deterministic coverage score.
    pub coverage_score: usize,
}

impl PathCandidate {
    /// Creates a path candidate.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        constraints: Vec<BranchConstraint>,
        coverage_score: usize,
    ) -> Self {
        Self {
            id: id.into(),
            constraints,
            coverage_score,
        }
    }
}
