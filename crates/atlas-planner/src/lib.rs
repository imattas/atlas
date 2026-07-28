//! Transparent rule-based strategy planner.

use atlas_analysis::Features;
use std::collections::BTreeSet;

/// Backend capability kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Capability {
    /// SMT bit-vector backend available.
    Smt,
    /// XOR-aware SAT/GF(2) backend available.
    Xor,
    /// Native algebra backend available.
    Algebra,
    /// Lattice backend available.
    Lattice,
    /// Bounded native/SIMD/GPU search available.
    BoundedSearch,
    /// Symbolic execution backend available.
    Execution,
}

/// Advertised backend capabilities.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Capabilities {
    /// Available backend capabilities.
    pub available: BTreeSet<Capability>,
}

impl Capabilities {
    /// Creates a capability set.
    #[must_use]
    pub fn new(available: impl IntoIterator<Item = Capability>) -> Self {
        Self {
            available: available.into_iter().collect(),
        }
    }

    fn has(&self, capability: Capability) -> bool {
        self.available.contains(&capability)
    }
}

/// Planner feature shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FeatureShape {
    /// Fixed-width arithmetic with carries or mixed bit operations.
    BitVectorArithmetic,
    /// Linear modular system.
    ModularLinear,
    /// Low-degree polynomial system.
    Polynomial,
    /// Lattice-shaped bounded relation.
    Lattice,
    /// Small bounded candidate region.
    BoundedSearch,
    /// Branch-heavy program.
    BranchHeavy,
    /// Multiple independent components exist.
    IndependentComponents,
}

/// Additional planner features not yet represented by `atlas-analysis`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlanningFeatures {
    /// Analysis-derived features.
    pub analysis: Features,
    /// Additional feature shapes.
    pub shapes: BTreeSet<FeatureShape>,
}

impl PlanningFeatures {
    /// Creates planning features.
    #[must_use]
    pub fn new(analysis: Features, shapes: impl IntoIterator<Item = FeatureShape>) -> Self {
        Self {
            analysis,
            shapes: shapes.into_iter().collect(),
        }
    }

    fn has(&self, shape: FeatureShape) -> bool {
        self.shapes.contains(&shape)
    }
}

/// Planned strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Strategy {
    /// Stable strategy name.
    pub name: String,
    /// Time budget in milliseconds.
    pub time_budget_ms: u64,
}

/// Ordered strategy portfolio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Portfolio {
    /// Ordered stages.
    pub stages: Vec<Strategy>,
}

/// Rule-based planner.
pub struct Planner;

impl Planner {
    /// Builds a deterministic strategy portfolio from explicit features.
    #[must_use]
    pub fn plan(features: &PlanningFeatures, capabilities: &Capabilities) -> Portfolio {
        let mut stages = Vec::new();
        push(&mut stages, "simplify", 500);
        if features.has(FeatureShape::IndependentComponents) {
            push(&mut stages, "partition-components", 500);
        }
        if features.analysis.has_gf2_affine && capabilities.has(Capability::Xor) {
            push(&mut stages, "gf2-elimination", 2_000);
            push(&mut stages, "xor-aware-sat", 5_000);
        }
        if features.has(FeatureShape::BitVectorArithmetic) && capabilities.has(Capability::Smt) {
            push(&mut stages, "bitvector-smt", 5_000);
        }
        if features.has(FeatureShape::ModularLinear) && capabilities.has(Capability::Algebra) {
            push(&mut stages, "modular-matrix", 3_000);
        }
        if features.has(FeatureShape::Polynomial) && capabilities.has(Capability::Algebra) {
            push(&mut stages, "polynomial-algebra", 8_000);
        }
        if features.has(FeatureShape::Lattice) && capabilities.has(Capability::Lattice) {
            push(&mut stages, "lattice-reduction", 10_000);
        }
        if features.has(FeatureShape::BoundedSearch) && capabilities.has(Capability::BoundedSearch)
        {
            push(&mut stages, "bounded-search", 4_000);
        }
        if features.has(FeatureShape::BranchHeavy) && capabilities.has(Capability::Execution) {
            push(&mut stages, "concolic-execution", 10_000);
        }
        if capabilities.has(Capability::Smt) {
            push(&mut stages, "general-smt", 10_000);
        }
        Portfolio { stages }
    }
}

fn push(stages: &mut Vec<Strategy>, name: &str, time_budget_ms: u64) {
    if stages.iter().all(|stage| stage.name != name) {
        stages.push(Strategy {
            name: name.to_owned(),
            time_budget_ms,
        });
    }
}
