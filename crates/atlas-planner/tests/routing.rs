//! Planner routing tests.

use atlas_analysis::Features;
use atlas_planner::{Capabilities, Capability, FeatureShape, Planner, PlanningFeatures};

#[test]
fn routes_xor_bitvector_modular_lattice_search_branching_and_components_deterministically() {
    let features = PlanningFeatures::new(
        Features {
            has_gf2_affine: true,
            ..Features::default()
        },
        [
            FeatureShape::BitVectorArithmetic,
            FeatureShape::ModularLinear,
            FeatureShape::Polynomial,
            FeatureShape::Lattice,
            FeatureShape::BoundedSearch,
            FeatureShape::BranchHeavy,
            FeatureShape::IndependentComponents,
        ],
    );
    let capabilities = Capabilities::new([
        Capability::Smt,
        Capability::Xor,
        Capability::Algebra,
        Capability::Lattice,
        Capability::BoundedSearch,
        Capability::Execution,
    ]);

    let portfolio = Planner::plan(&features, &capabilities);
    let names: Vec<_> = portfolio
        .stages
        .iter()
        .map(|stage| stage.name.as_str())
        .collect();

    assert_eq!(
        names,
        vec![
            "simplify",
            "partition-components",
            "gf2-elimination",
            "xor-aware-sat",
            "bitvector-smt",
            "modular-matrix",
            "polynomial-algebra",
            "lattice-reduction",
            "bounded-search",
            "concolic-execution",
            "general-smt"
        ]
    );
}
