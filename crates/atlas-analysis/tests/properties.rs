//! Bounded analysis property tests.

use atlas_analysis::analyze;
use atlas_ucir::{Builder, Evaluator, Model, Value};

#[test]
fn analysis_identity_graph_preserves_all_small_models() {
    for x_value in 0..16 {
        let mut builder = Builder::new();
        let x = builder.bitvec_var("x", 4).unwrap();
        let c = builder.bitvec_const(4, 7).unwrap();
        let root = builder.xor(x, c).unwrap();
        let graph = builder.finish_with_root(root).unwrap();
        let analyzed = analyze(&graph).unwrap();

        let mut model = Model::new();
        model.insert("x".to_owned(), Value::bitvec(4, x_value).unwrap());

        assert_eq!(
            Evaluator::evaluate(&graph, &model).unwrap(),
            Evaluator::evaluate(&analyzed.graph, &model).unwrap()
        );
        assert!(analyzed
            .derivations
            .iter()
            .all(|derivation| derivation.sound));
    }
}
