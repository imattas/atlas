//! Bounded property-style UCIR tests.

use atlas_ucir::{Builder, Evaluator, Model, Simplifier, Value};

#[test]
fn identity_simplification_preserves_bounded_bitvec_models() {
    for width in [1, 2, 4, 8] {
        let max = 1_u128 << width;
        for x_value in 0..max {
            let mut builder = Builder::new();
            let x = builder.bitvec_var("x", width).unwrap();
            let c = builder.bitvec_const(width, 3).unwrap();
            let expr = builder.add(x, c).unwrap();
            let graph = builder.finish_with_root(expr).unwrap();
            let simplified = Simplifier::identity(&graph);

            let mut model = Model::new();
            model.insert("x".to_owned(), Value::bitvec(width, x_value).unwrap());

            assert_eq!(
                Evaluator::evaluate(&graph, &model).unwrap(),
                Evaluator::evaluate(&simplified, &model).unwrap()
            );
        }
    }
}

#[test]
fn evaluator_reports_missing_model_values_with_variable_name() {
    let mut builder = Builder::new();
    let x = builder.bitvec_var("missing", 8).unwrap();
    let graph = builder.finish_with_root(x).unwrap();

    let error = Evaluator::evaluate(&graph, &Model::new()).unwrap_err();

    assert!(error.to_string().contains("missing"));
}
