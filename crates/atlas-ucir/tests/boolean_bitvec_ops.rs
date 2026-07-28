//! Boolean and bit-vector operator tests used by analysis.

use atlas_ucir::{Builder, Evaluator, Model, Value};

#[test]
fn xor_evaluates_fixed_width_bit_vectors() {
    let mut builder = Builder::new();
    let a = builder.bitvec_const(8, 0b1010_0000).unwrap();
    let b = builder.bitvec_const(8, 0b0011_1100).unwrap();
    let root = builder.xor(a, b).unwrap();
    let graph = builder.finish_with_root(root).unwrap();

    assert_eq!(
        Evaluator::evaluate(&graph, &Model::new()).unwrap(),
        Value::bitvec(8, 0b1001_1100).unwrap()
    );
}

#[test]
fn and_short_circuits_to_false_for_false_input() {
    let mut builder = Builder::new();
    let x = builder.bitvec_const(8, 1).unwrap();
    let y = builder.bitvec_const(8, 2).unwrap();
    let false_eq = builder.eq(x, y).unwrap();
    let true_eq = builder.eq(x, x).unwrap();
    let root = builder.and([true_eq, false_eq]).unwrap();
    let graph = builder.finish_with_root(root).unwrap();

    assert_eq!(
        Evaluator::evaluate(&graph, &Model::new()).unwrap(),
        Value::Bool(false)
    );
}
