//! UCIR semantic table tests.

use atlas_ucir::{
    canonical_hash, Builder, Endianness, Evaluator, Model, SourceLocation, Type, Value, WithSource,
};

#[test]
fn bit_vector_arithmetic_wraps_at_declared_width() {
    let mut builder = Builder::new();
    let a = builder.bitvec_const(8, 250).unwrap();
    let b = builder.bitvec_const(8, 10).unwrap();
    let sum = builder.add(a, b).unwrap();
    let graph = builder.finish_with_root(sum).unwrap();

    assert_eq!(
        Evaluator::evaluate(&graph, &Model::new()).unwrap(),
        Value::bitvec(8, 4).unwrap()
    );
}

#[test]
fn signed_and_unsigned_comparisons_are_distinct() {
    let mut builder = Builder::new();
    let high = builder.bitvec_const(8, 0xff).unwrap();
    let one = builder.bitvec_const(8, 1).unwrap();
    let unsigned = builder.unsigned_lt(high, one).unwrap();
    let signed = builder.signed_lt(high, one).unwrap();
    let unsigned_graph = builder.clone().finish_with_root(unsigned).unwrap();
    let signed_graph = builder.finish_with_root(signed).unwrap();

    assert_eq!(
        Evaluator::evaluate(&unsigned_graph, &Model::new()).unwrap(),
        Value::Bool(false)
    );
    assert_eq!(
        Evaluator::evaluate(&signed_graph, &Model::new()).unwrap(),
        Value::Bool(true)
    );
}

#[test]
fn memory_loads_respect_endianness_and_width() {
    let mut builder = Builder::new();
    let memory = builder.bytes_const([0x12, 0x34, 0x56, 0x78]).unwrap();
    let offset = builder.int_const(1);
    let big = builder.load(memory, offset, 16, Endianness::Big).unwrap();
    let little = builder
        .load(memory, offset, 16, Endianness::Little)
        .unwrap();
    let big_graph = builder.clone().finish_with_root(big).unwrap();
    let little_graph = builder.finish_with_root(little).unwrap();

    assert_eq!(
        Evaluator::evaluate(&big_graph, &Model::new()).unwrap(),
        Value::bitvec(16, 0x3456).unwrap()
    );
    assert_eq!(
        Evaluator::evaluate(&little_graph, &Model::new()).unwrap(),
        Value::bitvec(16, 0x5634).unwrap()
    );
}

#[test]
fn modular_values_reduce_into_domain() {
    let mut builder = Builder::new();
    let x = builder.modular_const(17, 20).unwrap();
    let y = builder.modular_const(17, 15).unwrap();
    let sum = builder.add(x, y).unwrap();
    let graph = builder.finish_with_root(sum).unwrap();

    assert_eq!(
        Evaluator::evaluate(&graph, &Model::new()).unwrap(),
        Value::modular(17, 1).unwrap()
    );
}

#[test]
fn arrays_store_and_load_fixed_width_values() {
    let mut builder = Builder::new();
    let array = builder.array_const(8, 8, 0).unwrap();
    let index = builder.bitvec_const(8, 3).unwrap();
    let value = builder.bitvec_const(8, 0xaa).unwrap();
    let stored = builder.store(array, index, value).unwrap();
    let loaded = builder.load_array(stored, index).unwrap();
    let graph = builder.finish_with_root(loaded).unwrap();

    assert_eq!(
        Evaluator::evaluate(&graph, &Model::new()).unwrap(),
        Value::bitvec(8, 0xaa).unwrap()
    );
}

#[test]
fn source_provenance_is_preserved_on_nodes() {
    let source = SourceLocation::new("challenge.py", 7, 12);
    let mut builder = Builder::new();
    let x = builder
        .bitvec_var("x", 32)
        .unwrap()
        .with_source(&mut builder, source.clone())
        .unwrap();
    let graph = builder.finish_with_root(x).unwrap();

    assert_eq!(graph.node(x).unwrap().source(), Some(&source));
    assert_eq!(graph.node(x).unwrap().ty(), &Type::BitVec { width: 32 });
}

#[test]
fn equivalent_canonical_graphs_have_equal_hashes() {
    fn make_graph() -> atlas_ucir::ExprGraph {
        let mut builder = Builder::new();
        let x = builder.bitvec_var("x", 8).unwrap();
        let one = builder.bitvec_const(8, 1).unwrap();
        let root = builder.add(x, one).unwrap();
        builder.finish_with_root(root).unwrap()
    }

    assert_eq!(canonical_hash(&make_graph()), canonical_hash(&make_graph()));
}
