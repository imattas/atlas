//! Golden analysis behavior tests.

use atlas_analysis::analyze;
use atlas_ucir::Builder;

#[test]
fn detects_constant_contradictions() {
    let mut builder = Builder::new();
    let one = builder.bitvec_const(8, 1).unwrap();
    let two = builder.bitvec_const(8, 2).unwrap();
    let root = builder.eq(one, two).unwrap();
    let graph = builder.finish_with_root(root).unwrap();

    let result = analyze(&graph).unwrap();

    assert!(result.features.has_contradiction);
    assert!(result.derivations.iter().any(|derivation| derivation.pass
        == "general.contradiction-detection"
        && derivation.sound));
}

#[test]
fn recognizes_xor_as_gf2_affine_bitvec_structure() {
    let mut builder = Builder::new();
    let x = builder.bitvec_var("x", 8).unwrap();
    let y = builder.bitvec_var("y", 8).unwrap();
    let root = builder.xor(x, y).unwrap();
    let graph = builder.finish_with_root(root).unwrap();

    let result = analyze(&graph).unwrap();

    assert_eq!(result.features.variable_count, 2);
    assert_eq!(result.features.xor_ops, 1);
    assert!(result.features.has_gf2_affine);
}

#[test]
fn partitions_top_level_boolean_conjunctions() {
    let mut builder = Builder::new();
    let x = builder.bitvec_var("x", 8).unwrap();
    let y = builder.bitvec_var("y", 8).unwrap();
    let one = builder.bitvec_const(8, 1).unwrap();
    let two = builder.bitvec_const(8, 2).unwrap();
    let x_eq = builder.eq(x, one).unwrap();
    let y_eq = builder.eq(y, two).unwrap();
    let root = builder.and([x_eq, y_eq]).unwrap();
    let graph = builder.finish_with_root(root).unwrap();

    let result = analyze(&graph).unwrap();

    assert_eq!(result.components.len(), 2);
    assert_eq!(result.components[0].variables, vec!["x"]);
    assert_eq!(result.components[1].variables, vec!["y"]);
}

#[test]
fn detects_modular_domains_for_algebra_routing() {
    let mut builder = Builder::new();
    let a = builder.modular_const(17, 20).unwrap();
    let b = builder.modular_const(17, 15).unwrap();
    let root = builder.add(a, b).unwrap();
    let graph = builder.finish_with_root(root).unwrap();

    let result = analyze(&graph).unwrap();

    assert_eq!(result.features.modular_nodes, 3);
    assert!(result
        .derivations
        .iter()
        .any(|derivation| derivation.pass == "algebra.domain-detection" && derivation.sound));
}
