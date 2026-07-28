//! UCIR canonical serialization tests.

use atlas_ucir::{canonical_bytes, canonical_hash, Builder};

#[test]
fn canonical_serialization_round_trips_by_rebuilding_equivalent_graph() {
    let mut first = Builder::new();
    let x = first.bitvec_var("flag0", 8).unwrap();
    let c = first.bitvec_const(8, 0x41).unwrap();
    let root = first.eq(x, c).unwrap();
    let first = first.finish_with_root(root).unwrap();

    let mut second = Builder::new();
    let x = second.bitvec_var("flag0", 8).unwrap();
    let c = second.bitvec_const(8, 0x41).unwrap();
    let root = second.eq(x, c).unwrap();
    let second = second.finish_with_root(root).unwrap();

    assert_eq!(canonical_bytes(&first), canonical_bytes(&second));
    assert_eq!(canonical_hash(&first), canonical_hash(&second));
}

#[test]
fn hash_consing_reuses_identical_nodes() {
    let mut builder = Builder::new();
    let a = builder.bitvec_const(8, 1).unwrap();
    let b = builder.bitvec_const(8, 1).unwrap();

    assert_eq!(a, b);
}
