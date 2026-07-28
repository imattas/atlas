//! Specialized strategy tests.

use atlas_strategies::{
    recognize_rsa_small_private_exponent, record_lattice_basis_variants, solve_gf2,
    solve_modular_linear,
};

#[test]
fn gf2_solver_matches_exhaustive_small_system() {
    let matrix = vec![vec![true, true], vec![true, false]];
    let rhs = vec![true, false];

    let solution = solve_gf2(&matrix, &rhs).unwrap();

    assert_eq!(solution.assignments, vec![false, true]);
}

#[test]
fn modular_solver_handles_invertible_pivots() {
    let matrix = vec![vec![2, 1], vec![1, 1]];
    let rhs = vec![1, 2];

    let solution = solve_modular_linear(&matrix, &rhs, 5).unwrap();

    assert_eq!(solution.residues, vec![4, 3]);
}

#[test]
fn modular_solver_rejects_non_invertible_pivot_systems() {
    let matrix = vec![vec![2]];
    let rhs = vec![1];

    assert!(solve_modular_linear(&matrix, &rhs, 4).is_none());
}

#[test]
fn lattice_strategy_records_multiple_basis_variants() {
    let variants = record_lattice_basis_variants(&[vec![1, 2], vec![3, 4]]);

    assert_eq!(variants.len(), 2);
    assert_eq!(variants[0].variant, "direct");
    assert_eq!(variants[1].rows[0], vec![2, 1]);
}

#[test]
fn crypto_recognizer_requires_confirmation() {
    let recognition = recognize_rsa_small_private_exponent(512, 65_537).unwrap();

    assert_eq!(recognition.name, "rsa-small-private-exponent");
    assert!(recognition.requires_confirmation);
}
