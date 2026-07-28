//! Native exact math kernel tests.

use atlas_math::{ModularLinearSystem, Polynomial, Rational};

#[test]
fn exact_rationals_normalize_and_preserve_arithmetic() {
    let left = Rational::new(2, 4).unwrap();
    let right = Rational::new(-3, 9).unwrap();

    assert_eq!(left.to_string(), "1/2");
    assert_eq!(right.to_string(), "-1/3");
    assert_eq!((left + right).to_string(), "1/6");
    assert_eq!((left * right).to_string(), "-1/6");
    assert!(Rational::new(1, 0).is_none());
}

#[test]
fn modular_linear_solver_handles_prime_fields_from_scratch() {
    // 2x + 3y = 1 mod 7
    // 4x +  y = 6 mod 7
    let system = ModularLinearSystem::new(7, vec![vec![2, 3], vec![4, 1]], vec![1, 6]).unwrap();
    let solution = system.solve().unwrap();

    assert_eq!(solution, vec![1, 2]);
}

#[test]
fn polynomial_gcd_over_prime_field_is_monic() {
    // (x + 1)(x + 2) and (x + 1)(x + 3) over GF(5) share x + 1.
    let left = Polynomial::new(5, vec![2, 3, 1]).unwrap();
    let right = Polynomial::new(5, vec![3, 4, 1]).unwrap();

    assert_eq!(
        left.gcd(&right).unwrap(),
        Polynomial::new(5, vec![1, 1]).unwrap()
    );
}

#[test]
fn bitvector_expression_solver_does_not_require_external_smt() {
    let matches = atlas_math::solve_u8_xor_eq(0xaa, 0xff);

    assert_eq!(matches, vec![0x55]);
}
