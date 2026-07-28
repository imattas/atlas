//! Native exact math kernel tests.

use atlas_math::{ModularLinearSystem, Polynomial, Rational};

#[test]
fn ctf_encoding_and_xor_utilities_are_native_and_deterministic() {
    assert_eq!(atlas_math::hex_decode("48656c6c6f").unwrap(), b"Hello");
    assert_eq!(atlas_math::hex_encode(b"Hello"), "48656c6c6f");
    assert_eq!(atlas_math::base64_decode("SGVsbG8=").unwrap(), b"Hello");
    assert_eq!(atlas_math::base64_encode(b"Hello"), "SGVsbG8=");
    assert_eq!(
        atlas_math::repeating_xor(b"ICE", b"\x01\x02"),
        vec![72, 65, 68]
    );
}

#[test]
fn ctf_classical_and_padding_helpers_cover_common_challenge_inputs() {
    assert_eq!(
        atlas_math::caesar_shift(b"Khoor, Zruog!", -3),
        b"Hello, World!"
    );
    assert_eq!(
        atlas_math::pkcs7_unpad(&[1, 2, 3, 3, 3]).unwrap(),
        &[1, 2][..]
    );
    assert!(atlas_math::pkcs7_unpad(&[1, 2, 3, 0]).is_none());
}

#[test]
fn ctf_hash_surface_has_known_sha256_vector() {
    assert_eq!(
        atlas_math::sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

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

#[test]
fn crt_and_modular_exponentiation_support_ctf_crypto_from_scratch() {
    let combined = atlas_math::chinese_remainder(&[(2, 3), (3, 5), (2, 7)]).unwrap();

    assert_eq!(combined, (23, 105));
    assert_eq!(atlas_math::mod_pow(4, 13, 497), Some(445));
    assert_eq!(atlas_math::chinese_remainder(&[(1, 4), (2, 6)]), None);
    assert_eq!(atlas_math::mod_pow(4, 13, 0), None);
}

#[test]
fn modular_square_roots_over_prime_fields_use_tonelli_shanks_from_scratch() {
    assert_eq!(atlas_math::mod_sqrt_prime(10, 13), Some(vec![6, 7]));
    assert_eq!(atlas_math::mod_sqrt_prime(56, 101), Some(vec![37, 64]));
    assert_eq!(atlas_math::mod_sqrt_prime(5, 13), None);
}

#[test]
fn discrete_log_prime_uses_baby_step_giant_step_from_scratch() {
    assert_eq!(atlas_math::discrete_log_prime(2, 22, 29), Some(26));
    assert_eq!(atlas_math::discrete_log_prime(5, 1, 23), Some(0));
    assert_eq!(atlas_math::discrete_log_prime(4, 3, 17), None);
}

#[test]
fn berlekamp_massey_recovers_ctf_lfsr_recurrence_from_scratch() {
    let stream = lfsr_bits(&[true, false, true], &[true, false, false], 21);
    let observed = &stream[..16];

    let recurrence = atlas_math::berlekamp_massey_gf2(observed).unwrap();

    assert_eq!(recurrence.linear_complexity(), 3);
    assert_eq!(recurrence.coefficients(), &[true, false, true]);
    assert_eq!(
        (0..5)
            .scan(observed.to_vec(), |state, _| {
                let next = recurrence.predict_next(state).unwrap();
                state.push(next);
                Some(next)
            })
            .collect::<Vec<_>>(),
        stream[16..21]
    );
}

#[test]
fn berlekamp_massey_handles_constant_and_empty_streams() {
    let zero = atlas_math::berlekamp_massey_gf2(&[false, false, false]).unwrap();
    assert_eq!(zero.linear_complexity(), 0);
    assert_eq!(zero.coefficients(), &[]);
    assert_eq!(zero.predict_next(&[false, false, false]), Some(false));

    assert!(atlas_math::berlekamp_massey_gf2(&[]).is_none());
}

fn lfsr_bits(coefficients: &[bool], seed: &[bool], count: usize) -> Vec<bool> {
    let mut stream = seed.to_vec();
    while stream.len() < count {
        let offset = stream.len() - coefficients.len();
        let next =
            coefficients
                .iter()
                .enumerate()
                .fold(false, |accumulator, (index, coefficient)| {
                    accumulator ^ (*coefficient && stream[offset + index])
                });
        stream.push(next);
    }
    stream
}
