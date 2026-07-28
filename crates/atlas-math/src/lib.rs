//! Native Atlas math kernel.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::ops::{Add, Mul};

/// Exact rational number backed by normalized signed integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rational {
    numerator: i64,
    denominator: i64,
}

impl Rational {
    /// Creates a normalized rational.
    #[must_use]
    pub fn new(numerator: i64, denominator: i64) -> Option<Self> {
        if denominator == 0 {
            return None;
        }
        let sign = if denominator < 0 { -1 } else { 1 };
        let gcd = gcd_i64(numerator, denominator);
        Some(Self {
            numerator: sign * numerator / gcd,
            denominator: sign * denominator / gcd,
        })
    }
}

impl Add for Rational {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let numerator = self
            .numerator
            .saturating_mul(rhs.denominator)
            .saturating_add(rhs.numerator.saturating_mul(self.denominator));
        let denominator = self.denominator.saturating_mul(rhs.denominator);
        Self::new(numerator, denominator).expect("non-zero rational denominator")
    }
}

impl Mul for Rational {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.numerator.saturating_mul(rhs.numerator),
            self.denominator.saturating_mul(rhs.denominator),
        )
        .expect("non-zero rational denominator")
    }
}

impl Display for Rational {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        if self.denominator == 1 {
            write!(formatter, "{}", self.numerator)
        } else {
            write!(formatter, "{}/{}", self.numerator, self.denominator)
        }
    }
}

/// Linear system over a prime modular field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularLinearSystem {
    modulus: u64,
    matrix: Vec<Vec<u64>>,
    rhs: Vec<u64>,
}

impl ModularLinearSystem {
    /// Creates a validated square modular linear system.
    #[must_use]
    pub fn new(modulus: u64, matrix: Vec<Vec<u64>>, rhs: Vec<u64>) -> Option<Self> {
        let size = rhs.len();
        if modulus < 2
            || !is_prime(modulus)
            || matrix.len() != size
            || matrix.iter().any(|row| row.len() != size)
        {
            return None;
        }
        Some(Self {
            modulus,
            matrix: matrix
                .into_iter()
                .map(|row| row.into_iter().map(|value| value % modulus).collect())
                .collect(),
            rhs: rhs.into_iter().map(|value| value % modulus).collect(),
        })
    }

    /// Solves the system by modular Gaussian elimination.
    #[must_use]
    pub fn solve(&self) -> Option<Vec<u64>> {
        let size = self.rhs.len();
        let mut rows: Vec<Vec<u64>> = self
            .matrix
            .iter()
            .zip(&self.rhs)
            .map(|(row, value)| {
                let mut row = row.clone();
                row.push(*value);
                row
            })
            .collect();

        for column in 0..size {
            let pivot = (column..size).find(|row| rows[*row][column] != 0)?;
            rows.swap(column, pivot);
            let inverse = mod_inverse(rows[column][column], self.modulus)?;
            for cell in &mut rows[column][column..=size] {
                *cell = (*cell * inverse) % self.modulus;
            }
            let pivot_tail = rows[column][column..=size].to_vec();
            for (row_index, row) in rows.iter_mut().enumerate() {
                if row_index == column {
                    continue;
                }
                let factor = row[column];
                for (cell, pivot_cell) in row[column..=size].iter_mut().zip(&pivot_tail) {
                    *cell = (*cell + self.modulus - (factor * *pivot_cell) % self.modulus)
                        % self.modulus;
                }
            }
        }
        Some(rows.iter().map(|row| row[size]).collect())
    }
}

/// Polynomial over a prime field, coefficients in ascending degree order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Polynomial {
    modulus: u64,
    coefficients: Vec<u64>,
}

impl Polynomial {
    /// Creates a normalized polynomial over a prime field.
    #[must_use]
    pub fn new(modulus: u64, coefficients: Vec<u64>) -> Option<Self> {
        if modulus < 2 || !is_prime(modulus) || coefficients.is_empty() {
            return None;
        }
        let mut polynomial = Self {
            modulus,
            coefficients: coefficients
                .into_iter()
                .map(|value| value % modulus)
                .collect(),
        };
        polynomial.trim();
        Some(polynomial)
    }

    /// Computes the monic greatest common divisor.
    #[must_use]
    pub fn gcd(&self, rhs: &Self) -> Option<Self> {
        if self.modulus != rhs.modulus {
            return None;
        }
        let mut left = self.clone();
        let mut right = rhs.clone();
        while !right.is_zero() {
            let remainder = left.remainder(&right)?;
            left = right;
            right = remainder;
        }
        left.monic()
    }

    fn degree(&self) -> usize {
        self.coefficients.len().saturating_sub(1)
    }

    fn is_zero(&self) -> bool {
        self.coefficients.len() == 1 && self.coefficients[0] == 0
    }

    fn trim(&mut self) {
        while self.coefficients.len() > 1 && self.coefficients.last() == Some(&0) {
            self.coefficients.pop();
        }
    }

    fn monic(mut self) -> Option<Self> {
        let leading = *self.coefficients.last()?;
        let inverse = mod_inverse(leading, self.modulus)?;
        for coefficient in &mut self.coefficients {
            *coefficient = (*coefficient * inverse) % self.modulus;
        }
        Some(self)
    }

    fn remainder(&self, divisor: &Self) -> Option<Self> {
        if divisor.is_zero() || self.modulus != divisor.modulus {
            return None;
        }
        let mut remainder = self.clone();
        let divisor_leading = *divisor.coefficients.last()?;
        let divisor_inverse = mod_inverse(divisor_leading, self.modulus)?;
        while !remainder.is_zero() && remainder.degree() >= divisor.degree() {
            let degree_delta = remainder.degree() - divisor.degree();
            let factor = (*remainder.coefficients.last()? * divisor_inverse) % self.modulus;
            for (index, coefficient) in divisor.coefficients.iter().enumerate() {
                let target = index + degree_delta;
                remainder.coefficients[target] = (remainder.coefficients[target] + self.modulus
                    - (factor * *coefficient) % self.modulus)
                    % self.modulus;
            }
            remainder.trim();
        }
        Some(remainder)
    }
}

/// Minimal GF(2) linear recurrence recovered from a bit stream.
///
/// Coefficients are ordered from oldest to newest history bit. For coefficients
/// `[c0, c1, ...]`, prediction is `xor(c_i & history[history.len() - L + i])`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gf2LinearRecurrence {
    coefficients: Vec<bool>,
}

impl Gf2LinearRecurrence {
    /// Returns the linear complexity, equal to the number of previous bits
    /// needed to predict the next bit.
    #[must_use]
    pub fn linear_complexity(&self) -> usize {
        self.coefficients.len()
    }

    /// Returns recurrence coefficients ordered from oldest to newest bit.
    #[must_use]
    pub fn coefficients(&self) -> &[bool] {
        &self.coefficients
    }

    /// Predicts the next bit from a history prefix.
    #[must_use]
    pub fn predict_next(&self, history: &[bool]) -> Option<bool> {
        let complexity = self.linear_complexity();
        if history.len() < complexity {
            return None;
        }
        let start = history.len() - complexity;
        Some(self.coefficients.iter().enumerate().fold(
            false,
            |accumulator, (index, coefficient)| {
                accumulator ^ (*coefficient && history[start + index])
            },
        ))
    }
}

/// Recovers the shortest GF(2) linear recurrence for a bit stream with the
/// Berlekamp-Massey algorithm.
#[must_use]
pub fn berlekamp_massey_gf2(stream: &[bool]) -> Option<Gf2LinearRecurrence> {
    if stream.is_empty() {
        return None;
    }

    let mut connection = vec![false; stream.len() + 1];
    let mut previous = vec![false; stream.len() + 1];
    connection[0] = true;
    previous[0] = true;
    let mut complexity = 0_usize;
    let mut shift = 1_usize;

    for index in 0..stream.len() {
        let discrepancy = (1..=complexity).fold(stream[index], |accumulator, offset| {
            accumulator ^ (connection[offset] && stream[index - offset])
        });
        if !discrepancy {
            shift += 1;
            continue;
        }

        let before_update = connection.clone();
        for previous_index in 0..(stream.len() + 1 - shift) {
            if previous[previous_index] {
                connection[previous_index + shift] ^= true;
            }
        }
        if 2 * complexity <= index {
            complexity = index + 1 - complexity;
            previous = before_update;
            shift = 1;
        } else {
            shift += 1;
        }
    }

    let coefficients = (1..=complexity)
        .rev()
        .map(|offset| connection[offset])
        .collect();
    Some(Gf2LinearRecurrence { coefficients })
}

/// Solves an 8-bit XOR equality from scratch.
#[must_use]
pub fn solve_u8_xor_eq(mask: u8, target: u8) -> Vec<u8> {
    vec![mask ^ target]
}

/// Computes `(base ^ exponent) mod modulus` by square-and-multiply.
#[must_use]
pub fn mod_pow(base: u64, mut exponent: u64, modulus: u64) -> Option<u64> {
    if modulus == 0 {
        return None;
    }
    let modulus_wide = u128::from(modulus);
    let mut result = 1_u128 % modulus_wide;
    let mut base = u128::from(base) % modulus_wide;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = (result * base) % modulus_wide;
        }
        base = (base * base) % modulus_wide;
        exponent >>= 1;
    }
    u64::try_from(result).ok()
}

/// Combines pairwise-coprime congruences with the Chinese Remainder Theorem.
///
/// Each input pair is `(residue, modulus)`. The return value is the normalized
/// `(residue, modulus)` for the combined congruence.
#[must_use]
pub fn chinese_remainder(congruences: &[(u64, u64)]) -> Option<(u64, u64)> {
    let (&(first_residue, first_modulus), rest) = congruences.split_first()?;
    if first_modulus == 0 {
        return None;
    }
    let mut residue = u128::from(first_residue % first_modulus);
    let mut modulus = u128::from(first_modulus);
    for &(next_residue, next_modulus) in rest {
        if next_modulus == 0 {
            return None;
        }
        let next_modulus = u128::from(next_modulus);
        let next_residue = u128::from(next_residue) % next_modulus;
        if gcd_u128(modulus, next_modulus) != 1 {
            return None;
        }
        let inverse = mod_inverse_u128(modulus % next_modulus, next_modulus)?;
        let delta = (next_residue + next_modulus - (residue % next_modulus)) % next_modulus;
        let step = (delta * inverse) % next_modulus;
        residue += modulus * step;
        modulus *= next_modulus;
        residue %= modulus;
    }
    Some((u64::try_from(residue).ok()?, u64::try_from(modulus).ok()?))
}

/// Computes all modular square roots of `value` over an odd prime field.
///
/// Uses Tonelli-Shanks and returns roots in ascending order. `None` means the
/// modulus is invalid or the value is a quadratic non-residue.
#[must_use]
pub fn mod_sqrt_prime(value: u64, modulus: u64) -> Option<Vec<u64>> {
    if !is_prime(modulus) {
        return None;
    }
    let value = value % modulus;
    if value == 0 {
        return Some(vec![0]);
    }
    if modulus == 2 {
        return Some(vec![value]);
    }
    if mod_pow(value, (modulus - 1) / 2, modulus)? != 1 {
        return None;
    }
    if modulus % 4 == 3 {
        return Some(sorted_prime_roots(
            mod_pow(value, (modulus + 1) / 4, modulus)?,
            modulus,
        ));
    }

    let mut odd_factor = modulus - 1;
    let mut two_adic_exponent = 0_u64;
    while odd_factor.is_multiple_of(2) {
        odd_factor /= 2;
        two_adic_exponent += 1;
    }

    let mut non_residue = 2_u64;
    while mod_pow(non_residue, (modulus - 1) / 2, modulus)? != modulus - 1 {
        non_residue += 1;
    }

    let mut residue_factor = mod_pow(non_residue, odd_factor, modulus)?;
    let mut root_candidate = mod_pow(value, odd_factor.div_ceil(2), modulus)?;
    let mut residue_power = mod_pow(value, odd_factor, modulus)?;
    let mut exponent_window = two_adic_exponent;

    while residue_power != 1 {
        let mut witness_index = 1_u64;
        let mut powered_residue = mul_mod(residue_power, residue_power, modulus);
        while witness_index < exponent_window && powered_residue != 1 {
            powered_residue = mul_mod(powered_residue, powered_residue, modulus);
            witness_index += 1;
        }
        if witness_index == exponent_window {
            return None;
        }
        let correction = mod_pow(
            residue_factor,
            1_u64 << (exponent_window - witness_index - 1),
            modulus,
        )?;
        root_candidate = mul_mod(root_candidate, correction, modulus);
        let correction_squared = mul_mod(correction, correction, modulus);
        residue_power = mul_mod(residue_power, correction_squared, modulus);
        residue_factor = correction_squared;
        exponent_window = witness_index;
    }

    Some(sorted_prime_roots(root_candidate, modulus))
}

/// Solves `base^x = target (mod modulus)` over a prime field by
/// baby-step/giant-step.
///
/// Returns the smallest exponent found in the multiplicative group, or `None`
/// if `target` is not in the subgroup generated by `base`.
#[must_use]
pub fn discrete_log_prime(base: u64, target: u64, modulus: u64) -> Option<u64> {
    if !is_prime(modulus) || modulus < 2 {
        return None;
    }
    let base = base % modulus;
    let target = target % modulus;
    if target == 1 {
        return Some(0);
    }
    if base == 0 {
        return None;
    }

    let order = modulus - 1;
    let step = ceil_sqrt(order);
    let mut baby_steps = HashMap::new();
    let mut value = 1_u64;
    for exponent in 0..step {
        baby_steps.entry(value).or_insert(exponent);
        value = mul_mod(value, base, modulus);
    }

    let giant_stride = mod_inverse(mod_pow(base, step, modulus)?, modulus)?;
    let mut gamma = target;
    for giant in 0..=step {
        if let Some(&baby) = baby_steps.get(&gamma) {
            let exponent = giant.saturating_mul(step).saturating_add(baby);
            if exponent < order {
                return Some(exponent);
            }
        }
        gamma = mul_mod(gamma, giant_stride, modulus);
    }
    None
}

/// Decodes hexadecimal bytes, accepting upper- or lower-case digits.
#[must_use]
pub fn hex_decode(input: &str) -> Option<Vec<u8>> {
    let bytes = input.as_bytes();
    if !bytes.len().is_multiple_of(2) {
        return None;
    }
    bytes
        .chunks_exact(2)
        .map(|pair| Some((hex_digit(pair[0])? << 4) | hex_digit(pair[1])?))
        .collect()
}

/// Encodes bytes as lower-case hexadecimal.
#[must_use]
pub fn hex_encode(input: &[u8]) -> String {
    input.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Encodes bytes using RFC 4648 base64 without external dependencies.
#[must_use]
pub fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::new();
    for chunk in input.chunks(3) {
        let a = chunk[0];
        let b = chunk.get(1).copied().unwrap_or(0);
        let c = chunk.get(2).copied().unwrap_or(0);
        output.push(ALPHABET[(a >> 2) as usize] as char);
        output.push(ALPHABET[((a & 3) << 4 | b >> 4) as usize] as char);
        output.push(if chunk.len() > 1 {
            ALPHABET[((b & 15) << 2 | c >> 6) as usize] as char
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            ALPHABET[(c & 63) as usize] as char
        } else {
            '='
        });
    }
    output
}

/// Decodes strict RFC 4648 base64.
#[must_use]
pub fn base64_decode(input: &str) -> Option<Vec<u8>> {
    if input.is_empty() || !input.len().is_multiple_of(4) {
        return if input.is_empty() {
            Some(Vec::new())
        } else {
            None
        };
    }
    let bytes = input.as_bytes();
    let mut output = Vec::new();
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let padding = usize::from(chunk[2] == b'=') + usize::from(chunk[3] == b'=');
        if padding > 2
            || (padding > 0 && index + 1 != bytes.len() / 4)
            || (padding == 1 && chunk[2] == b'=')
        {
            return None;
        }
        let a = base64_digit(chunk[0])?;
        let b = base64_digit(chunk[1])?;
        let c = if chunk[2] == b'=' {
            0
        } else {
            base64_digit(chunk[2])?
        };
        let d = if chunk[3] == b'=' {
            0
        } else {
            base64_digit(chunk[3])?
        };
        if (padding == 2 && b & 15 != 0) || (padding == 1 && c & 3 != 0) {
            return None;
        }
        output.push((a << 2) | (b >> 4));
        if padding < 2 {
            output.push((b << 4) | (c >> 2));
        }
        if padding == 0 {
            output.push((c << 6) | d);
        }
    }
    Some(output)
}

/// Applies a repeating-key XOR operation, a common CTF stream primitive.
#[must_use]
pub fn repeating_xor(input: &[u8], key: &[u8]) -> Vec<u8> {
    if key.is_empty() {
        return Vec::new();
    }
    input
        .iter()
        .enumerate()
        .map(|(index, byte)| byte ^ key[index % key.len()])
        .collect()
}

/// Applies a Caesar shift to ASCII letters while preserving case and symbols.
#[must_use]
pub fn caesar_shift(input: &[u8], shift: i32) -> Vec<u8> {
    input
        .iter()
        .map(|byte| {
            let (start, width) = if byte.is_ascii_lowercase() {
                (b'a', 26)
            } else if byte.is_ascii_uppercase() {
                (b'A', 26)
            } else {
                return *byte;
            };
            start + ((*byte - start) as i32 + shift).rem_euclid(width) as u8
        })
        .collect()
}

/// Removes PKCS#7 padding, returning a slice into the input on success.
#[must_use]
pub fn pkcs7_unpad(input: &[u8]) -> Option<&[u8]> {
    let &padding = input.last()?;
    let length = usize::from(padding);
    if length == 0
        || length > input.len()
        || input[input.len() - length..]
            .iter()
            .any(|byte| *byte != padding)
    {
        return None;
    }
    Some(&input[..input.len() - length])
}

/// Computes SHA-256 and returns the lower-case hexadecimal digest.
#[must_use]
pub fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut data = input.to_vec();
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(
        &(u64::try_from(input.len())
            .unwrap_or(u64::MAX)
            .saturating_mul(8))
        .to_be_bytes(),
    );
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    for block in data.chunks_exact(64) {
        let mut w = [0_u32; 64];
        for (i, word) in w[..16].iter_mut().enumerate() {
            *word = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (value, add) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *value = (*value).wrapping_add(add);
        }
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
fn base64_digit(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn gcd_i64(left: i64, right: i64) -> i64 {
    let mut left = left.unsigned_abs();
    let mut right = right.unsigned_abs();
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    i64::try_from(left.max(1)).unwrap_or(i64::MAX)
}

fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let next = left % right;
        left = right;
        right = next;
    }
    left
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value == 2 {
        return true;
    }
    if value.is_multiple_of(2) {
        return false;
    }
    let mut divisor = 3_u64;
    while divisor.saturating_mul(divisor) <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

fn mul_mod(left: u64, right: u64, modulus: u64) -> u64 {
    u64::try_from((u128::from(left) * u128::from(right)) % u128::from(modulus))
        .expect("modular product fits in u64")
}

fn sorted_prime_roots(root: u64, modulus: u64) -> Vec<u64> {
    let other = (modulus - root) % modulus;
    match root.cmp(&other) {
        Ordering::Equal => vec![root],
        Ordering::Less => vec![root, other],
        Ordering::Greater => vec![other, root],
    }
}

fn ceil_sqrt(value: u64) -> u64 {
    let mut root = 0_u64;
    while u128::from(root) * u128::from(root) < u128::from(value) {
        root += 1;
    }
    root
}

fn mod_inverse(value: u64, modulus: u64) -> Option<u64> {
    let (mut old_r, mut r) = (i128::from(modulus), i128::from(value % modulus));
    let (mut old_s, mut s) = (0_i128, 1_i128);
    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
    }
    if old_r != 1 {
        return None;
    }
    let normalized = old_s.rem_euclid(i128::from(modulus));
    u64::try_from(normalized).ok()
}

fn mod_inverse_u128(value: u128, modulus: u128) -> Option<u128> {
    let (mut old_r, mut r) = (i128::try_from(modulus).ok()?, i128::try_from(value).ok()?);
    let (mut old_s, mut s) = (0_i128, 1_i128);
    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
    }
    if old_r != 1 {
        return None;
    }
    u128::try_from(old_s.rem_euclid(i128::try_from(modulus).ok()?)).ok()
}
