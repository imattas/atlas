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
