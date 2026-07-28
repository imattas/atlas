//! Native Atlas math kernel.

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

/// Solves an 8-bit XOR equality from scratch.
#[must_use]
pub fn solve_u8_xor_eq(mask: u8, target: u8) -> Vec<u8> {
    vec![mask ^ target]
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
