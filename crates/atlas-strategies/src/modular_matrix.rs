//! Modular linear algebra.

/// Modular linear solution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModularSolution {
    /// Residues modulo `modulus`.
    pub residues: Vec<u64>,
    /// Modulus.
    pub modulus: u64,
}

/// Solves a square modular linear system with invertible pivots.
#[must_use]
pub fn solve_modular_linear(
    matrix: &[Vec<u64>],
    rhs: &[u64],
    modulus: u64,
) -> Option<ModularSolution> {
    if modulus < 2 {
        return None;
    }
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut rows: Vec<Vec<u64>> = matrix
        .iter()
        .zip(rhs)
        .map(|(row, value)| {
            let mut row: Vec<u64> = row.iter().map(|value| value % modulus).collect();
            row.push(value % modulus);
            row
        })
        .collect();

    for column in 0..n {
        let pivot = (column..n).find(|row| inverse_mod(rows[*row][column], modulus).is_some())?;
        rows.swap(column, pivot);
        let inverse = inverse_mod(rows[column][column], modulus)?;
        for cell in &mut rows[column][column..=n] {
            *cell = (*cell * inverse) % modulus;
        }
        for row in 0..n {
            if row != column {
                let factor = rows[row][column];
                let pivot_tail = rows[column][column..=n].to_vec();
                for (cell, pivot_cell) in rows[row][column..=n].iter_mut().zip(pivot_tail) {
                    *cell = (modulus + *cell - ((factor * pivot_cell) % modulus)) % modulus;
                }
            }
        }
    }

    Some(ModularSolution {
        residues: rows.iter().map(|row| row[n]).collect(),
        modulus,
    })
}

fn inverse_mod(value: u64, modulus: u64) -> Option<u64> {
    (1..modulus).find(|candidate| (value * candidate) % modulus == 1)
}
