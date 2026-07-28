//! GF(2) linear solving.

/// GF(2) assignment solution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gf2Solution {
    /// Boolean variable assignments.
    pub assignments: Vec<bool>,
}

/// Solves a square GF(2) linear system by Gaussian elimination.
#[must_use]
pub fn solve_gf2(matrix: &[Vec<bool>], rhs: &[bool]) -> Option<Gf2Solution> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut rows: Vec<Vec<bool>> = matrix
        .iter()
        .zip(rhs)
        .map(|(row, value)| {
            let mut row = row.clone();
            row.push(*value);
            row
        })
        .collect();

    for column in 0..n {
        let pivot = (column..n).find(|row| rows[*row][column])?;
        rows.swap(column, pivot);
        for row in 0..n {
            if row != column && rows[row][column] {
                let pivot_tail = rows[column][column..=n].to_vec();
                for (cell, pivot_cell) in rows[row][column..=n].iter_mut().zip(pivot_tail) {
                    *cell ^= pivot_cell;
                }
            }
        }
    }

    Some(Gf2Solution {
        assignments: rows.iter().map(|row| row[n]).collect(),
    })
}
