//! GF(2) linear solving.

/// GF(2) assignment solution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gf2Solution {
    /// Boolean variable assignments.
    pub assignments: Vec<bool>,
}

/// Affine GF(2) solution for rectangular systems.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gf2AffineSolution {
    /// One assignment satisfying the system.
    pub particular: Vec<bool>,
    /// Columns that can be chosen freely.
    pub free_columns: Vec<usize>,
    /// Nullspace basis vectors, one per free column.
    pub basis: Vec<Vec<bool>>,
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

/// Solves a rectangular GF(2) linear system and returns an affine solution.
#[must_use]
pub fn solve_gf2_affine(matrix: &[Vec<bool>], rhs: &[bool]) -> Option<Gf2AffineSolution> {
    if matrix.len() != rhs.len() {
        return None;
    }
    let variable_count = matrix.first().map_or(0, Vec::len);
    if variable_count == 0 || matrix.iter().any(|row| row.len() != variable_count) {
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

    let mut pivot_columns = Vec::new();
    let mut pivot_row = 0;
    for column in 0..variable_count {
        let Some(pivot) = (pivot_row..rows.len()).find(|row| rows[*row][column]) else {
            continue;
        };
        rows.swap(pivot_row, pivot);
        let pivot_tail = rows[pivot_row][column..=variable_count].to_vec();
        for (row_index, row) in rows.iter_mut().enumerate() {
            if row_index != pivot_row && row[column] {
                for (cell, pivot_cell) in row[column..=variable_count].iter_mut().zip(&pivot_tail) {
                    *cell ^= *pivot_cell;
                }
            }
        }
        pivot_columns.push(column);
        pivot_row += 1;
    }

    if rows[pivot_row..].iter().any(|row| {
        row[..variable_count]
            .iter()
            .all(|coefficient| !*coefficient)
            && row[variable_count]
    }) {
        return None;
    }

    let mut is_pivot = vec![false; variable_count];
    for &column in &pivot_columns {
        is_pivot[column] = true;
    }
    let free_columns: Vec<usize> = (0..variable_count)
        .filter(|column| !is_pivot[*column])
        .collect();

    let mut particular = vec![false; variable_count];
    for (row, &column) in pivot_columns.iter().enumerate() {
        particular[column] = rows[row][variable_count];
    }

    let basis = free_columns
        .iter()
        .map(|&free_column| {
            let mut vector = vec![false; variable_count];
            vector[free_column] = true;
            for (row, &pivot_column) in pivot_columns.iter().enumerate() {
                vector[pivot_column] = rows[row][free_column];
            }
            vector
        })
        .collect();

    Some(Gf2AffineSolution {
        particular,
        free_columns,
        basis,
    })
}
