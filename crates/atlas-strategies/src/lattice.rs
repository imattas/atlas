//! Lattice basis variant recording.

/// Recorded lattice basis formulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatticeBasis {
    /// Variant name.
    pub variant: String,
    /// Basis rows.
    pub rows: Vec<Vec<i64>>,
}

/// Records deterministic basis variants for later backend execution.
#[must_use]
pub fn record_lattice_basis_variants(base: &[Vec<i64>]) -> Vec<LatticeBasis> {
    vec![
        LatticeBasis {
            variant: "direct".to_owned(),
            rows: base.to_vec(),
        },
        LatticeBasis {
            variant: "reversed".to_owned(),
            rows: base
                .iter()
                .map(|row| row.iter().rev().copied().collect())
                .collect(),
        },
    ]
}
