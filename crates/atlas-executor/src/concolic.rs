//! Concolic seed mutation.

/// Concrete seed used to guide symbolic exploration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcolicSeed {
    /// Seed bytes.
    pub bytes: Vec<u8>,
}

/// Deterministic concolic seed mutator.
pub struct SeedMutator;

impl SeedMutator {
    /// Mutates one byte by applying an XOR mask.
    #[must_use]
    pub fn flip(seed: &ConcolicSeed, index: usize, mask: u8) -> ConcolicSeed {
        let mut bytes = seed.bytes.clone();
        if let Some(byte) = bytes.get_mut(index) {
            *byte ^= mask;
        }
        ConcolicSeed { bytes }
    }
}
