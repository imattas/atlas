//! Integrity-checked content cache.

use std::collections::BTreeMap;

/// Cache lookup failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    /// Entry bytes do not match their integrity hash.
    Corrupt,
}

/// In-memory content cache used by tests and the initial runtime.
#[derive(Debug, Clone, Default)]
pub struct Cache {
    entries: BTreeMap<String, Entry>,
}

#[derive(Debug, Clone)]
struct Entry {
    bytes: Vec<u8>,
    hash: [u8; 32],
}

impl Cache {
    /// Creates an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores bytes under a content-addressed key.
    pub fn put(&mut self, key: impl Into<String>, bytes: impl Into<Vec<u8>>) {
        let bytes = bytes.into();
        let hash = hash(&bytes);
        self.entries.insert(key.into(), Entry { bytes, hash });
    }

    /// Returns a verified cached entry.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError::Corrupt`] when integrity verification fails.
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>, CacheError> {
        let Some(entry) = self.entries.get(key) else {
            return Ok(None);
        };
        if hash(&entry.bytes) != entry.hash {
            return Err(CacheError::Corrupt);
        }
        Ok(Some(entry.bytes.clone()))
    }

    /// Test/support hook that simulates on-disk corruption.
    pub fn corrupt_for_test(&mut self, key: &str) {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.bytes.push(0xff);
        }
    }
}

fn hash(bytes: &[u8]) -> [u8; 32] {
    let mut out = [0_u8; 32];
    for (index, byte) in bytes.iter().enumerate() {
        let slot = index % out.len();
        out[slot] = out[slot].wrapping_mul(31).wrapping_add(*byte);
        out[slot] ^= u8::try_from(index & 0xff).unwrap_or(0);
    }
    out
}
