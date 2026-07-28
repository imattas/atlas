//! Symbolic memory with alias-aware byte cells.

use std::collections::BTreeMap;

/// Sparse symbolic memory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SymbolicMemory {
    cells: BTreeMap<u64, String>,
    aliases: BTreeMap<String, u64>,
}

impl SymbolicMemory {
    /// Stores a symbolic byte.
    pub fn store(&mut self, address: u64, value: impl Into<String>) {
        self.cells.insert(address, value.into());
    }

    /// Loads a symbolic byte.
    #[must_use]
    pub fn load(&self, address: u64) -> Option<&str> {
        self.cells.get(&address).map(String::as_str)
    }

    /// Assigns an alias to a concrete address.
    pub fn alias(&mut self, name: impl Into<String>, address: u64) {
        self.aliases.insert(name.into(), address);
    }

    /// Loads through an alias.
    #[must_use]
    pub fn load_alias(&self, name: &str) -> Option<&str> {
        self.aliases
            .get(name)
            .and_then(|address| self.load(*address))
    }
}
