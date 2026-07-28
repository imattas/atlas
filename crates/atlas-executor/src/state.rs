//! Symbolic machine state.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::SymbolicMemory;

/// Copy-on-write symbolic state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicState {
    registers: Arc<BTreeMap<String, String>>,
    /// Symbolic memory.
    pub memory: SymbolicMemory,
    path_predicates: Arc<Vec<String>>,
}

impl Default for SymbolicState {
    fn default() -> Self {
        Self {
            registers: Arc::new(BTreeMap::new()),
            memory: SymbolicMemory::default(),
            path_predicates: Arc::new(Vec::new()),
        }
    }
}

impl SymbolicState {
    /// Creates an empty state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads a symbolic register expression.
    #[must_use]
    pub fn register(&self, name: &str) -> Option<&str> {
        self.registers.get(name).map(String::as_str)
    }

    /// Writes a symbolic register expression.
    pub fn write_register(&mut self, name: impl Into<String>, value: impl Into<String>) {
        Arc::make_mut(&mut self.registers).insert(name.into(), value.into());
    }

    /// Adds a path predicate.
    pub fn assume(&mut self, predicate: impl Into<String>) {
        Arc::make_mut(&mut self.path_predicates).push(predicate.into());
    }

    /// Returns path predicates.
    #[must_use]
    pub fn path_predicates(&self) -> &[String] {
        &self.path_predicates
    }
}
