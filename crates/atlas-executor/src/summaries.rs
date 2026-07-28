//! Versioned function summaries.

use std::collections::BTreeMap;

/// Pure function summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionSummary {
    /// Summary version.
    pub version: u32,
    /// Function name.
    pub name: String,
    /// Symbolic effect description.
    pub effect: String,
}

/// Summary store.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SummaryStore {
    summaries: BTreeMap<String, FunctionSummary>,
}

impl SummaryStore {
    /// Inserts a summary.
    pub fn insert(&mut self, summary: FunctionSummary) {
        self.summaries.insert(summary.name.clone(), summary);
    }

    /// Gets a summary.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&FunctionSummary> {
        self.summaries.get(name)
    }
}
