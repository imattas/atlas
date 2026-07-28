//! Pure call summaries.

use std::collections::BTreeMap;

/// Pure function call summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSummary {
    /// Function name.
    pub name: String,
    /// Versioned summary effect.
    pub effect: String,
    /// Summary version.
    pub version: u32,
}

/// Summary registry.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SummaryRegistry {
    summaries: BTreeMap<String, CallSummary>,
}

impl SummaryRegistry {
    /// Registers a summary.
    pub fn register(&mut self, summary: CallSummary) {
        self.summaries.insert(summary.name.clone(), summary);
    }

    /// Retrieves a summary.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&CallSummary> {
        self.summaries.get(name)
    }
}
