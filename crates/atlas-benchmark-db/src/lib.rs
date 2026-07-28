//! Versioned benchmark warehouse records.

use std::collections::BTreeMap;

/// Benchmark record schema version.
pub const BENCHMARK_SCHEMA_MAJOR: u32 = 1;

/// Benchmark corpus split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CorpusSplit {
    /// Training corpus.
    Train,
    /// Test corpus.
    Test,
}

/// Historical benchmark record.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkRecord {
    /// Schema major version.
    pub schema_major: u32,
    /// Challenge id.
    pub challenge_id: String,
    /// Strategy id.
    pub strategy_id: String,
    /// Extracted numeric features.
    pub features: BTreeMap<String, f64>,
    /// Runtime in milliseconds.
    pub runtime_ms: u64,
    /// Corpus split.
    pub split: CorpusSplit,
}

/// Ingestion error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkError {
    /// Incompatible schema version.
    IncompatibleSchema,
    /// Required field or feature is missing.
    MissingField(String),
}

/// In-memory benchmark warehouse.
#[derive(Debug, Clone, Default)]
pub struct BenchmarkWarehouse {
    records: Vec<BenchmarkRecord>,
}

impl BenchmarkWarehouse {
    /// Creates an empty warehouse.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingests one benchmark record.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible schema or missing required fields.
    pub fn ingest(&mut self, record: BenchmarkRecord) -> Result<(), BenchmarkError> {
        if record.schema_major != BENCHMARK_SCHEMA_MAJOR {
            return Err(BenchmarkError::IncompatibleSchema);
        }
        if record.challenge_id.is_empty() {
            return Err(BenchmarkError::MissingField("challenge_id".to_owned()));
        }
        if record.strategy_id.is_empty() {
            return Err(BenchmarkError::MissingField("strategy_id".to_owned()));
        }
        if record.features.is_empty() {
            return Err(BenchmarkError::MissingField("features".to_owned()));
        }
        self.records.push(record);
        Ok(())
    }

    /// Returns all records for a split.
    #[must_use]
    pub fn split(&self, split: CorpusSplit) -> Vec<&BenchmarkRecord> {
        self.records
            .iter()
            .filter(|record| record.split == split)
            .collect()
    }
}
