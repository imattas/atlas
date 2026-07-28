//! Structured solve reports.

use atlas_validator::ResultLevel;

/// Versioned solve report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolveReportV1 {
    /// Report schema major version.
    pub schema_major: u32,
    /// Terminal or nonterminal result level.
    pub result_level: ResultLevel,
    /// Input content hash.
    pub input_hash: String,
    /// Reproduction command.
    pub reproduction: String,
    /// Human-readable explanation.
    pub explanation: String,
}

impl SolveReportV1 {
    /// Creates a report.
    #[must_use]
    pub fn new(
        result_level: ResultLevel,
        input_hash: impl Into<String>,
        reproduction: impl Into<String>,
        explanation: impl Into<String>,
    ) -> Self {
        Self {
            schema_major: 1,
            result_level,
            input_hash: input_hash.into(),
            reproduction: reproduction.into(),
            explanation: redact(&explanation.into()),
        }
    }

    /// Encodes a stable JSON object without external dependencies.
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            "{{\"schema_major\":{},\"result_level\":\"{:?}\",\"input_hash\":\"{}\",\"reproduction\":\"{}\",\"explanation\":\"{}\"}}",
            self.schema_major,
            self.result_level,
            escape(&self.input_hash),
            escape(&self.reproduction),
            escape(&self.explanation)
        )
    }
}

fn redact(value: &str) -> String {
    value.replace("SECRET=", "SECRET=<redacted>")
}

fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}
