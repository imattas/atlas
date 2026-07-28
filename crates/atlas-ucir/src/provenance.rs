//! Source provenance metadata.

/// A source location attached to a UCIR node.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceLocation {
    file: String,
    line: u32,
    column: u32,
}

impl SourceLocation {
    /// Creates a source location.
    #[must_use]
    pub fn new(file: impl Into<String>, line: u32, column: u32) -> Self {
        Self {
            file: file.into(),
            line,
            column,
        }
    }

    /// Source file or artifact name.
    #[must_use]
    pub fn file(&self) -> &str {
        &self.file
    }

    /// One-based line number.
    #[must_use]
    pub fn line(&self) -> u32 {
        self.line
    }

    /// One-based column number.
    #[must_use]
    pub fn column(&self) -> u32 {
        self.column
    }
}
