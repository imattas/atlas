//! Versioned function-summary library.

use std::collections::BTreeMap;

/// Summary manifest schema version.
pub const SUMMARY_SCHEMA_MAJOR: u32 = 1;

/// Function summary manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryManifest {
    /// Schema major version.
    pub schema_major: u32,
    /// Symbol name.
    pub symbol: String,
    /// ABI name.
    pub abi: String,
    /// Summary version.
    pub version: u32,
    /// Memory effect description.
    pub memory_effect: MemoryEffect,
    /// Return effect.
    pub return_effect: ReturnEffect,
    /// Provenance string.
    pub provenance: String,
}

/// Memory effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryEffect {
    /// No memory writes.
    ReadOnly,
    /// Writes bounded output.
    WritesBounded {
        /// Maximum number of bytes written.
        max_bytes: usize,
    },
}

/// Return effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReturnEffect {
    /// Returns bounded length.
    BoundedLength {
        /// Maximum returned length.
        max: usize,
    },
    /// Returns checksum modulo value.
    Checksum {
        /// Checksum modulus.
        modulus: u64,
    },
    /// Returns error code on invalid input.
    ErrorCode {
        /// Error code returned for invalid input.
        code: i32,
    },
}

/// Summary registry error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SummaryError {
    /// Incompatible schema.
    IncompatibleSchema,
    /// Ambiguous summary.
    Ambiguous,
    /// Unsupported call.
    Unsupported,
}

/// Summary registry.
#[derive(Debug, Clone, Default)]
pub struct SummaryRegistry {
    summaries: BTreeMap<(String, String, u32), SummaryManifest>,
}

impl SummaryRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a manifest.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible schemas or ambiguous duplicate keys.
    pub fn register(&mut self, manifest: SummaryManifest) -> Result<(), SummaryError> {
        if manifest.schema_major != SUMMARY_SCHEMA_MAJOR {
            return Err(SummaryError::IncompatibleSchema);
        }
        let key = (
            manifest.symbol.clone(),
            manifest.abi.clone(),
            manifest.version,
        );
        if self.summaries.contains_key(&key) {
            return Err(SummaryError::Ambiguous);
        }
        self.summaries.insert(key, manifest);
        Ok(())
    }

    /// Resolves an exact symbol/ABI/version summary.
    ///
    /// # Errors
    ///
    /// Returns an error when the summary is unsupported.
    pub fn resolve(
        &self,
        symbol: &str,
        abi: &str,
        version: u32,
    ) -> Result<&SummaryManifest, SummaryError> {
        self.summaries
            .get(&(symbol.to_owned(), abi.to_owned(), version))
            .ok_or(SummaryError::Unsupported)
    }
}

/// Built-in bounded libc summaries used by authorized fixtures.
#[must_use]
pub fn builtin_libc() -> Vec<SummaryManifest> {
    vec![
        SummaryManifest {
            schema_major: SUMMARY_SCHEMA_MAJOR,
            symbol: "strlen".to_owned(),
            abi: "sysv".to_owned(),
            version: 1,
            memory_effect: MemoryEffect::ReadOnly,
            return_effect: ReturnEffect::BoundedLength { max: 4096 },
            provenance: "libc bounded string fixture".to_owned(),
        },
        SummaryManifest {
            schema_major: SUMMARY_SCHEMA_MAJOR,
            symbol: "crc32".to_owned(),
            abi: "sysv".to_owned(),
            version: 1,
            memory_effect: MemoryEffect::ReadOnly,
            return_effect: ReturnEffect::Checksum {
                modulus: u64::from(u32::MAX) + 1,
            },
            provenance: "authorized checksum fixture".to_owned(),
        },
    ]
}

/// Concrete bounded strlen reference used for differential tests.
#[must_use]
pub fn bounded_strlen(bytes: &[u8], max: usize) -> usize {
    bytes
        .iter()
        .take(max)
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len().min(max))
}
