//! Versioned wire contracts for `AtlasCTF` runtime components.

/// Current supported major schema version.
pub const SCHEMA_MAJOR: u32 = 1;

/// Current minor schema version.
pub const SCHEMA_MINOR: u32 = 0;

const MAGIC: &[u8; 4] = b"ATLS";
const HEADER_LEN: usize = 20;

/// Types corresponding to the `atlas.v1` public protocol.
pub mod v1 {
    pub use crate::{decode_envelope, Envelope, ProtocolError, SCHEMA_MAJOR, SCHEMA_MINOR};
}

/// A small deterministic protocol envelope used by local adapters and tests.
///
/// The full platform schemas live in `schemas/atlas/v1/*.proto`. This envelope
/// is intentionally dependency-free so the protocol crate can validate version
/// compatibility before generated bindings are introduced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Envelope {
    /// Major schema version. Unknown major versions are rejected.
    pub schema_major: u32,
    /// Minor schema version. Minor versions are accepted within the same major.
    pub schema_minor: u32,
    /// UTF-8-ish logical payload type, for example `atlas.v1.ucir.Graph`.
    pub message_type: String,
    /// Opaque encoded payload.
    pub payload: Vec<u8>,
}

impl Envelope {
    /// Creates a new v1 envelope.
    #[must_use]
    pub fn new(message_type: impl Into<String>, payload: impl Into<Vec<u8>>) -> Self {
        Self {
            schema_major: SCHEMA_MAJOR,
            schema_minor: SCHEMA_MINOR,
            message_type: message_type.into(),
            payload: payload.into(),
        }
    }

    /// Encodes the envelope deterministically.
    ///
    /// Format: `ATLS | major | minor | type_len | payload_len | type | payload`,
    /// with all integer fields in big-endian order.
    ///
    /// # Panics
    ///
    /// Panics if the message type or payload exceeds `u32::MAX` bytes. Atlas
    /// protocol messages are intentionally bounded to 32-bit lengths.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let type_bytes = self.message_type.as_bytes();
        let type_len = u32::try_from(type_bytes.len()).expect("message type too large");
        let payload_len = u32::try_from(self.payload.len()).expect("payload too large");

        let mut out = Vec::with_capacity(HEADER_LEN + type_bytes.len() + self.payload.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&self.schema_major.to_be_bytes());
        out.extend_from_slice(&self.schema_minor.to_be_bytes());
        out.extend_from_slice(&type_len.to_be_bytes());
        out.extend_from_slice(&payload_len.to_be_bytes());
        out.extend_from_slice(type_bytes);
        out.extend_from_slice(&self.payload);
        out
    }
}

/// Protocol decode failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    /// The byte stream is too short for an Atlas envelope.
    Truncated,
    /// The magic prefix is not `ATLS`.
    InvalidMagic,
    /// The major schema version is unsupported.
    UnsupportedMajor {
        /// Major version found in the incoming envelope.
        found: u32,
        /// Major version supported by this crate.
        supported: u32,
    },
    /// Length fields do not match the actual byte stream length.
    InvalidLength,
    /// The message type is not valid UTF-8.
    InvalidMessageType,
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "truncated Atlas protocol envelope"),
            Self::InvalidMagic => write!(f, "invalid Atlas protocol magic"),
            Self::UnsupportedMajor { found, supported } => {
                write!(
                    f,
                    "unsupported schema major {found}; supported major is {supported}"
                )
            }
            Self::InvalidLength => write!(f, "invalid Atlas protocol envelope length"),
            Self::InvalidMessageType => write!(f, "invalid Atlas protocol message type"),
        }
    }
}

impl std::error::Error for ProtocolError {}

/// Decodes a deterministic Atlas protocol envelope.
///
/// Unknown major versions are rejected. Unknown minor versions are accepted so
/// long as the major version remains compatible.
///
/// # Errors
///
/// Returns a [`ProtocolError`] when the envelope is malformed or incompatible.
pub fn decode_envelope(bytes: &[u8]) -> Result<Envelope, ProtocolError> {
    if bytes.len() < HEADER_LEN {
        return Err(ProtocolError::Truncated);
    }
    if &bytes[0..4] != MAGIC {
        return Err(ProtocolError::InvalidMagic);
    }

    let major = read_u32(bytes, 4);
    if major != SCHEMA_MAJOR {
        return Err(ProtocolError::UnsupportedMajor {
            found: major,
            supported: SCHEMA_MAJOR,
        });
    }
    let minor = read_u32(bytes, 8);
    let type_len =
        usize::try_from(read_u32(bytes, 12)).map_err(|_| ProtocolError::InvalidLength)?;
    let payload_len =
        usize::try_from(read_u32(bytes, 16)).map_err(|_| ProtocolError::InvalidLength)?;
    let expected = HEADER_LEN
        .checked_add(type_len)
        .and_then(|n| n.checked_add(payload_len))
        .ok_or(ProtocolError::InvalidLength)?;
    if bytes.len() != expected {
        return Err(ProtocolError::InvalidLength);
    }

    let type_start = HEADER_LEN;
    let type_end = type_start + type_len;
    let message_type = std::str::from_utf8(&bytes[type_start..type_end])
        .map_err(|_| ProtocolError::InvalidMessageType)?
        .to_owned();
    let payload = bytes[type_end..].to_vec();

    Ok(Envelope {
        schema_major: major,
        schema_minor: minor,
        message_type,
        payload,
    })
}

fn read_u32(bytes: &[u8], start: usize) -> u32 {
    u32::from_be_bytes([
        bytes[start],
        bytes[start + 1],
        bytes[start + 2],
        bytes[start + 3],
    ])
}
