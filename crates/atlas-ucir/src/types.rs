//! UCIR type and value definitions.

use std::collections::BTreeMap;

/// Byte order for memory operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Endianness {
    /// Most significant byte first.
    Big,
    /// Least significant byte first.
    Little,
}

/// UCIR static type.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Type {
    /// Boolean proposition.
    Bool,
    /// Arbitrary precision mathematical integer.
    Integer,
    /// Fixed-width bit-vector.
    BitVec {
        /// Width in bits.
        width: u32,
    },
    /// Integer modulo an explicit modulus.
    Modular {
        /// Positive modulus.
        modulus: u128,
    },
    /// Immutable byte string.
    Bytes {
        /// Number of bytes.
        len: usize,
    },
    /// Finite map with fixed-width bit-vector keys and values.
    Array {
        /// Index bit width.
        index_width: u32,
        /// Value bit width.
        value_width: u32,
    },
}

/// Concrete UCIR value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    /// Boolean value.
    Bool(bool),
    /// Mathematical integer.
    Int(i128),
    /// Fixed-width bit-vector value, normalized to the declared width.
    BitVec {
        /// Width in bits.
        width: u32,
        /// Normalized value.
        value: u128,
    },
    /// Modular integer value, normalized modulo `modulus`.
    Modular {
        /// Positive modulus.
        modulus: u128,
        /// Normalized residue.
        value: u128,
    },
    /// Byte string.
    Bytes(Vec<u8>),
    /// Fixed-width array value.
    Array {
        /// Index bit width.
        index_width: u32,
        /// Value bit width.
        value_width: u32,
        /// Default value for missing cells.
        default: u128,
        /// Explicit cells.
        cells: BTreeMap<u128, u128>,
    },
}

impl Value {
    /// Creates a normalized bit-vector value.
    ///
    /// # Errors
    ///
    /// Returns an error when `width` is zero or larger than 128.
    pub fn bitvec(width: u32, value: u128) -> Result<Self, String> {
        if width == 0 || width > 128 {
            return Err(format!("invalid bit-vector width {width}"));
        }
        Ok(Self::BitVec {
            width,
            value: value & mask(width),
        })
    }

    /// Creates a normalized modular value.
    ///
    /// # Errors
    ///
    /// Returns an error when `modulus` is zero.
    pub fn modular(modulus: u128, value: u128) -> Result<Self, String> {
        if modulus == 0 {
            return Err("modulus must be positive".to_owned());
        }
        Ok(Self::Modular {
            modulus,
            value: value % modulus,
        })
    }

    /// Returns this value's UCIR type.
    #[must_use]
    pub fn ty(&self) -> Type {
        match self {
            Self::Bool(_) => Type::Bool,
            Self::Int(_) => Type::Integer,
            Self::BitVec { width, .. } => Type::BitVec { width: *width },
            Self::Modular { modulus, .. } => Type::Modular { modulus: *modulus },
            Self::Bytes(bytes) => Type::Bytes { len: bytes.len() },
            Self::Array {
                index_width,
                value_width,
                ..
            } => Type::Array {
                index_width: *index_width,
                value_width: *value_width,
            },
        }
    }
}

/// Returns a bit mask for the given fixed width.
#[must_use]
pub fn mask(width: u32) -> u128 {
    if width == 128 {
        u128::MAX
    } else {
        (1_u128 << width) - 1
    }
}
