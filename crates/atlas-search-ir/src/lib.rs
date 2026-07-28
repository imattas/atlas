//! Restricted bounded-search IR.

/// Supported search operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchOp {
    /// Candidate plus constant equals target modulo width.
    AddEq {
        /// Addend.
        addend: u64,
        /// Target value.
        target: u64,
    },
    /// Candidate XOR mask equals target.
    XorEq {
        /// XOR mask.
        mask: u64,
        /// Target value.
        target: u64,
    },
    /// Candidate checksum modulo `modulus` equals target.
    ChecksumEq {
        /// Modulus.
        modulus: u64,
        /// Target value.
        target: u64,
    },
    /// Candidate multiplied by `multiplier` plus `addend` equals target modulo width.
    MulAddEq {
        /// Multiplicative constant.
        multiplier: u64,
        /// Additive constant.
        addend: u64,
        /// Target value.
        target: u64,
    },
    /// Candidate rotated left within its declared width and XOR-ed with mask equals target.
    RotateXorEq {
        /// Left rotation amount.
        rotate_left: u32,
        /// XOR mask.
        mask: u64,
        /// Target value.
        target: u64,
    },
    /// Candidate byte at little-endian `byte_index` equals value.
    ByteEq {
        /// Zero-based little-endian byte index.
        byte_index: u32,
        /// Required byte value.
        value: u8,
    },
}

/// Restricted search program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchProgram {
    /// Candidate width in bits.
    pub width: u32,
    /// Operations all candidates must satisfy.
    pub ops: Vec<SearchOp>,
}

/// Bounded search domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchDomain {
    /// Inclusive start candidate.
    pub start: u64,
    /// Exclusive end candidate.
    pub end: u64,
}

/// Lowering validation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchIrError {
    /// Width is unsupported.
    UnsupportedWidth,
    /// Program contains forbidden memory aliasing.
    ForbiddenMemoryAliasing,
    /// Program contains data-dependent unbounded loops.
    UnboundedLoop,
    /// No supported operations exist.
    Empty,
}

impl SearchProgram {
    /// Creates a validated restricted search program.
    ///
    /// # Errors
    ///
    /// Returns an error when the program is outside the auditable subset.
    pub fn new(width: u32, ops: Vec<SearchOp>) -> Result<Self, SearchIrError> {
        if width == 0 || width > 64 {
            return Err(SearchIrError::UnsupportedWidth);
        }
        if ops.is_empty() {
            return Err(SearchIrError::Empty);
        }
        Ok(Self { width, ops })
    }

    /// Synthetic lowerer used by fixtures until direct UCIR search lowering is wired.
    ///
    /// # Errors
    ///
    /// Returns an error for fixture markers representing forbidden shapes.
    pub fn try_from_fixture(fixture: &str) -> Result<Self, SearchIrError> {
        match fixture {
            "add" => Self::new(
                8,
                vec![SearchOp::AddEq {
                    addend: 1,
                    target: 4,
                }],
            ),
            "xor" => Self::new(
                8,
                vec![SearchOp::XorEq {
                    mask: 0xaa,
                    target: 0xff,
                }],
            ),
            "xor64" => Self::new(
                64,
                vec![SearchOp::XorEq {
                    mask: 1,
                    target: 0x8000_0000_0000_0001,
                }],
            ),
            "checksum" => Self::new(
                8,
                vec![SearchOp::ChecksumEq {
                    modulus: 17,
                    target: 3,
                }],
            ),
            "dense" => Self::new(
                16,
                vec![SearchOp::ChecksumEq {
                    modulus: 1,
                    target: 0,
                }],
            ),
            "alias" => Err(SearchIrError::ForbiddenMemoryAliasing),
            "loop" => Err(SearchIrError::UnboundedLoop),
            _ => Err(SearchIrError::Empty),
        }
    }

    /// Evaluates one candidate against the restricted program.
    #[must_use]
    pub fn accepts(&self, candidate: u64) -> bool {
        let mask = if self.width == 64 {
            u64::MAX
        } else {
            (1_u64 << self.width) - 1
        };
        let candidate = candidate & mask;
        self.ops.iter().all(|op| match *op {
            SearchOp::AddEq { addend, target } => candidate.wrapping_add(addend) & mask == target,
            SearchOp::XorEq {
                mask: xor_mask,
                target,
            } => (candidate ^ xor_mask) & mask == target,
            SearchOp::ChecksumEq { modulus, target } => {
                modulus != 0 && candidate % modulus == target
            }
            SearchOp::MulAddEq {
                multiplier,
                addend,
                target,
            } => candidate.wrapping_mul(multiplier).wrapping_add(addend) & mask == target,
            SearchOp::RotateXorEq {
                rotate_left,
                mask: xor_mask,
                target,
            } => {
                rotate_left_width(candidate, rotate_left, self.width) ^ (xor_mask & mask) == target
            }
            SearchOp::ByteEq { byte_index, value } => {
                let shift = byte_index.saturating_mul(8);
                shift < self.width && ((candidate >> shift) & 0xff) == u64::from(value)
            }
        })
    }
}

fn rotate_left_width(value: u64, rotate_left: u32, width: u32) -> u64 {
    let mask = if width == 64 {
        u64::MAX
    } else {
        (1_u64 << width) - 1
    };
    let value = value & mask;
    let amount = rotate_left % width;
    if amount == 0 {
        value
    } else {
        ((value << amount) | (value >> (width - amount))) & mask
    }
}

impl SearchDomain {
    /// Creates a bounded domain.
    #[must_use]
    pub fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }
}
