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
        if width == 0 || width > 32 {
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
            "checksum" => Self::new(
                8,
                vec![SearchOp::ChecksumEq {
                    modulus: 17,
                    target: 3,
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
        })
    }
}

impl SearchDomain {
    /// Creates a bounded domain.
    #[must_use]
    pub fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }
}
