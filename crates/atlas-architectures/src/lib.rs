//! Architecture semantics for advanced frontends.

use atlas_program::Architecture;

/// Endianness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    /// Little endian.
    Little,
    /// Big endian.
    Big,
}

/// Architecture semantics summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureSemantics {
    /// Architecture.
    pub architecture: Architecture,
    /// Pointer width.
    pub pointer_width: u32,
    /// Native byte order.
    pub byte_order: ByteOrder,
    /// General-purpose registers.
    pub registers: Vec<String>,
}

/// Instruction semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructionSemantics {
    /// Wrapping addition with explicit width.
    Add {
        /// Operation width.
        width: u32,
        /// Left operand.
        lhs: u64,
        /// Right operand.
        rhs: u64,
    },
    /// Branch based on boolean condition.
    Branch {
        /// Branch condition.
        condition: bool,
    },
    /// Load/store width and offset.
    LoadStore {
        /// Access width.
        width: u32,
        /// Access offset.
        offset: u64,
    },
}

/// Unsupported architecture/operation diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureDiagnostic {
    /// Located diagnostic.
    pub location: String,
    /// Message.
    pub message: String,
}

/// Returns semantics for supported Track 4 architectures.
///
/// # Errors
///
/// Returns a located diagnostic when the architecture is not part of the Track
/// 4 advanced frontend set.
pub fn semantics(
    architecture: Architecture,
) -> Result<ArchitectureSemantics, ArchitectureDiagnostic> {
    match architecture {
        Architecture::X86_32 => Ok(ArchitectureSemantics {
            architecture,
            pointer_width: 32,
            byte_order: ByteOrder::Little,
            registers: ["eax", "ebx", "ecx", "edx", "esp", "ebp"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        }),
        Architecture::Arm64 => Ok(ArchitectureSemantics {
            architecture,
            pointer_width: 64,
            byte_order: ByteOrder::Little,
            registers: ["x0", "x1", "x2", "x3", "sp", "lr"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        }),
        Architecture::WebAssembly => Ok(ArchitectureSemantics {
            architecture,
            pointer_width: 32,
            byte_order: ByteOrder::Little,
            registers: ["stack", "memory", "local0", "local1"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        }),
        other => Err(ArchitectureDiagnostic {
            location: "architecture".to_owned(),
            message: format!("unsupported advanced architecture: {other:?}"),
        }),
    }
}

/// Evaluates supported instruction semantics against reference behavior.
#[must_use]
pub fn evaluate_instruction(instruction: InstructionSemantics) -> u64 {
    match instruction {
        InstructionSemantics::Add { width, lhs, rhs } => {
            let mask = if width == 64 {
                u64::MAX
            } else {
                (1_u64 << width) - 1
            };
            lhs.wrapping_add(rhs) & mask
        }
        InstructionSemantics::Branch { condition } => u64::from(condition),
        InstructionSemantics::LoadStore { width, offset } => (u64::from(width) << 32) | offset,
    }
}
