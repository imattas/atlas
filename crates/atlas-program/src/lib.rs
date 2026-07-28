//! Safe program and artifact intake for Track 2 frontends.

/// Supported architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// x86-64 machine code.
    X86_64,
    /// x86-32 machine code.
    X86_32,
    /// ARM64 machine code.
    Arm64,
    /// WebAssembly module.
    WebAssembly,
    /// Restricted C source.
    RestrictedC,
    /// Restricted Python checker.
    RestrictedPython,
    /// Concrete trace artifact.
    Trace,
}

/// Detected artifact kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    /// ELF executable or object.
    Elf,
    /// C source.
    CSource,
    /// Python source.
    PythonSource,
    /// JSON-ish concrete trace.
    Trace,
}

/// Intake artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact {
    /// Artifact name.
    pub name: String,
    /// Artifact bytes.
    pub bytes: Vec<u8>,
}

impl Artifact {
    /// Creates an artifact.
    #[must_use]
    pub fn new(name: impl Into<String>, bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            name: name.into(),
            bytes: bytes.into(),
        }
    }
}

/// Lowered program metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    /// Detected architecture or frontend language.
    pub architecture: Architecture,
    /// Entry point label.
    pub entry: String,
    /// Source locations retained by intake.
    pub sources: Vec<String>,
}

/// Lowering result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweringResult {
    /// Program metadata.
    pub program: Program,
    /// Located diagnostics.
    pub diagnostics: Vec<String>,
}

/// Frontend error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrontendError {
    /// Artifact exceeds parser limits.
    TooLarge,
    /// Artifact type is not supported.
    Unsupported,
    /// Artifact is malformed.
    Malformed(String),
}

/// Frontend boundary.
pub trait Frontend {
    /// Lowers an artifact without executing intake-time code.
    ///
    /// # Errors
    ///
    /// Returns a frontend error when the artifact is unsupported or malformed.
    fn lower(artifact: &Artifact) -> Result<LoweringResult, FrontendError>;
}

/// Safe intake frontend.
pub struct IntakeFrontend;

impl Frontend for IntakeFrontend {
    fn lower(artifact: &Artifact) -> Result<LoweringResult, FrontendError> {
        lower(artifact)
    }
}

/// Detects artifact kind from name and bytes.
///
/// # Errors
///
/// Returns an error when no supported kind matches.
pub fn detect(artifact: &Artifact) -> Result<ArtifactKind, FrontendError> {
    if artifact.bytes.len() > 1024 * 1024 {
        return Err(FrontendError::TooLarge);
    }
    if artifact.bytes.starts_with(b"\x7fELF") {
        return Ok(ArtifactKind::Elf);
    }
    let path = std::path::Path::new(&artifact.name);
    if path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("c"))
    {
        return Ok(ArtifactKind::CSource);
    }
    if path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("py"))
    {
        return Ok(ArtifactKind::PythonSource);
    }
    if artifact.name.ends_with(".trace.json") {
        return Ok(ArtifactKind::Trace);
    }
    Err(FrontendError::Unsupported)
}

/// Lowers an artifact into program metadata.
///
/// # Errors
///
/// Returns an error for unsupported, malformed, or oversized artifacts.
pub fn lower(artifact: &Artifact) -> Result<LoweringResult, FrontendError> {
    let kind = detect(artifact)?;
    let text = String::from_utf8_lossy(&artifact.bytes);
    let (architecture, entry) = match kind {
        ArtifactKind::Elf => {
            if artifact.bytes.len() < 5 {
                return Err(FrontendError::Malformed("truncated ELF".to_owned()));
            }
            (Architecture::X86_64, "elf-entry".to_owned())
        }
        ArtifactKind::CSource => {
            reject_dangerous_source(&text)?;
            (Architecture::RestrictedC, "main".to_owned())
        }
        ArtifactKind::PythonSource => {
            reject_dangerous_source(&text)?;
            (Architecture::RestrictedPython, "module".to_owned())
        }
        ArtifactKind::Trace => {
            if !text.contains("\"events\"") {
                return Err(FrontendError::Malformed("trace missing events".to_owned()));
            }
            (Architecture::Trace, "trace".to_owned())
        }
    };
    Ok(LoweringResult {
        program: Program {
            architecture,
            entry,
            sources: vec![artifact.name.clone()],
        },
        diagnostics: Vec::new(),
    })
}

fn reject_dangerous_source(text: &str) -> Result<(), FrontendError> {
    for token in ["socket", "subprocess", "system(", "#include <sys/socket.h>"] {
        if text.contains(token) {
            return Err(FrontendError::Malformed(format!(
                "restricted source contains disallowed token {token}"
            )));
        }
    }
    Ok(())
}
