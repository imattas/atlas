//! Immutable UCIR expression graph.

use crate::{Endianness, SourceLocation, Type, Value};

/// Stable expression identifier within a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExprId(pub(crate) usize);

/// UCIR expression operation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExprKind {
    /// Constant literal.
    Const(String),
    /// Named symbolic variable.
    Var(String),
    /// Addition.
    Add(ExprId, ExprId),
    /// Equality.
    Eq(ExprId, ExprId),
    /// Unsigned less-than.
    UnsignedLt(ExprId, ExprId),
    /// Signed less-than.
    SignedLt(ExprId, ExprId),
    /// Load bit-vector from byte string.
    LoadBytes {
        /// Byte-string expression.
        memory: ExprId,
        /// Integer byte offset expression.
        offset: ExprId,
        /// Loaded width in bits.
        width: u32,
        /// Load byte order.
        endian: Endianness,
    },
    /// Store a bit-vector into an array.
    StoreArray {
        /// Array expression.
        array: ExprId,
        /// Index expression.
        index: ExprId,
        /// Value expression.
        value: ExprId,
    },
    /// Load a bit-vector from an array.
    LoadArray {
        /// Array expression.
        array: ExprId,
        /// Index expression.
        index: ExprId,
    },
}

/// Immutable expression node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    ty: Type,
    kind: ExprKind,
    value: Option<Value>,
    source: Option<SourceLocation>,
}

impl Node {
    /// Creates a node.
    #[must_use]
    pub fn new(ty: Type, kind: ExprKind, value: Option<Value>) -> Self {
        Self {
            ty,
            kind,
            value,
            source: None,
        }
    }

    /// Node type.
    #[must_use]
    pub fn ty(&self) -> &Type {
        &self.ty
    }

    /// Node operation.
    #[must_use]
    pub fn kind(&self) -> &ExprKind {
        &self.kind
    }

    /// Constant value for literal nodes.
    #[must_use]
    pub fn value(&self) -> Option<&Value> {
        self.value.as_ref()
    }

    /// Optional source location.
    #[must_use]
    pub fn source(&self) -> Option<&SourceLocation> {
        self.source.as_ref()
    }

    pub(crate) fn set_source(&mut self, source: SourceLocation) {
        self.source = Some(source);
    }
}

/// Immutable expression graph with one root expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExprGraph {
    nodes: Vec<Node>,
    root: ExprId,
}

impl ExprGraph {
    /// Creates an expression graph from already validated nodes.
    #[must_use]
    pub fn new(nodes: Vec<Node>, root: ExprId) -> Self {
        Self { nodes, root }
    }

    /// Root expression id.
    #[must_use]
    pub fn root(&self) -> ExprId {
        self.root
    }

    /// Returns all nodes in topological insertion order.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Returns a node by id.
    #[must_use]
    pub fn node(&self, id: ExprId) -> Option<&Node> {
        self.nodes.get(id.0)
    }
}

/// Initial simplifier placeholder for identity-preserving transformations.
pub struct Simplifier;

impl Simplifier {
    /// Returns an unchanged graph.
    #[must_use]
    pub fn identity(graph: &ExprGraph) -> ExprGraph {
        graph.clone()
    }
}
