//! Type-checking UCIR builder.

use std::collections::BTreeMap;

use crate::expr::{ExprId, ExprKind, Node};
use crate::types::mask;
use crate::{Endianness, ExprGraph, SourceLocation, Type, Value};

/// Incremental builder for immutable UCIR graphs.
#[derive(Debug, Clone, Default)]
pub struct Builder {
    nodes: Vec<Node>,
    memo: BTreeMap<String, ExprId>,
}

impl Builder {
    /// Creates an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an integer constant.
    pub fn int_const(&mut self, value: i128) -> ExprId {
        self.intern(Node::new(
            Type::Integer,
            ExprKind::Const(format!("int:{value}")),
            Some(Value::Int(value)),
        ))
    }

    /// Adds a byte-string constant.
    ///
    /// # Errors
    ///
    /// Returns an error for empty byte strings.
    pub fn bytes_const(&mut self, bytes: impl Into<Vec<u8>>) -> Result<ExprId, String> {
        let bytes = bytes.into();
        if bytes.is_empty() {
            return Err("byte string constants must not be empty".to_owned());
        }
        Ok(self.intern(Node::new(
            Type::Bytes { len: bytes.len() },
            ExprKind::Const(format!("bytes:{bytes:02x?}")),
            Some(Value::Bytes(bytes)),
        )))
    }

    /// Adds a bit-vector constant.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid widths.
    pub fn bitvec_const(&mut self, width: u32, value: u128) -> Result<ExprId, String> {
        let value = Value::bitvec(width, value)?;
        Ok(self.intern(Node::new(
            value.ty(),
            ExprKind::Const(format!("bv{width}:{}", value_bits(&value))),
            Some(value),
        )))
    }

    /// Adds a bit-vector variable.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid widths or empty names.
    pub fn bitvec_var(&mut self, name: impl Into<String>, width: u32) -> Result<ExprId, String> {
        let name = name.into();
        if name.is_empty() {
            return Err("variable name must not be empty".to_owned());
        }
        if width == 0 || width > 128 {
            return Err(format!("invalid bit-vector width {width}"));
        }
        Ok(self.intern(Node::new(Type::BitVec { width }, ExprKind::Var(name), None)))
    }

    /// Adds a modular constant.
    ///
    /// # Errors
    ///
    /// Returns an error when the modulus is invalid.
    pub fn modular_const(&mut self, modulus: u128, value: u128) -> Result<ExprId, String> {
        let value = Value::modular(modulus, value)?;
        Ok(self.intern(Node::new(
            value.ty(),
            ExprKind::Const(format!("mod{modulus}:{}", value_bits(&value))),
            Some(value),
        )))
    }

    /// Adds an array constant.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid widths.
    pub fn array_const(
        &mut self,
        index_width: u32,
        value_width: u32,
        default: u128,
    ) -> Result<ExprId, String> {
        if index_width == 0 || index_width > 128 || value_width == 0 || value_width > 128 {
            return Err("array widths must be in 1..=128".to_owned());
        }
        let default = default & mask(value_width);
        let value = Value::Array {
            index_width,
            value_width,
            default,
            cells: BTreeMap::new(),
        };
        Ok(self.intern(Node::new(
            value.ty(),
            ExprKind::Const(format!("array:{index_width}:{value_width}:{default}")),
            Some(value),
        )))
    }

    /// Adds wrapping addition.
    ///
    /// # Errors
    ///
    /// Returns an error if operand types differ or are unsupported.
    pub fn add(&mut self, left: ExprId, right: ExprId) -> Result<ExprId, String> {
        let ty = self.same_type(left, right)?;
        match ty {
            Type::BitVec { .. } | Type::Modular { .. } | Type::Integer => {
                Ok(self.intern(Node::new(ty, ExprKind::Add(left, right), None)))
            }
            _ => Err("add supports integers, bit-vectors, and modular values".to_owned()),
        }
    }

    /// Adds equality comparison.
    ///
    /// # Errors
    ///
    /// Returns an error if operand types differ.
    pub fn eq(&mut self, left: ExprId, right: ExprId) -> Result<ExprId, String> {
        self.same_type(left, right)?;
        Ok(self.intern(Node::new(Type::Bool, ExprKind::Eq(left, right), None)))
    }

    /// Adds unsigned less-than.
    ///
    /// # Errors
    ///
    /// Returns an error unless both operands are same-width bit-vectors.
    pub fn unsigned_lt(&mut self, left: ExprId, right: ExprId) -> Result<ExprId, String> {
        self.require_same_bitvec(left, right)?;
        Ok(self.intern(Node::new(
            Type::Bool,
            ExprKind::UnsignedLt(left, right),
            None,
        )))
    }

    /// Adds signed less-than.
    ///
    /// # Errors
    ///
    /// Returns an error unless both operands are same-width bit-vectors.
    pub fn signed_lt(&mut self, left: ExprId, right: ExprId) -> Result<ExprId, String> {
        self.require_same_bitvec(left, right)?;
        Ok(self.intern(Node::new(Type::Bool, ExprKind::SignedLt(left, right), None)))
    }

    /// Loads a fixed-width bit-vector from a byte string.
    ///
    /// # Errors
    ///
    /// Returns an error for non-byte memory, non-integer offset, or non-byte widths.
    pub fn load(
        &mut self,
        memory: ExprId,
        offset: ExprId,
        width: u32,
        endian: Endianness,
    ) -> Result<ExprId, String> {
        if !matches!(self.ty(memory), Some(Type::Bytes { .. })) {
            return Err("load memory must be bytes".to_owned());
        }
        if !matches!(self.ty(offset), Some(Type::Integer)) {
            return Err("load offset must be an integer".to_owned());
        }
        if width == 0 || width > 128 || !width.is_multiple_of(8) {
            return Err("load width must be a positive byte multiple up to 128".to_owned());
        }
        Ok(self.intern(Node::new(
            Type::BitVec { width },
            ExprKind::LoadBytes {
                memory,
                offset,
                width,
                endian,
            },
            None,
        )))
    }

    /// Stores a value into an array.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible array/index/value types.
    pub fn store(&mut self, array: ExprId, index: ExprId, value: ExprId) -> Result<ExprId, String> {
        let Some(Type::Array {
            index_width,
            value_width,
        }) = self.ty(array)
        else {
            return Err("store target must be an array".to_owned());
        };
        if self.ty(index) != Some(Type::BitVec { width: index_width }) {
            return Err("store index type does not match array".to_owned());
        }
        if self.ty(value) != Some(Type::BitVec { width: value_width }) {
            return Err("store value type does not match array".to_owned());
        }
        Ok(self.intern(Node::new(
            Type::Array {
                index_width,
                value_width,
            },
            ExprKind::StoreArray {
                array,
                index,
                value,
            },
            None,
        )))
    }

    /// Loads a value from an array.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible array/index types.
    pub fn load_array(&mut self, array: ExprId, index: ExprId) -> Result<ExprId, String> {
        let Some(Type::Array {
            index_width,
            value_width,
        }) = self.ty(array)
        else {
            return Err("load target must be an array".to_owned());
        };
        if self.ty(index) != Some(Type::BitVec { width: index_width }) {
            return Err("load index type does not match array".to_owned());
        }
        Ok(self.intern(Node::new(
            Type::BitVec { width: value_width },
            ExprKind::LoadArray { array, index },
            None,
        )))
    }

    /// Attaches source provenance to an existing node.
    ///
    /// # Errors
    ///
    /// Returns an error when `id` does not exist.
    pub fn set_source(&mut self, id: ExprId, source: SourceLocation) -> Result<ExprId, String> {
        let node = self
            .nodes
            .get_mut(id.0)
            .ok_or_else(|| format!("unknown expression id {}", id.0))?;
        node.set_source(source);
        Ok(id)
    }

    /// Finalizes this builder as an immutable graph.
    ///
    /// # Errors
    ///
    /// Returns an error when the root id does not exist.
    pub fn finish_with_root(self, root: ExprId) -> Result<ExprGraph, String> {
        if root.0 >= self.nodes.len() {
            return Err(format!("unknown root expression id {}", root.0));
        }
        Ok(ExprGraph::new(self.nodes, root))
    }

    fn intern(&mut self, node: Node) -> ExprId {
        let key = format!("{node:?}");
        if let Some(id) = self.memo.get(&key) {
            return *id;
        }
        let id = ExprId(self.nodes.len());
        self.nodes.push(node);
        self.memo.insert(key, id);
        id
    }

    fn ty(&self, id: ExprId) -> Option<Type> {
        self.nodes.get(id.0).map(|node| node.ty().clone())
    }

    fn same_type(&self, left: ExprId, right: ExprId) -> Result<Type, String> {
        let left_ty = self
            .ty(left)
            .ok_or_else(|| format!("unknown expression id {}", left.0))?;
        let right_ty = self
            .ty(right)
            .ok_or_else(|| format!("unknown expression id {}", right.0))?;
        if left_ty == right_ty {
            Ok(left_ty)
        } else {
            Err(format!("type mismatch: {left_ty:?} vs {right_ty:?}"))
        }
    }

    fn require_same_bitvec(&self, left: ExprId, right: ExprId) -> Result<u32, String> {
        match self.same_type(left, right)? {
            Type::BitVec { width } => Ok(width),
            other => Err(format!("expected bit-vector operands, found {other:?}")),
        }
    }
}

/// Convenience trait for test ergonomics when annotating source locations.
pub trait WithSource {
    /// Attaches source metadata to this expression id.
    ///
    /// # Errors
    ///
    /// Returns an error when the expression id does not exist in the builder.
    fn with_source(self, builder: &mut Builder, source: SourceLocation) -> Result<ExprId, String>;
}

impl WithSource for ExprId {
    fn with_source(self, builder: &mut Builder, source: SourceLocation) -> Result<ExprId, String> {
        builder.set_source(self, source)
    }
}

fn value_bits(value: &Value) -> u128 {
    match value {
        Value::BitVec { value, .. } | Value::Modular { value, .. } => *value,
        _ => 0,
    }
}
