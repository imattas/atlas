//! UCIR evaluator.

use std::collections::BTreeMap;

use crate::types::mask;
use crate::{Endianness, ExprGraph, ExprId, ExprKind, Value};

/// Symbolic model mapping variable names to concrete values.
pub type Model = BTreeMap<String, Value>;

/// Evaluation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalError {
    message: String,
}

impl EvalError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for EvalError {}

/// Concrete evaluator for UCIR graphs.
pub struct Evaluator;

impl Evaluator {
    /// Evaluates a graph root under a model.
    ///
    /// # Errors
    ///
    /// Returns an error when model values are missing, types are incompatible,
    /// or memory accesses are out of bounds.
    pub fn evaluate(graph: &ExprGraph, model: &Model) -> Result<Value, EvalError> {
        eval_id(graph, graph.root(), model)
    }
}

fn eval_id(graph: &ExprGraph, id: ExprId, model: &Model) -> Result<Value, EvalError> {
    let node = graph
        .node(id)
        .ok_or_else(|| EvalError::new(format!("unknown expression id {}", id.0)))?;
    match node.kind() {
        ExprKind::Const(_) => node
            .value()
            .cloned()
            .ok_or_else(|| EvalError::new("constant node missing value")),
        ExprKind::Var(name) => model
            .get(name)
            .cloned()
            .ok_or_else(|| EvalError::new(format!("missing model value for variable '{name}'"))),
        ExprKind::Add(left, right) => add(
            eval_id(graph, *left, model)?,
            eval_id(graph, *right, model)?,
        ),
        ExprKind::Eq(left, right) => Ok(Value::Bool(
            eval_id(graph, *left, model)? == eval_id(graph, *right, model)?,
        )),
        ExprKind::UnsignedLt(left, right) => unsigned_lt(
            eval_id(graph, *left, model)?,
            eval_id(graph, *right, model)?,
        ),
        ExprKind::SignedLt(left, right) => signed_lt(
            eval_id(graph, *left, model)?,
            eval_id(graph, *right, model)?,
        ),
        ExprKind::LoadBytes {
            memory,
            offset,
            width,
            endian,
        } => {
            let memory = eval_id(graph, *memory, model)?;
            let offset = eval_id(graph, *offset, model)?;
            load_bytes(memory, &offset, *width, *endian)
        }
        ExprKind::StoreArray {
            array,
            index,
            value,
        } => {
            let array = eval_id(graph, *array, model)?;
            let index = eval_id(graph, *index, model)?;
            let value = eval_id(graph, *value, model)?;
            store_array(array, &index, &value)
        }
        ExprKind::LoadArray { array, index } => {
            let array = eval_id(graph, *array, model)?;
            let index = eval_id(graph, *index, model)?;
            load_array(array, &index)
        }
    }
}

fn add(left: Value, right: Value) -> Result<Value, EvalError> {
    match (left, right) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
        (
            Value::BitVec { width, value: a },
            Value::BitVec {
                width: right_width,
                value: b,
            },
        ) if width == right_width => {
            Value::bitvec(width, a.wrapping_add(b)).map_err(EvalError::new)
        }
        (
            Value::Modular { modulus, value: a },
            Value::Modular {
                modulus: right_modulus,
                value: b,
            },
        ) if modulus == right_modulus => Value::modular(modulus, a + b).map_err(EvalError::new),
        (left, right) => Err(EvalError::new(format!(
            "cannot add values of type {:?} and {:?}",
            left.ty(),
            right.ty()
        ))),
    }
}

fn unsigned_lt(left: Value, right: Value) -> Result<Value, EvalError> {
    match (left, right) {
        (
            Value::BitVec { width, value: a },
            Value::BitVec {
                width: right_width,
                value: b,
            },
        ) if width == right_width => Ok(Value::Bool(a < b)),
        (left, right) => Err(EvalError::new(format!(
            "cannot compare unsigned values of type {:?} and {:?}",
            left.ty(),
            right.ty()
        ))),
    }
}

fn signed_lt(left: Value, right: Value) -> Result<Value, EvalError> {
    match (left, right) {
        (
            Value::BitVec { width, value: a },
            Value::BitVec {
                width: right_width,
                value: b,
            },
        ) if width == right_width => {
            let sign_bit = 1_u128 << (width - 1);
            let signed_a = if a & sign_bit == 0 {
                i128::try_from(a).map_err(|_| EvalError::new("signed conversion failed"))?
            } else {
                i128::try_from(a).map_err(|_| EvalError::new("signed conversion failed"))?
                    - i128::try_from(1_u128 << width)
                        .map_err(|_| EvalError::new("signed conversion failed"))?
            };
            let signed_b = if b & sign_bit == 0 {
                i128::try_from(b).map_err(|_| EvalError::new("signed conversion failed"))?
            } else {
                i128::try_from(b).map_err(|_| EvalError::new("signed conversion failed"))?
                    - i128::try_from(1_u128 << width)
                        .map_err(|_| EvalError::new("signed conversion failed"))?
            };
            Ok(Value::Bool(signed_a < signed_b))
        }
        (left, right) => Err(EvalError::new(format!(
            "cannot compare signed values of type {:?} and {:?}",
            left.ty(),
            right.ty()
        ))),
    }
}

fn load_bytes(
    memory: Value,
    offset: &Value,
    width: u32,
    endian: Endianness,
) -> Result<Value, EvalError> {
    let Value::Bytes(bytes) = memory else {
        return Err(EvalError::new("load memory is not bytes"));
    };
    let Value::Int(offset) = offset else {
        return Err(EvalError::new("load offset is not an integer"));
    };
    let offset = usize::try_from(*offset).map_err(|_| EvalError::new("negative load offset"))?;
    let len = usize::try_from(width / 8).map_err(|_| EvalError::new("invalid load width"))?;
    let end = offset
        .checked_add(len)
        .ok_or_else(|| EvalError::new("load offset overflow"))?;
    let window = bytes
        .get(offset..end)
        .ok_or_else(|| EvalError::new("byte load out of bounds"))?;
    let mut value = 0_u128;
    match endian {
        Endianness::Big => {
            for byte in window {
                value = (value << 8) | u128::from(*byte);
            }
        }
        Endianness::Little => {
            for byte in window.iter().rev() {
                value = (value << 8) | u128::from(*byte);
            }
        }
    }
    Value::bitvec(width, value).map_err(EvalError::new)
}

fn store_array(array: Value, index: &Value, value: &Value) -> Result<Value, EvalError> {
    let Value::Array {
        index_width,
        value_width,
        default,
        mut cells,
    } = array
    else {
        return Err(EvalError::new("store target is not an array"));
    };
    let Value::BitVec {
        width: actual_index_width,
        value: index,
    } = *index
    else {
        return Err(EvalError::new("store index is not a bit-vector"));
    };
    let Value::BitVec {
        width: actual_value_width,
        value,
    } = *value
    else {
        return Err(EvalError::new("store value is not a bit-vector"));
    };
    if index_width != actual_index_width || value_width != actual_value_width {
        return Err(EvalError::new("array store width mismatch"));
    }
    cells.insert(index & mask(index_width), value & mask(value_width));
    Ok(Value::Array {
        index_width,
        value_width,
        default,
        cells,
    })
}

fn load_array(array: Value, index: &Value) -> Result<Value, EvalError> {
    let Value::Array {
        index_width,
        value_width,
        default,
        cells,
    } = array
    else {
        return Err(EvalError::new("load target is not an array"));
    };
    let Value::BitVec {
        width: actual_index_width,
        value: index,
    } = *index
    else {
        return Err(EvalError::new("array index is not a bit-vector"));
    };
    if index_width != actual_index_width {
        return Err(EvalError::new("array load width mismatch"));
    }
    let value = cells.get(&index).copied().unwrap_or(default);
    Value::bitvec(value_width, value).map_err(EvalError::new)
}
