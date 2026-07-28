//! Unified Constraint Intermediate Representation for `AtlasCTF`.

mod builder;
mod canonical;
mod eval;
mod expr;
mod provenance;
mod types;

pub use builder::{Builder, WithSource};
pub use canonical::{canonical_bytes, canonical_hash};
pub use eval::{EvalError, Evaluator, Model};
pub use expr::{ExprGraph, ExprId, ExprKind, Node, Simplifier};
pub use provenance::SourceLocation;
pub use types::{Endianness, Type, Value};
