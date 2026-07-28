//! Bit-vector analysis passes.

use atlas_ucir::{ExprGraph, ExprKind};

/// Returns whether the graph contains a GF(2)-affine XOR expression.
#[must_use]
pub fn has_gf2_affine(graph: &ExprGraph) -> bool {
    graph
        .nodes()
        .iter()
        .any(|node| matches!(node.kind(), ExprKind::Xor(_, _)))
}
