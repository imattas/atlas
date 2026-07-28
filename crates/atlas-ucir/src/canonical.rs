//! Deterministic UCIR serialization and hashing.

use std::fmt::Write as _;

use crate::ExprGraph;

/// Serializes a graph deterministically.
#[must_use]
pub fn canonical_bytes(graph: &ExprGraph) -> Vec<u8> {
    let mut out = String::new();
    out.push_str("atlas-ucir-v1\n");
    let _ = writeln!(out, "root:{}", graph.root().0);
    for (index, node) in graph.nodes().iter().enumerate() {
        let _ = writeln!(out, "{index}:{node:?}");
    }
    out.into_bytes()
}

/// Computes a deterministic 32-byte content hash for a graph.
#[must_use]
pub fn canonical_hash(graph: &ExprGraph) -> [u8; 32] {
    let bytes = canonical_bytes(graph);
    let mut state = [
        0x243f_6a88_85a3_08d3_u64,
        0x1319_8a2e_0370_7344_u64,
        0xa409_3822_299f_31d0_u64,
        0x082e_fa98_ec4e_6c89_u64,
    ];
    for (index, byte) in bytes.iter().enumerate() {
        let slot = index % state.len();
        state[slot] ^= u64::from(*byte);
        state[slot] = state[slot].wrapping_mul(0x1000_0000_01b3);
        state[slot] = state[slot].rotate_left(13);
    }
    let mut out = [0_u8; 32];
    for (chunk, value) in out.chunks_exact_mut(8).zip(state) {
        chunk.copy_from_slice(&value.to_be_bytes());
    }
    out
}
