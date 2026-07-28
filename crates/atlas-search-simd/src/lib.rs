//! Hardware-independent SIMD-style batched bounded search.

use atlas_scheduler::CancellationToken;
use atlas_search_ir::{SearchDomain, SearchOp, SearchProgram, Searcher};
use wide::u64x4;

const OUTPUT_LIMIT: usize = 1024;
const WIDE_LANES: usize = 4;

/// SIMD execution engine used for a search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdEngine {
    /// Safe portable wide-vector execution with four `u64` lanes.
    WideU64x4,
    /// Scalar tail or unsupported-lane fallback.
    Scalar,
}

/// SIMD search report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimdSearchReport {
    /// Engine used for the main search body.
    pub engine: SimdEngine,
    /// CPU-semantic matches found by the SIMD searcher.
    pub matches: Vec<u64>,
}

/// SIMD searcher using deterministic fixed-size batches.
pub struct SimdSearcher;

impl SimdSearcher {
    /// Searches a bounded domain in batches while preserving scalar semantics.
    #[must_use]
    pub fn search(
        program: &SearchProgram,
        domain: SearchDomain,
        cancellation: &CancellationToken,
        lanes: usize,
    ) -> Vec<u64> {
        Self::search_report(program, domain, cancellation, lanes).matches
    }

    /// Searches a bounded domain and reports the concrete SIMD engine used.
    #[must_use]
    pub fn search_report(
        program: &SearchProgram,
        domain: SearchDomain,
        cancellation: &CancellationToken,
        lanes: usize,
    ) -> SimdSearchReport {
        let lanes = lanes.max(1);
        if lanes < WIDE_LANES {
            return SimdSearchReport {
                engine: SimdEngine::Scalar,
                matches: scalar_search(program, domain, cancellation, lanes),
            };
        }
        let mut matches = Vec::new();
        let mut cursor = domain.start;
        while cursor < domain.end && matches.len() < OUTPUT_LIMIT {
            if cancellation.is_cancelled() {
                break;
            }
            let vector_end = cursor
                .saturating_add(u64::try_from(WIDE_LANES).unwrap_or(u64::MAX))
                .min(domain.end);
            if vector_end.saturating_sub(cursor) == WIDE_LANES as u64 {
                let candidates = u64x4::new([cursor, cursor + 1, cursor + 2, cursor + 3]);
                let accepted_mask = accepts_wide(program, candidates);
                for lane in 0..WIDE_LANES {
                    if accepted_mask & (1 << lane) != 0 {
                        matches.push(cursor + u64::try_from(lane).unwrap_or(0));
                        if matches.len() >= OUTPUT_LIMIT {
                            break;
                        }
                    }
                }
                cursor = vector_end;
                continue;
            }
            for candidate in cursor..vector_end {
                if program.accepts(candidate) {
                    matches.push(candidate);
                    if matches.len() >= OUTPUT_LIMIT {
                        break;
                    }
                }
            }
            cursor = vector_end;
        }
        SimdSearchReport {
            engine: SimdEngine::WideU64x4,
            matches,
        }
    }
}

impl Searcher for SimdSearcher {
    fn search(
        program: &SearchProgram,
        domain: SearchDomain,
        cancellation: &CancellationToken,
    ) -> Vec<u64> {
        SimdSearcher::search(program, domain, cancellation, 8)
    }
}

fn scalar_search(
    program: &SearchProgram,
    domain: SearchDomain,
    cancellation: &CancellationToken,
    lanes: usize,
) -> Vec<u64> {
    let mut matches = Vec::new();
    let mut cursor = domain.start;
    while cursor < domain.end && matches.len() < OUTPUT_LIMIT {
        if cancellation.is_cancelled() {
            break;
        }
        let batch_end = cursor
            .saturating_add(u64::try_from(lanes).unwrap_or(u64::MAX))
            .min(domain.end);
        for candidate in cursor..batch_end {
            if program.accepts(candidate) {
                matches.push(candidate);
                if matches.len() >= OUTPUT_LIMIT {
                    break;
                }
            }
        }
        cursor = batch_end;
    }
    matches
}

fn accepts_wide(program: &SearchProgram, raw_candidates: u64x4) -> u8 {
    let mask = u64x4::splat(width_mask(program.width));
    let candidates = raw_candidates & mask;
    let mut accepted = [true; WIDE_LANES];
    for op in &program.ops {
        let lane_values = accepts_op_wide(op, candidates, mask, program.width).to_array();
        for (lane, value) in lane_values.into_iter().enumerate() {
            accepted[lane] &= value == u64::MAX;
        }
    }
    accepted
        .into_iter()
        .enumerate()
        .fold(0_u8, |bits, (lane, is_accepted)| {
            bits | (u8::from(is_accepted) << lane)
        })
}

fn accepts_op_wide(op: &SearchOp, candidates: u64x4, mask: u64x4, width: u32) -> u64x4 {
    match *op {
        SearchOp::AddEq { addend, target } => {
            ((candidates + u64x4::splat(addend)) & mask).simd_eq(u64x4::splat(target))
        }
        SearchOp::XorEq {
            mask: xor_mask,
            target,
        } => ((candidates ^ u64x4::splat(xor_mask)) & mask).simd_eq(u64x4::splat(target)),
        SearchOp::ChecksumEq { modulus, target } => {
            if modulus == 0 {
                u64x4::ZERO
            } else {
                (candidates % u64x4::splat(modulus)).simd_eq(u64x4::splat(target))
            }
        }
        SearchOp::MulAddEq {
            multiplier,
            addend,
            target,
        } => ((candidates * u64x4::splat(multiplier) + u64x4::splat(addend)) & mask)
            .simd_eq(u64x4::splat(target)),
        SearchOp::RotateXorEq {
            rotate_left,
            mask: xor_mask,
            target,
        } => ((rotate_left_width_wide(candidates, rotate_left, width) ^ u64x4::splat(xor_mask))
            & mask)
            .simd_eq(u64x4::splat(target)),
        SearchOp::ByteEq { byte_index, value } => {
            let shift = byte_index.saturating_mul(8);
            if shift >= width {
                u64x4::ZERO
            } else {
                ((candidates >> shift) & u64x4::splat(0xff)).simd_eq(u64x4::splat(u64::from(value)))
            }
        }
    }
}

fn rotate_left_width_wide(values: u64x4, rotate_left: u32, width: u32) -> u64x4 {
    let mask = u64x4::splat(width_mask(width));
    let values = values & mask;
    let amount = rotate_left % width;
    if amount == 0 {
        values
    } else {
        ((values << amount) | (values >> (width - amount))) & mask
    }
}

fn width_mask(width: u32) -> u64 {
    if width == 64 {
        u64::MAX
    } else {
        (1_u64 << width) - 1
    }
}
