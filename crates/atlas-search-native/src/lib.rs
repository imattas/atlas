//! Verified scalar bounded search.

use atlas_scheduler::CancellationToken;
use atlas_search_ir::{SearchDomain, SearchOp, SearchProgram};

/// Candidate match stream.
pub type MatchStream = Vec<u64>;

/// Bounded search result with execution statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    /// Matching candidates.
    pub matches: MatchStream,
    /// Number of candidates evaluated through the program predicate.
    pub candidates_evaluated: u64,
    /// Whether an exact closed-form single-operation solver was used.
    pub used_closed_form: bool,
}

/// Scalar native searcher.
pub struct NativeSearcher;

impl NativeSearcher {
    const DEFAULT_MATCH_LIMIT: usize = 1024;

    /// Searches a bounded domain with cancellation polling and bounded output.
    #[must_use]
    pub fn search(
        program: &SearchProgram,
        domain: SearchDomain,
        cancellation: &CancellationToken,
    ) -> MatchStream {
        Self::search_with_stats(program, domain, cancellation).matches
    }

    /// Searches a bounded domain with cancellation polling and caller-selected
    /// bounded output.
    #[must_use]
    pub fn search_with_match_limit(
        program: &SearchProgram,
        domain: SearchDomain,
        cancellation: &CancellationToken,
        match_limit: usize,
    ) -> MatchStream {
        Self::search_with_stats_and_match_limit(program, domain, cancellation, match_limit).matches
    }

    /// Searches a bounded domain and returns execution statistics.
    #[must_use]
    pub fn search_with_stats(
        program: &SearchProgram,
        domain: SearchDomain,
        cancellation: &CancellationToken,
    ) -> SearchResult {
        Self::search_with_stats_and_match_limit(
            program,
            domain,
            cancellation,
            Self::DEFAULT_MATCH_LIMIT,
        )
    }

    fn search_with_stats_and_match_limit(
        program: &SearchProgram,
        domain: SearchDomain,
        cancellation: &CancellationToken,
        match_limit: usize,
    ) -> SearchResult {
        if cancellation.is_cancelled() {
            return SearchResult {
                matches: Vec::new(),
                candidates_evaluated: 0,
                used_closed_form: false,
            };
        }
        if match_limit == 0 {
            return SearchResult {
                matches: Vec::new(),
                candidates_evaluated: 0,
                used_closed_form: false,
            };
        }
        if let Some(matches) = closed_form_matches(program, domain, match_limit) {
            let candidates_evaluated = u64::try_from(matches.len()).unwrap_or(u64::MAX);
            return SearchResult {
                matches,
                candidates_evaluated,
                used_closed_form: true,
            };
        }

        let mut matches = Vec::new();
        let mut candidates_evaluated = 0_u64;
        for candidate in domain.start..domain.end {
            if cancellation.is_cancelled() || matches.len() >= match_limit {
                break;
            }
            candidates_evaluated = candidates_evaluated.saturating_add(1);
            if program.accepts(candidate) {
                matches.push(candidate);
            }
        }
        SearchResult {
            matches,
            candidates_evaluated,
            used_closed_form: false,
        }
    }
}

fn closed_form_matches(
    program: &SearchProgram,
    domain: SearchDomain,
    match_limit: usize,
) -> Option<MatchStream> {
    if program
        .ops
        .iter()
        .all(|op| matches!(op, SearchOp::ByteEq { .. }))
    {
        return byte_constraint_matches(program, domain, match_limit);
    }
    let [op] = program.ops.as_slice() else {
        return None;
    };
    let mask = if program.width == 64 {
        u64::MAX
    } else {
        (1_u64 << program.width) - 1
    };
    let candidate = match *op {
        SearchOp::AddEq { addend, target } => target.wrapping_sub(addend) & mask,
        SearchOp::MulAddEq {
            multiplier,
            addend,
            target,
        } => solve_mul_add(multiplier, addend, target, mask)?,
        SearchOp::XorEq {
            mask: xor_mask,
            target,
        } => (target ^ xor_mask) & mask,
        SearchOp::RotateXorEq {
            rotate_left,
            mask: xor_mask,
            target,
        } => rotate_right_width((target ^ xor_mask) & mask, rotate_left, program.width),
        SearchOp::ChecksumEq { modulus, target } => {
            return checksum_matches(program, domain, mask, modulus, target, match_limit);
        }
        SearchOp::ByteEq { .. } => return byte_constraint_matches(program, domain, match_limit),
    };
    residue_matches(
        program,
        domain,
        mask.saturating_add(1),
        candidate,
        match_limit,
    )
}

fn byte_constraint_matches(
    program: &SearchProgram,
    domain: SearchDomain,
    match_limit: usize,
) -> Option<MatchStream> {
    let mut fixed_mask = 0_u64;
    let mut fixed_value = 0_u64;
    for op in &program.ops {
        let SearchOp::ByteEq { byte_index, value } = *op else {
            return None;
        };
        let shift = byte_index.checked_mul(8)?;
        if shift >= program.width {
            return Some(Vec::new());
        }
        let byte_mask = 0xff_u64.checked_shl(shift)?;
        let byte_value = u64::from(value).checked_shl(shift)?;
        if fixed_mask & byte_mask != 0 && fixed_value & byte_mask != byte_value {
            return Some(Vec::new());
        }
        fixed_mask |= byte_mask;
        fixed_value = (fixed_value & !byte_mask) | byte_value;
    }
    let width_mask = if program.width == 64 {
        u64::MAX
    } else {
        (1_u64 << program.width) - 1
    };
    let free_positions: Vec<u32> = (0..program.width)
        .filter(|bit| fixed_mask & (1_u64 << bit) == 0)
        .collect();
    let mut matches = Vec::new();
    let combinations = 1_u64.checked_shl(u32::try_from(free_positions.len()).ok()?)?;
    for ordinal in 0..combinations {
        let mut candidate = fixed_value & width_mask;
        for (index, bit) in free_positions.iter().enumerate() {
            if ordinal & (1_u64 << u32::try_from(index).ok()?) != 0 {
                candidate |= 1_u64 << bit;
            }
        }
        if candidate >= domain.start && candidate < domain.end && program.accepts(candidate) {
            matches.push(candidate);
            if matches.len() >= match_limit {
                break;
            }
        }
    }
    matches.sort_unstable();
    Some(matches)
}

fn solve_mul_add(multiplier: u64, addend: u64, target: u64, mask: u64) -> Option<u64> {
    if multiplier & 1 == 0 {
        return None;
    }
    let modulus = u128::from(mask).checked_add(1)?;
    let rhs = u128::from(target.wrapping_sub(addend) & mask);
    let inverse = modular_inverse_power_two(u128::from(multiplier), modulus)?;
    u64::try_from((rhs * inverse) % modulus).ok()
}

fn modular_inverse_power_two(value: u128, modulus: u128) -> Option<u128> {
    if value & 1 == 0 || modulus == 0 {
        return None;
    }
    let mut inverse = 1_u128;
    let mut bits = 1_u32;
    while (1_u128 << bits) < modulus {
        let next_modulus = 1_u128 << (bits * 2).min(64);
        inverse =
            inverse.wrapping_mul(2_u128.wrapping_sub(value.wrapping_mul(inverse))) % next_modulus;
        bits *= 2;
    }
    Some(inverse % modulus)
}

fn rotate_right_width(value: u64, rotate_left: u32, width: u32) -> u64 {
    let mask = if width == 64 {
        u64::MAX
    } else {
        (1_u64 << width) - 1
    };
    let amount = rotate_left % width;
    if amount == 0 {
        value & mask
    } else {
        ((value >> amount) | (value << (width - amount))) & mask
    }
}

fn checksum_matches(
    program: &SearchProgram,
    domain: SearchDomain,
    _mask: u64,
    modulus: u64,
    target: u64,
    match_limit: usize,
) -> Option<MatchStream> {
    if modulus == 0 || target >= modulus {
        return Some(Vec::new());
    }
    let mut matches = Vec::new();
    let mut candidate = first_checksum_candidate_at_or_above(domain.start, modulus, target)?;
    while candidate < domain.end && matches.len() < match_limit {
        if program.accepts(candidate) {
            matches.push(candidate);
        }
        candidate = candidate.checked_add(modulus)?;
    }
    Some(matches)
}

fn first_checksum_candidate_at_or_above(start: u64, modulus: u64, target: u64) -> Option<u64> {
    let remainder = start % modulus;
    if remainder <= target {
        start.checked_add(target - remainder)
    } else {
        start.checked_add(modulus - (remainder - target))
    }
}

fn residue_matches(
    program: &SearchProgram,
    domain: SearchDomain,
    stride: u64,
    residue: u64,
    match_limit: usize,
) -> Option<MatchStream> {
    let mut matches = Vec::new();
    extend_residue_matches(program, domain, stride, residue, match_limit, &mut matches)?;
    Some(matches)
}

fn extend_residue_matches(
    program: &SearchProgram,
    domain: SearchDomain,
    stride: u64,
    residue: u64,
    match_limit: usize,
    matches: &mut MatchStream,
) -> Option<()> {
    let mut current = first_candidate_at_or_above(residue, stride, domain.start)?;
    while current < domain.end && matches.len() < match_limit {
        if program.accepts(current) {
            matches.push(current);
        }
        current = current.checked_add(stride)?;
    }
    Some(())
}

fn first_candidate_at_or_above(residue: u64, stride: u64, start: u64) -> Option<u64> {
    if residue >= start {
        return Some(residue);
    }
    let delta = start - residue;
    let steps = delta.div_ceil(stride);
    residue.checked_add(steps.checked_mul(stride)?)
}
