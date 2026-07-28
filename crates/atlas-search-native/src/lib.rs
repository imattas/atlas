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
    /// Searches a bounded domain with cancellation polling and bounded output.
    #[must_use]
    pub fn search(
        program: &SearchProgram,
        domain: SearchDomain,
        cancellation: &CancellationToken,
    ) -> MatchStream {
        Self::search_with_stats(program, domain, cancellation).matches
    }

    /// Searches a bounded domain and returns execution statistics.
    #[must_use]
    pub fn search_with_stats(
        program: &SearchProgram,
        domain: SearchDomain,
        cancellation: &CancellationToken,
    ) -> SearchResult {
        if cancellation.is_cancelled() {
            return SearchResult {
                matches: Vec::new(),
                candidates_evaluated: 0,
                used_closed_form: false,
            };
        }
        if let Some(matches) = closed_form_matches(program, domain) {
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
            if cancellation.is_cancelled() || matches.len() >= 1024 {
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

fn closed_form_matches(program: &SearchProgram, domain: SearchDomain) -> Option<MatchStream> {
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
        SearchOp::XorEq {
            mask: xor_mask,
            target,
        } => (target ^ xor_mask) & mask,
        SearchOp::ChecksumEq { modulus, target } => {
            return checksum_matches(program, domain, mask, modulus, target);
        }
    };
    residue_matches(program, domain, mask.saturating_add(1), candidate)
}

fn checksum_matches(
    program: &SearchProgram,
    domain: SearchDomain,
    mask: u64,
    modulus: u64,
    target: u64,
) -> Option<MatchStream> {
    if modulus == 0 || target >= modulus {
        return Some(Vec::new());
    }
    let stride = mask.saturating_add(1);
    let residues = checksum_residues(mask, modulus, target)?;
    let mut base = (domain.start / stride).checked_mul(stride)?;
    let mut matches = Vec::new();
    while base < domain.end && matches.len() < 1024 {
        for residue in &residues {
            let Some(candidate) = base.checked_add(*residue) else {
                continue;
            };
            if candidate >= domain.start
                && candidate < domain.end
                && program.accepts(candidate)
                && matches.len() < 1024
            {
                matches.push(candidate);
            }
        }
        base = base.checked_add(stride)?;
    }
    Some(matches)
}

fn checksum_residues(mask: u64, modulus: u64, target: u64) -> Option<Vec<u64>> {
    let mut residues = Vec::new();
    let mut residue = target;
    while residue <= mask {
        residues.push(residue);
        residue = residue.checked_add(modulus)?;
    }
    Some(residues)
}

fn residue_matches(
    program: &SearchProgram,
    domain: SearchDomain,
    stride: u64,
    residue: u64,
) -> Option<MatchStream> {
    let mut matches = Vec::new();
    extend_residue_matches(program, domain, stride, residue, &mut matches)?;
    Some(matches)
}

fn extend_residue_matches(
    program: &SearchProgram,
    domain: SearchDomain,
    stride: u64,
    residue: u64,
    matches: &mut MatchStream,
) -> Option<()> {
    let mut current = first_candidate_at_or_above(residue, stride, domain.start)?;
    while current < domain.end && matches.len() < 1024 {
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
