//! Hardware-independent SIMD-style batched bounded search.

use atlas_scheduler::CancellationToken;
use atlas_search_ir::{SearchDomain, SearchProgram};

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
        let lanes = lanes.max(1);
        let mut matches = Vec::new();
        let mut cursor = domain.start;
        while cursor < domain.end {
            if cancellation.is_cancelled() {
                break;
            }
            let batch_end = cursor
                .saturating_add(u64::try_from(lanes).unwrap_or(u64::MAX))
                .min(domain.end);
            for candidate in cursor..batch_end {
                if program.accepts(candidate) {
                    matches.push(candidate);
                }
            }
            cursor = batch_end;
        }
        matches
    }
}
