//! Verified scalar bounded search.

use atlas_scheduler::CancellationToken;
use atlas_search_ir::{SearchDomain, SearchProgram};

/// Candidate match stream.
pub type MatchStream = Vec<u64>;

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
        let mut matches = Vec::new();
        for candidate in domain.start..domain.end {
            if cancellation.is_cancelled() || matches.len() >= 1024 {
                break;
            }
            if program.accepts(candidate) {
                matches.push(candidate);
            }
        }
        matches
    }
}
