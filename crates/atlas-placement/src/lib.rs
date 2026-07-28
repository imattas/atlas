//! Accelerator placement cost model.

/// Search workload features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchFeatures {
    /// Candidate count.
    pub candidates: u64,
    /// Whether the verifier is regular and branch-light.
    pub regular: bool,
    /// Whether compiled GPU kernel cache is expected to hit.
    pub kernel_cache_hit: bool,
}

/// Accelerator capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacementCapabilities {
    /// Scalar CPU available.
    pub scalar: bool,
    /// SIMD available.
    pub simd: bool,
    /// GPU available.
    pub gpu: bool,
}

/// Placement target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementTarget {
    /// Scalar CPU.
    Scalar,
    /// CPU SIMD.
    Simd,
    /// GPU.
    Gpu,
}

/// Placement decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlacementDecision {
    /// Selected target.
    pub target: PlacementTarget,
    /// Recorded rationale for reports.
    pub rationale: String,
}

/// Deterministic placement model.
pub struct PlacementModel;

impl PlacementModel {
    /// Chooses an available target with recorded rationale.
    #[must_use]
    pub fn choose(
        features: SearchFeatures,
        capabilities: PlacementCapabilities,
    ) -> PlacementDecision {
        if capabilities.gpu
            && features.regular
            && (features.candidates >= 1_000_000
                || (features.kernel_cache_hit && features.candidates >= 100_000))
        {
            return PlacementDecision {
                target: PlacementTarget::Gpu,
                rationale: "regular workload above GPU break-even".to_owned(),
            };
        }
        if capabilities.simd && features.regular && features.candidates >= 1_024 {
            return PlacementDecision {
                target: PlacementTarget::Simd,
                rationale: "medium regular workload fits SIMD batching".to_owned(),
            };
        }
        if capabilities.scalar {
            return PlacementDecision {
                target: PlacementTarget::Scalar,
                rationale: "tiny, divergent, or accelerator-unavailable workload".to_owned(),
            };
        }
        PlacementDecision {
            target: PlacementTarget::Scalar,
            rationale: "no advertised accelerator capability; scalar fallback required".to_owned(),
        }
    }
}
