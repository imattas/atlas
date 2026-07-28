//! Placement model tests.

use atlas_placement::{PlacementCapabilities, PlacementModel, PlacementTarget, SearchFeatures};

#[test]
fn selects_scalar_for_tiny_or_divergent_jobs() {
    let capabilities = PlacementCapabilities {
        scalar: true,
        simd: true,
        gpu: true,
    };
    let decision = PlacementModel::choose(
        SearchFeatures {
            candidates: 16,
            regular: true,
            kernel_cache_hit: false,
        },
        capabilities,
    );

    assert_eq!(decision.target, PlacementTarget::Scalar);
    assert!(decision.rationale.contains("tiny"));
}

#[test]
fn selects_simd_for_medium_regular_jobs() {
    let decision = PlacementModel::choose(
        SearchFeatures {
            candidates: 10_000,
            regular: true,
            kernel_cache_hit: false,
        },
        PlacementCapabilities {
            scalar: true,
            simd: true,
            gpu: true,
        },
    );

    assert_eq!(decision.target, PlacementTarget::Simd);
}

#[test]
fn selects_gpu_above_break_even_or_cache_hit_threshold() {
    let capabilities = PlacementCapabilities {
        scalar: true,
        simd: true,
        gpu: true,
    };

    assert_eq!(
        PlacementModel::choose(
            SearchFeatures {
                candidates: 1_000_000,
                regular: true,
                kernel_cache_hit: false,
            },
            capabilities,
        )
        .target,
        PlacementTarget::Gpu
    );
    assert_eq!(
        PlacementModel::choose(
            SearchFeatures {
                candidates: 100_000,
                regular: true,
                kernel_cache_hit: true,
            },
            capabilities,
        )
        .target,
        PlacementTarget::Gpu
    );
}

#[test]
fn never_selects_unavailable_accelerators() {
    let decision = PlacementModel::choose(
        SearchFeatures {
            candidates: 10_000_000,
            regular: true,
            kernel_cache_hit: true,
        },
        PlacementCapabilities {
            scalar: true,
            simd: false,
            gpu: false,
        },
    );

    assert_eq!(decision.target, PlacementTarget::Scalar);
}
