//! Accelerator placement cost model.

use std::fs;
use std::path::Path;

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

/// Calibration thresholds for accelerator placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlacementCalibration {
    /// Minimum regular candidate count where SIMD is worthwhile.
    pub simd_min_candidates: u64,
    /// Minimum regular candidate count where GPU is worthwhile without a cache hit.
    pub gpu_min_candidates: u64,
    /// Minimum regular candidate count where GPU is worthwhile with a kernel cache hit.
    pub gpu_cache_hit_min_candidates: u64,
}

impl Default for PlacementCalibration {
    fn default() -> Self {
        Self {
            simd_min_candidates: 1_024,
            gpu_min_candidates: 1_000_000,
            gpu_cache_hit_min_candidates: 100_000,
        }
    }
}

impl PlacementCalibration {
    /// Loads placement thresholds from a Track 3 calibration manifest.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or a required threshold is
    /// missing or malformed.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|error| {
            format!(
                "cannot read placement calibration {}: {error}",
                path.display()
            )
        })?;
        Self::parse(&text)
    }

    /// Parses placement thresholds from manifest text.
    ///
    /// # Errors
    ///
    /// Returns an error when a required threshold is missing or malformed.
    pub fn parse(text: &str) -> Result<Self, String> {
        Ok(Self {
            simd_min_candidates: parse_threshold(text, "simd_min_candidates")?,
            gpu_min_candidates: parse_threshold(text, "gpu_min_candidates")?,
            gpu_cache_hit_min_candidates: parse_threshold(text, "gpu_cache_hit_min_candidates")?,
        })
    }
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
        Self::choose_with_calibration(features, capabilities, PlacementCalibration::default())
    }

    /// Chooses an available target using supplied calibration thresholds.
    #[must_use]
    pub fn choose_with_calibration(
        features: SearchFeatures,
        capabilities: PlacementCapabilities,
        calibration: PlacementCalibration,
    ) -> PlacementDecision {
        if capabilities.gpu
            && features.regular
            && (features.candidates >= calibration.gpu_min_candidates
                || (features.kernel_cache_hit
                    && features.candidates >= calibration.gpu_cache_hit_min_candidates))
        {
            return PlacementDecision {
                target: PlacementTarget::Gpu,
                rationale: "regular workload above GPU break-even".to_owned(),
            };
        }
        if capabilities.simd
            && features.regular
            && features.candidates >= calibration.simd_min_candidates
        {
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

fn parse_threshold(text: &str, key: &str) -> Result<u64, String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .find_map(|line| {
            let (name, value) = line.split_once('=')?;
            (name.trim() == key).then_some(value.trim())
        })
        .ok_or_else(|| format!("missing placement calibration threshold {key}"))?
        .parse()
        .map_err(|_| format!("invalid placement calibration threshold {key}"))
}
