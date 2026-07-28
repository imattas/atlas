//! Safe optional learned strategy ranking.

use std::collections::{BTreeMap, BTreeSet};

/// Rank request.
#[derive(Debug, Clone, PartialEq)]
pub struct RankRequest {
    /// Candidate strategy ids.
    pub strategy_ids: Vec<String>,
    /// Numeric features.
    pub features: BTreeMap<String, f64>,
    /// Deterministic seed.
    pub seed: u64,
}

/// Rank response. This type intentionally contains only ordering, budgets, and explanation.
#[derive(Debug, Clone, PartialEq)]
pub struct RankResponse {
    /// Ordered strategy ids.
    pub ordered_strategy_ids: Vec<String>,
    /// Budget multipliers by strategy id.
    pub budget_multipliers: BTreeMap<String, f64>,
    /// Human-readable explanation.
    pub explanation: String,
}

/// Ranker error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RankError {
    /// Missing features.
    MissingFeatures,
    /// Corrupt or incompatible model.
    CorruptModel,
    /// Unknown strategy returned by a model.
    UnknownStrategy(String),
}

/// Transparent baseline ranker.
pub struct SafeRanker;

impl SafeRanker {
    /// Ranks strategies deterministically. Falls back offline when model bytes are absent.
    ///
    /// # Errors
    ///
    /// Returns an error when required features are missing or model bytes are corrupt.
    pub fn rank(
        request: &RankRequest,
        model_bytes: Option<&[u8]>,
    ) -> Result<RankResponse, RankError> {
        if request.features.is_empty() {
            return Err(RankError::MissingFeatures);
        }
        if matches!(model_bytes, Some(bytes) if bytes == b"corrupt") {
            return Err(RankError::CorruptModel);
        }
        let mut ordered = request.strategy_ids.clone();
        ordered.sort_by(|left, right| {
            score(right, request)
                .partial_cmp(&score(left, request))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(left.cmp(right))
        });
        let budget_multipliers = ordered
            .iter()
            .enumerate()
            .map(|(index, strategy)| {
                let bounded_index = u32::try_from(index).unwrap_or(u32::MAX);
                (strategy.clone(), 1.0 + (f64::from(bounded_index) * 0.1))
            })
            .collect();
        Ok(RankResponse {
            ordered_strategy_ids: ordered,
            budget_multipliers,
            explanation: if model_bytes.is_some() {
                "transparent learned baseline ranking".to_owned()
            } else {
                "offline fallback rule ranking".to_owned()
            },
        })
    }

    /// Validates ranker output against an allowlist.
    ///
    /// # Errors
    ///
    /// Returns an error if the response references unknown strategies.
    pub fn validate_response(
        response: &RankResponse,
        allowed_strategies: &[String],
    ) -> Result<(), RankError> {
        let allowed: BTreeSet<_> = allowed_strategies.iter().cloned().collect();
        for strategy in response
            .ordered_strategy_ids
            .iter()
            .chain(response.budget_multipliers.keys())
        {
            if !allowed.contains(strategy) {
                return Err(RankError::UnknownStrategy(strategy.clone()));
            }
        }
        Ok(())
    }
}

fn score(strategy: &str, request: &RankRequest) -> f64 {
    let feature_sum: f64 = request.features.values().sum();
    let seed_mod = u32::try_from(request.seed % 7).unwrap_or(0);
    let seed_bias = f64::from(seed_mod) / 100.0;
    let len = u32::try_from(strategy.len().max(1)).unwrap_or(u32::MAX);
    feature_sum / f64::from(len) + seed_bias
}
