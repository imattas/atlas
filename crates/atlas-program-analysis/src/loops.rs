//! Loop-bound inference.

/// Infers a loop bound from an explicit constant trip count.
#[must_use]
pub fn infer_loop_bound(initial: i64, limit: i64, step: i64) -> Option<usize> {
    if step <= 0 || initial > limit {
        return None;
    }
    let distance = limit.checked_sub(initial)?;
    usize::try_from((distance / step) + 1).ok()
}
