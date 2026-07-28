//! Branch pruning.

/// Prunes a constant branch and returns the reachable label.
#[must_use]
pub fn prune_constant_branches(
    condition: Option<bool>,
    then_label: &str,
    else_label: &str,
) -> Vec<String> {
    match condition {
        Some(true) => vec![then_label.to_owned()],
        Some(false) => vec![else_label.to_owned()],
        None => vec![then_label.to_owned(), else_label.to_owned()],
    }
}
