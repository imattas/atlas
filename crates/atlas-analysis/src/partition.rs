//! Independence partitioning.

use atlas_ucir::{ExprGraph, ExprKind};

use crate::pipeline::{collect_variables, Component};

/// Partitions top-level conjunctions into independent components.
#[must_use]
pub fn components(graph: &ExprGraph) -> Vec<Component> {
    let Some(root) = graph.node(graph.root()) else {
        return Vec::new();
    };
    if let ExprKind::And(inputs) = root.kind() {
        inputs
            .iter()
            .map(|input| {
                let mut variables = Vec::new();
                collect_variables(graph, *input, &mut variables);
                Component {
                    root: *input,
                    variables,
                }
            })
            .collect()
    } else {
        let mut variables = Vec::new();
        collect_variables(graph, graph.root(), &mut variables);
        vec![Component {
            root: graph.root(),
            variables,
        }]
    }
}
