//! Deterministic analysis pass manager.

use atlas_ucir::{Evaluator, ExprGraph, ExprId, ExprKind, Model, Type, Value};

use crate::{bitvec, partition};

/// Analysis failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisError {
    message: String,
}

impl AnalysisError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for AnalysisError {}

/// Transparent graph features used by planners.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Features {
    /// Number of symbolic variables.
    pub variable_count: usize,
    /// Number of boolean constraint expressions.
    pub constraint_count: usize,
    /// Number of bit-vector XOR operations.
    pub xor_ops: usize,
    /// Number of modular-typed nodes.
    pub modular_nodes: usize,
    /// Whether the graph contains a GF(2)-affine shape.
    pub has_gf2_affine: bool,
    /// Whether a top-level contradiction was proven.
    pub has_contradiction: bool,
}

/// Independent component summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Component {
    /// Root expression for this component.
    pub root: ExprId,
    /// Variable names referenced by this component.
    pub variables: Vec<String>,
}

/// Recorded analysis derivation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Derivation {
    /// Pass or rewrite name.
    pub pass: String,
    /// Human-readable explanation.
    pub detail: String,
    /// Whether this derivation is sound and can affect mandatory reasoning.
    pub sound: bool,
}

/// Result of analyzing a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisResult {
    /// Graph after currently implemented sound identity-preserving passes.
    pub graph: ExprGraph,
    /// Independent components.
    pub components: Vec<Component>,
    /// Extracted planner features.
    pub features: Features,
    /// Provenance derivations produced by analysis passes.
    pub derivations: Vec<Derivation>,
}

/// Analyzes a UCIR graph.
///
/// # Errors
///
/// Returns an error when the graph root does not exist.
pub fn analyze(graph: &ExprGraph) -> Result<AnalysisResult, AnalysisError> {
    graph
        .node(graph.root())
        .ok_or_else(|| AnalysisError::new("graph root is missing"))?;

    let mut features = Features::default();
    let mut derivations = Vec::new();

    for node in graph.nodes() {
        match node.kind() {
            ExprKind::Var(_) => features.variable_count += 1,
            ExprKind::Eq(_, _) | ExprKind::And(_) => features.constraint_count += 1,
            ExprKind::Xor(_, _) => {
                features.xor_ops += 1;
                derivations.push(Derivation {
                    pass: "bitvec.gf2-affine-recognition".to_owned(),
                    detail: "XOR over fixed-width bit-vectors is affine over GF(2)".to_owned(),
                    sound: true,
                });
            }
            _ => {}
        }
        if matches!(node.ty(), Type::Modular { .. }) {
            features.modular_nodes += 1;
            derivations.push(Derivation {
                pass: "algebra.domain-detection".to_owned(),
                detail: "modular arithmetic domain recorded for algebra routing".to_owned(),
                sound: true,
            });
        }
    }

    features.has_gf2_affine = bitvec::has_gf2_affine(graph);
    if matches!(
        Evaluator::evaluate(graph, &Model::new()),
        Ok(Value::Bool(false))
    ) {
        features.has_contradiction = true;
        derivations.push(Derivation {
            pass: "general.contradiction-detection".to_owned(),
            detail: "root constraint evaluates to false without symbolic assumptions".to_owned(),
            sound: true,
        });
    }

    derivations.push(Derivation {
        pass: "general.identity-normalization".to_owned(),
        detail: "no unsound approximation introduced".to_owned(),
        sound: true,
    });

    Ok(AnalysisResult {
        graph: graph.clone(),
        components: partition::components(graph),
        features,
        derivations,
    })
}

pub(crate) fn collect_variables(graph: &ExprGraph, root: ExprId, out: &mut Vec<String>) {
    let Some(node) = graph.node(root) else {
        return;
    };
    match node.kind() {
        ExprKind::Var(name) => {
            if !out.iter().any(|existing| existing == name) {
                out.push(name.clone());
            }
        }
        ExprKind::Add(a, b)
        | ExprKind::Xor(a, b)
        | ExprKind::Eq(a, b)
        | ExprKind::UnsignedLt(a, b)
        | ExprKind::SignedLt(a, b)
        | ExprKind::LoadArray { array: a, index: b } => {
            collect_variables(graph, *a, out);
            collect_variables(graph, *b, out);
        }
        ExprKind::And(inputs) => {
            for input in inputs {
                collect_variables(graph, *input, out);
            }
        }
        ExprKind::LoadBytes { memory, offset, .. } => {
            collect_variables(graph, *memory, out);
            collect_variables(graph, *offset, out);
        }
        ExprKind::StoreArray {
            array,
            index,
            value,
        } => {
            collect_variables(graph, *array, out);
            collect_variables(graph, *index, out);
            collect_variables(graph, *value, out);
        }
        ExprKind::Const(_) => {}
    }
    out.sort();
}
