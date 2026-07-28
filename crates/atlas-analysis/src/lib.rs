//! Sound analysis and normalization pipeline for `AtlasCTF` UCIR graphs.

mod algebra;
mod bitvec;
mod general;
mod partition;
mod pipeline;

pub use pipeline::{analyze, AnalysisError, AnalysisResult, Component, Derivation, Features};
