# AtlasCTF Track 1 Core Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a secure constraint-orchestration release that solves, validates, explains, and benchmarks equation and SMT challenges through one API.

**Architecture:** A Cargo workspace separates immutable UCIR, protocol, facts, planning, backend processes, validation, and reporting. Python and CLI clients call the same orchestrator service and never depend directly on solver-native types.

**Tech Stack:** Rust stable, Python 3.11-3.13, PyO3/maturin, serde, Protobuf/tonic, Z3, SageMath, SQLite, proptest, pytest.

## Global Constraints

- Linux is authoritative; client tests run on Linux, Windows, and macOS.
- All UCIR operations preserve explicit domains, widths, signedness, and endianness.
- Backends run out of process; backend failure cannot terminate the orchestrator.
- Only proven or assumption-validated facts become mandatory.

---

### Task 1: Workspace, schemas, and verification harness

**Files:** Create `Cargo.toml`, `rust-toolchain.toml`, `crates/atlas-protocol/{Cargo.toml,src/lib.rs}`, `schemas/atlas/v1/{ucir.proto,facts.proto,jobs.proto,events.proto}`, `scripts/{verify.sh,verify.ps1}`, `.github/workflows/ci.yml`, `.gitignore`.

**Interfaces:** Produces `atlas_protocol::v1` generated types and `scripts/verify.* --profile core`.

- [ ] Write `crates/atlas-protocol/tests/schema.rs` asserting schema version `1`, deterministic encode/decode, and rejection of unknown major versions.
- [ ] Run `cargo test -p atlas-protocol --test schema`; expect failure because the crate does not exist.
- [ ] Create the workspace and protocol crate; expose `pub const SCHEMA_MAJOR: u32 = 1` and `decode_envelope(bytes: &[u8]) -> Result<Envelope, ProtocolError>`.
- [ ] Make both verification scripts run format checks, clippy with `-D warnings`, Rust tests, and Python tests, returning the first nonzero status.
- [ ] Run `./scripts/verify.ps1 -Profile core`; expect all Track 1 tests currently present to pass.
- [ ] Commit with `git commit -am "build: establish Atlas workspace and schemas"` after adding new files.

### Task 2: Typed immutable UCIR and canonical serialization

**Files:** Create `crates/atlas-ucir/{Cargo.toml,src/{lib.rs,types.rs,expr.rs,builder.rs,eval.rs,canonical.rs,provenance.rs},tests/{semantics.rs,properties.rs,serialization.rs}}`.

**Interfaces:** Produces `Type`, `ExprId`, `ExprGraph`, `Builder`, `Value`, `Evaluator::evaluate(&ExprGraph, &Model)`, and `canonical_hash(&ExprGraph) -> [u8; 32]`.

- [ ] Write table tests for 8/32-bit wrapping arithmetic, signed/unsigned comparisons, big/little-endian load/store, modular reduction, arrays, and source provenance.
- [ ] Write proptests asserting serialization round trips and `evaluate(original) == evaluate(simplified)` for the initial identity transforms.
- [ ] Run `cargo test -p atlas-ucir`; expect compilation failure for missing UCIR types.
- [ ] Implement hash-consed immutable nodes, explicit type checking in `Builder`, deterministic topological serialization, and an evaluator returning located errors for missing model values.
- [ ] Run `cargo test -p atlas-ucir`; expect all semantic, property, and serialization cases to pass.
- [ ] Commit with message `feat(ucir): add exact typed constraint representation`.

### Task 3: Sound simplification and component analysis

**Files:** Create `crates/atlas-analysis/{Cargo.toml,src/{lib.rs,pipeline.rs,general.rs,bitvec.rs,algebra.rs,partition.rs},tests/{golden.rs,properties.rs}}`; create `tests/fixtures/ucir/*.json`.

**Interfaces:** Consumes `ExprGraph`; produces `AnalysisResult { graph, components, features, derivations }` through `analyze(graph: &ExprGraph) -> Result<AnalysisResult, AnalysisError>`.

- [ ] Add golden cases for constant/equality/range propagation, XOR cancellation, rotate/mask normalization, independent components, contradiction detection, and GF(2)-affine recognition.
- [ ] Add property tests that compare all models for randomly generated small original and simplified graphs.
- [ ] Run `cargo test -p atlas-analysis`; expect failures for missing `analyze`.
- [ ] Implement a deterministic pass manager; each rewrite must emit a provenance derivation and may not mark an approximate inference as sound.
- [ ] Run `cargo test -p atlas-analysis`; expect golden snapshots and exhaustive bounded-model comparisons to pass.
- [ ] Commit with message `feat(analysis): add sound normalization and partitioning`.

### Task 4: Facts, planner, scheduler, and backend protocol

**Files:** Create `crates/atlas-facts`, `crates/atlas-planner`, `crates/atlas-scheduler`, `backends/sdk-python/atlas_backend_sdk`, `backends/z3`, `backends/sage`, and corresponding Rust/Python tests.

**Interfaces:** Produces `FactStore::publish(Fact) -> PublishOutcome`, `Planner::plan(&Features, &Capabilities) -> Portfolio`, and `Scheduler::run(Portfolio, CancellationToken) -> EventStream`; backend RPCs are `prepare`, `solve`, `publish_facts`, `accept_facts`, `model`, `explain`, `cancel`, and `health`.

- [ ] Test rejection of heuristic mandatory facts, assumption-scope mismatch, conflicting assignments, backend crashes, timeout/cancellation, and deterministic ten-rule routing for XOR, bit-vector, modular, polynomial, lattice, bounded-search, branching, and independent-component features.
- [ ] Run `cargo test -p atlas-facts -p atlas-planner -p atlas-scheduler`; expect missing-crate failures.
- [ ] Implement immutable fact records and provenance-aware conflict outcomes; implement staged budgets and process supervision with bounded stdout/stderr.
- [ ] Implement Z3 translation for Track 1 UCIR domains and a SageMath JSON-RPC adapter for modular, polynomial, and matrix requests; both report versions/capabilities on health.
- [ ] Run the three Rust suites and `pytest backends`; expect crash isolation, cancellation, routing, and adapter conformance to pass.
- [ ] Commit with message `feat(runtime): orchestrate isolated solver portfolios`.

### Task 5: Validation, cache, reports, CLI, and Python SDK

**Files:** Create `crates/atlas-validator`, `crates/atlas-cache`, `crates/atlas-report`, `crates/atlas-cli`, `python/{pyproject.toml,atlasctf,tests}`, `examples`, and `tests/e2e`.

**Interfaces:** Produces `Validator::validate(Candidate, ValidationContext) -> ResultLevel`, `Cache::get/put`, JSON `SolveReportV1`, CLI commands `solve|inspect|benchmark|doctor`, and Python `Project.solve(...) -> Result`.

- [ ] Add failing end-to-end fixtures for valid SAT, checker-rejected model, proven UNSAT, model-only, partial timeout, backend crash, cache corruption, and deterministic report reproduction.
- [ ] Run `cargo test -p atlas-validator -p atlas-cache -p atlas-report -p atlas-cli` and `pytest python/tests`; expect missing API failures.
- [ ] Implement UCIR reevaluation before result promotion, isolated checker replay, integrity-checked SQLite/content cache, redacted structured reports, CLI JSON mode, and PyO3-backed Python project/value/result APIs.
- [ ] Add 20 authorized micro-benchmarks and record preprocessing and fact-sharing on/off comparisons in machine-readable manifests.
- [ ] Run `./scripts/verify.ps1 -Profile core`; expect all tests and benchmark correctness gates to pass.
- [ ] Commit with message `feat: deliver Atlas core orchestration release`.

