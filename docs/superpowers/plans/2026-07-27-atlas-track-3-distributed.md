# AtlasCTF Track 3 Acceleration and Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add correctness-preserving native, SIMD, GPU, and authenticated distributed execution for bounded regular search.

**Architecture:** A restricted search IR is compiled to local targets only after a cost model selects them. Remote workers execute signed content-addressed jobs in the same isolation model and return facts that pass coordinator validation.

**Tech Stack:** Rust, Cranelift, portable SIMD, CUDA, tonic/rustls mutual TLS, OCI containers, property/differential testing.

## Global Constraints

- Accelerators never bypass CPU UCIR/checker validation.
- Jobs are idempotent, signed, scoped, integrity-checked, cancellable, and free of host credentials.
- GPU absence must skip optional hardware execution while still running compile and CPU differential coverage.

---

### Task 1: Restricted search IR and native compiler

**Files:** Create `crates/atlas-search-ir`, `crates/atlas-search-native`, and tests.

**Interfaces:** Produces `SearchProgram::try_from_ucir(slice)`, `SearchDomain`, and `NativeSearcher::search(program, domain, cancellation) -> MatchStream`.

- [ ] Add accepted/rejected lowering fixtures, including arithmetic, bitwise operations, hashes/checksums in the supported set, forbidden memory aliasing, and data-dependent unbounded loops.
- [ ] Run tests; expect missing lowerer/compiler failures.
- [ ] Implement auditable validation and Cranelift scalar kernels with bounded output and cancellation polling.
- [ ] Compare all small-domain matches with the UCIR evaluator.
- [ ] Run search IR/native suites and commit `feat(search): add verified native bounded search`.

### Task 2: SIMD and GPU execution

**Files:** Create `crates/atlas-search-simd`, `gpu/cuda`, kernel cache tests, and CI compile jobs.

**Interfaces:** Produces `SimdSearcher` and `GpuSearcher` implementing the same `Searcher` trait as native execution.

- [ ] Add randomized batch tests for widths, tails, cancellation, multiple matches, no matches, and kernel-cache key changes across compiler/device/options.
- [ ] Run differential tests; expect missing SIMD/GPU implementations.
- [ ] Implement portable-SIMD batches and CUDA code generation for the restricted IR, with bounds-checked buffers and deterministic match compaction.
- [ ] Compare CPU, SIMD, and GPU outputs on every generated program; CPU validation must reject an injected false GPU match.
- [ ] Run hardware-independent compile tests and GPU-tagged tests on a configured runner; commit `feat(search): add differential-tested SIMD and CUDA search`.

### Task 3: Placement cost model

**Files:** Create `crates/atlas-placement` and benchmark calibration fixtures.

**Interfaces:** Produces `PlacementModel::choose(SearchFeatures, Capabilities) -> PlacementDecision` with recorded rationale.

- [ ] Test scalar selection for tiny/divergent jobs, SIMD for medium regular jobs, GPU above measured transfer/compile break-even, and cache-hit effects.
- [ ] Run placement tests; expect missing decisions.
- [ ] Implement deterministic baseline rules plus loadable calibration measurements; never select unavailable capabilities.
- [ ] Assert decision explanations appear in solve reports and benchmark manifests.
- [ ] Run placement tests and commit `feat(planner): add accelerator placement model`.

### Task 4: Authenticated coordinator and worker

**Files:** Create `crates/atlas-worker`, `crates/atlas-coordinator`, `deploy/worker`, and integration tests.

**Interfaces:** Produces worker registration, capability heartbeat, job lease, result submission, cancellation, and artifact-fetch RPCs over mutual TLS.

- [ ] Test valid registration, untrusted certificate, expired lease, tampered job/result, duplicate result, disconnect/requeue, cancellation, least-capability scheduling, and coordinator restart recovery.
- [ ] Run distributed tests; expect missing services.
- [ ] Implement signed content-addressed envelopes, SQLite lease state, capability matching, bounded artifact transfer, and deduplicated fact ingestion.
- [ ] Run workers as non-root network-disabled job containers; test denial of host environment, filesystem, Docker socket, and unrelated artifacts.
- [ ] Run distributed/security suites and commit `feat(distributed): add isolated authenticated workers`.

### Task 5: Track 3 release gate

**Files:** Create `benchmarks/track3`, `tests/e2e/track3`, and `docs/guides/workers.md`.

**Interfaces:** Extends CLI with `atlas worker` and coordinator configuration while preserving local defaults.

- [ ] Add bounded verifier benchmarks with scalar, SIMD, GPU, and two-worker expected result equivalence.
- [ ] Run `./scripts/verify.ps1 -Profile distributed`; expect incomplete benchmark/security evidence.
- [ ] Wire placement, kernel caching, remote scheduling, cooperative cancellation, and report provenance end to end.
- [ ] Record acceleration break-even and distributed overhead with hardware metadata and repeated samples.
- [ ] Run the distributed profile; expect all differential, isolation, retry, docs, and benchmark gates to pass; commit `feat: deliver Atlas distributed acceleration release`.

