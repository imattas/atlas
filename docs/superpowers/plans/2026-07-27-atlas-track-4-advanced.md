# AtlasCTF Track 4 Advanced Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Atlas with safe learned ranking, reusable summaries, x86-32/ARM64/WebAssembly execution, and a production-event-driven notebook debugger.

**Architecture:** Historical benchmark features train an optional ranker whose output is restricted to portfolio ordering and budgets. Architecture adapters and the notebook consume existing UCIR, execution, event, and provenance contracts rather than creating parallel semantics.

**Tech Stack:** Rust, Python/scikit-learn or equivalent transparent ranker, ONNX, architecture reference emulators, Jupyter widgets, TypeScript, Playwright.

## Global Constraints

- Learned output cannot publish facts, alter trust, validate candidates, or create terminal result levels.
- Architecture semantics must pass reference-emulator differential tests.
- Notebook state is derived solely from versioned production events and report artifacts.

---

### Task 1: Benchmark warehouse and safe learned ranker

**Files:** Create `crates/atlas-benchmark-db`, `python/atlas_ranker`, `models/model-card.md`, and tests.

**Interfaces:** Produces versioned `BenchmarkRecord`, `RankRequest`, `RankResponse { ordered_strategy_ids, budget_multipliers, explanation }`, and planner fallback rules.

- [ ] Test schema ingestion, train/test corpus separation, deterministic seeds, missing features, corrupt/incompatible model, offline fallback, and allowlisted output fields.
- [ ] Run ranker tests; expect missing schema/model service.
- [ ] Implement benchmark ingestion, a transparent ranking baseline, ONNX export, model metadata, and orchestrator output validation that rejects unknown strategies and unsafe fields.
- [ ] Add an invariant test proving ranker responses cannot construct facts, assumptions, candidates, validation evidence, or result levels.
- [ ] Run tests and benchmark ranking against the rule baseline; commit `feat(planner): add safe optional learned ranking`.

### Task 2: Versioned function-summary library

**Files:** Create `summaries/schema`, `summaries/libc`, `crates/atlas-summaries`, and differential tests.

**Interfaces:** Produces `SummaryManifest`, `SummaryRegistry::resolve(symbol, abi, version)`, and UCIR pre/postcondition effects.

- [ ] Test exact ABI/version selection, ambiguity rejection, memory effects, error returns, unsupported calls, and provenance.
- [ ] Run summary tests; expect missing registry.
- [ ] Implement schema validation and summaries for the bounded pure/string/checksum functions used by the authorized corpus.
- [ ] Differentially compare summaries with sandboxed concrete calls over generated bounded inputs.
- [ ] Run tests and commit `feat(executor): add verified function summaries`.

### Task 3: x86-32, ARM64, and WebAssembly frontends

**Files:** Extend `frontends/binary`, add architecture modules and fixtures under `tests/fixtures/architectures`.

**Interfaces:** Implements the Track 2 `Architecture` and `InstructionSemantics` contracts for all three targets.

- [ ] Add calling-convention, register/flag, arithmetic, branch, load/store, and supported SIMD fixtures per architecture.
- [ ] Run architecture tests; expect unsupported-architecture diagnostics.
- [ ] Implement lowering with explicit pointer widths and endianness; WebAssembly uses its stack/memory semantics without pretending to be a native ABI.
- [ ] Differentially compare supported instructions and small programs against reference engines; unsupported subsets must report precise locations.
- [ ] Run architecture/e2e suites and commit `feat(frontends): add x86-32 ARM64 and WebAssembly`.

### Task 4: Notebook debugger and event replay

**Files:** Create `notebook/atlas_widget/{python,src,tests}`, event golden fixtures, and browser tests.

**Interfaces:** Consumes `EventEnvelopeV1` and `SolveReportV1`; displays constraints, domains, strategies, resources, facts, models, validation, provenance, and replay commands.

- [ ] Add Python model tests and Playwright tests for live streaming, recorded replay, reconnect/deduplication, large graphs, cancellation, redaction, and unknown compatible event fields.
- [ ] Run widget tests; expect missing package and UI.
- [ ] Implement a bounded virtualized event store, graph/detail views, accessible keyboard navigation, progress/resource panels, fact provenance, and report export.
- [ ] Replay Track 1-3 recorded production event streams; assert UI state matches golden report summaries and never exposes redacted values.
- [ ] Run notebook/browser tests and commit `feat(ui): add production event notebook debugger`.

### Task 5: Full-platform completion audit and release

**Files:** Create `release/manifest.schema.json`, `release/write-manifest.sh`, `docs/{installation,security,plugins,architecture}.md`, and full verification fixtures.

**Interfaces:** Produces a signed release manifest containing source revision, schema/tool versions, test evidence, supported capability matrix, image digests, and benchmark report hashes.

- [ ] Add a manifest test rejecting absent required suites, skipped required tests, unsigned artifacts, missing architecture evidence, and benchmark claims without hardware/sample metadata.
- [ ] Run `./scripts/verify.ps1 -Profile full`; expect failure until all evidence is present.
- [ ] Complete cross-platform client packaging, Linux OCI images, dependency/license inventory, security policy, plugin conformance docs, installation guides, and reproducible examples for every track.
- [ ] Run the full verifier in clean Linux and client OS environments; independently replay every release example and validate manifest hashes.
- [ ] Confirm every completion criterion in the approved spec maps to passing evidence in the manifest; commit with message `release: complete AtlasCTF full platform`.

