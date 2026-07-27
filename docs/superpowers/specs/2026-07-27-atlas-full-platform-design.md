# AtlasCTF Full Platform Design

## Status

Final design approved for implementation on 2026-07-27.

## 1. Product Definition

AtlasCTF is a defensive, competition-focused platform for solving authorized CTF, educational-lab, and locally supplied constraint, cryptography, and reversing challenges. It coordinates mature solver engines rather than replacing them. Atlas normalizes each challenge into a typed common representation, discovers structure, partitions the work, selects appropriate engines and hardware, exchanges trustworthy facts between strategies, validates every candidate, and emits a reproducible explanation.

The full product includes the capabilities described across all five phases of the source design: constraint orchestration, reversing, specialized mathematics, local and distributed acceleration, learned strategy ranking, expanded architecture support, and notebook visualization.

Atlas does not target unauthorized systems, claim to break correctly implemented modern cryptography, or treat heuristic or learned output as proof.

## 2. Deployment Model

The authoritative runtime targets Linux. Native Linux, a network-disabled Docker environment, or WSL2 hosts the runtime. The CLI and Python SDK are cross-platform and communicate with a local runtime over a Unix socket or localhost transport, or with an explicitly configured remote coordinator over authenticated TLS.

The core is written in Rust. Python provides the public SDK, notebooks, SageMath integration, and user plugins. Existing C and C++ APIs are isolated behind backend adapter processes. GPU kernels use CUDA first, with a backend boundary that permits HIP or another portable implementation later.

## 3. Delivery Architecture

Atlas is built as four vertical release tracks. Each track must produce end-to-end, benchmarked functionality rather than disconnected infrastructure.

1. **Core orchestration:** UCIR, Python/CLI interfaces, simplification, planning, Z3 and SageMath adapters, fact exchange, validation, explanations, caching, and resource limits.
2. **Program and specialized solving:** binary/source/trace frontends, x86-64 symbolic and concolic execution, GF(2), modular matrix, lattice, crypto recognizers, and concrete replay.
3. **Acceleration and distribution:** native and SIMD candidate checking, GPU lowering and execution, signed remote jobs, worker capability scheduling, isolation, cancellation, and artifact return.
4. **Advanced automation and experience:** learned strategy ranking, reusable function summaries, x86-32/ARM64/WebAssembly expansion, and a visual notebook debugger.

Every track extends stable contracts for UCIR, facts, jobs, validation, and provenance. A later track must not weaken correctness or require consumers to bypass those contracts.

## 4. System Components

### 4.1 Public interfaces

The `atlas` CLI supports at least `solve`, `inspect`, `benchmark`, `worker`, and `doctor`. Machine-readable JSON output is available for every non-interactive command. The `atlasctf` Python package exposes projects, typed values, constraints, solver configuration, results, explanations, and plugin APIs. Notebook widgets consume the same event stream and result model; they are not a separate solver implementation.

### 4.2 Challenge intake and frontends

Intake accepts equations, SMT-LIB, UCIR JSON/binary documents, restricted C/Python checkers, binaries, concrete traces, and plugin-provided artifacts. Every input is copied or mounted read-only into an isolated job workspace and content-addressed.

Frontends lower inputs to UCIR while retaining source locations and original artifacts. Initial binary lifting covers x86-64; later releases add x86-32, ARM64, and WebAssembly. Unsupported instructions or constructs produce located diagnostics and may be handled through controlled concretization, emulation, or plugin summaries.

### 4.3 Unified Constraint IR

UCIR is a typed, immutable expression graph. It supports booleans, arbitrary integers, rationals/reals, floating point, fixed-width bit-vectors, bytes, arrays and symbolic memory, modular integers, finite fields, polynomials, matrices/vectors, and elliptic-curve points.

Nodes carry domain, bit width, signedness, endianness, source location, transformation history, confidence, and approximation status. Memory operations always state width and endianness. Signed and unsigned operations are distinct. Wrapping machine operations never silently become mathematical integers.

Canonical serialization is versioned and deterministic. Equivalent canonical graphs yield identical content hashes. Unknown future fields are rejected or preserved according to the schema version; they are never silently discarded.

### 4.4 Analysis and normalization

The analysis pipeline performs constant folding, dead-expression removal, common-subexpression reuse, equality and range propagation, substitution, contradiction detection, deduplication, independence partitioning, and domain-specific passes.

Bit-vector passes include XOR cancellation, rotation normalization, mask propagation, slicing, carry isolation, GF(2) affine detection, independent-lane discovery, and controlled Boolean lowering. Algebra passes include polynomial normalization, GCDs, factorization requests, CRT decomposition, finite-field linear algebra, low-degree detection, and small-root recognition. Program passes include branch pruning, taint slicing, bounded-loop analysis, function summaries, and removal of target-irrelevant instructions.

Every sound transformation records a checkable provenance edge. Approximate transformations may rank work but cannot remove candidate solutions.

### 4.5 Strategy planner and scheduler

The planner extracts transparent features such as domains, variable/constraint counts, widths, operation distribution, polynomial degree, modulus size, sparsity, memory and branch shape, component independence, and recognized crypto structures.

The baseline planner uses explicit, inspectable routing rules. It produces a staged portfolio with time, CPU, memory, process, output, path, and accelerator budgets. Independent components and suitable strategies run concurrently. Progress signals determine budget extensions. All jobs support cooperative cancellation and return useful validated facts before shutdown.

The learned planner is introduced only after benchmark data is available. It ranks strategies, tactics, and budgets; it never determines satisfiability, mandatory facts, or final validity.

### 4.6 Backend adapters

Adapters implement prepare, solve, fact streaming, fact acceptance, model extraction, explanation, health, and cancellation. Backends run out of process by default.

The required complete-platform adapter families are:

- SMT/SAT: Z3 plus adapters for cvc5, Bitwuzla, at least one modern SAT engine, and an XOR-aware SAT engine.
- Algebra: SageMath, with protocol support for PARI/GP, FLINT, Singular, or NTL capabilities where installed.
- Lattice: fplll/flatter/Sage wrappers with multiple recorded basis formulations.
- Execution: symbolic/concolic program states, coverage-guided exploration, loop controls, state merging, summaries, and concrete replay.
- Search: native scalar, CPU SIMD, GPU, and remote worker execution for bounded regular candidate checks.

Backends advertise capabilities and versions. Missing optional backends degrade to a precise diagnostic and alternate strategy; missing required capability for a requested strategy is not silently ignored.

### 4.7 Shared fact store

Facts are immutable records with type, payload, scope, assumptions, producer, derivation, trust level, and content hash. Supported facts include assignments, bounds, equalities/disequalities, residues, independent groups, conflict clauses, partial models, factors, and candidate prefixes.

Trust levels are proven, solver-certified, concrete-observed, heuristic, and approximate. Only proven facts and facts validated for the current assumptions may become mandatory. Conflicts trigger provenance-aware validation; they do not use last-writer-wins behavior.

### 4.8 Validation and result model

Every candidate is re-evaluated against original UCIR. When an original checker exists, it is also replayed in isolation. Results use these levels:

- `PROVEN_UNSAT`: a configured trusted backend produced an accepted unsatisfiability result.
- `VALIDATED_SAT`: UCIR and available original-checker validation both succeeded.
- `MODEL_ONLY`: a model exists, but original validation is unavailable.
- `PARTIAL`: sound useful deductions exist without a complete answer.
- `UNKNOWN`: no conclusion was reached within supported theories and budgets.

Reports include inputs and hashes, assumptions, transformations, strategy decisions, backend versions, facts exchanged, resource usage, candidate provenance, validation evidence, and a reproduction command. Secrets and raw challenge data are redacted according to explicit reporting policy.

### 4.9 Caching

Content-addressed cache keys include canonical UCIR, solver and plugin versions, tactic configuration, architecture semantics, and assumptions. Cacheable artifacts include lifted functions, simplified graphs, summaries, clauses, factors, lattice bases, kernels, and validated models.

Cache entries carry schema versions, trust status, integrity hashes, and provenance. Heuristics are never read back as proof. Corrupt or incompatible entries are quarantined and recomputed.

### 4.10 Acceleration

The search compiler accepts only a restricted, auditable verifier subset. A cost model accounts for compilation, transfer, batch size, divergence, and expected search reduction before selecting SIMD or GPU execution. Matches always return to CPU validation.

GPU kernels are cached by verifier IR, target architecture, compiler version, and options. FPGA support is a plugin/worker capability for stable repeated kernels, not a mandatory developer dependency.

### 4.11 Distributed workers

Workers register CPU, memory, accelerator, backend, and architecture capabilities. The coordinator schedules content-addressed idempotent jobs. Jobs and results are signed, scoped, versioned, and replayable. Remote transport uses mutual authentication and authorization; a worker never receives host credentials or unrelated artifacts.

Disconnects requeue safe jobs. Duplicate results are deduplicated by job and fact hashes. Cancellation is propagated, and returned facts are accepted only after schema, integrity, assumption, and trust validation.

### 4.12 Plugins

Plugins declare category, compatibility, supported UCIR operations, dependencies, expected features, soundness, costs, hardware requirements, permissions, and output fact types. Categories include frontends, recognizers, simplifiers, solver adapters, strategies, kernels, validators, and reporters.

Plugins run with least privilege. Native and Python plugins are isolated unless explicitly designated as trusted in local configuration. Plugin API and manifest schemas are versioned and tested with conformance fixtures.

## 5. Data Flow

1. Intake hashes and isolates supplied artifacts.
2. A frontend lowers relevant semantics to UCIR with provenance.
3. Normalization simplifies and partitions the graph.
4. The planner constructs a staged, budgeted portfolio.
5. The scheduler launches isolated local or remote jobs.
6. Backends publish facts; the fact store validates and redistributes eligible deductions.
7. Candidate models are validated against UCIR and available original checkers.
8. Cancellation stops speculative work after a terminal validated result.
9. The reporter emits structured results, explanation, artifacts, and reproduction instructions.

## 6. Security and Safety

Atlas assumes every artifact, script, binary, plugin, backend output, and remote worker is untrusted. Challenge execution has networking disabled by default, a read-only challenge mount, ephemeral writable storage, syscall restrictions, process/user namespaces or microVM isolation, non-root execution, environment allowlisting, and CPU, memory, process, file, output, and time limits.

Remote targets require explicit authorization and are outside the automatic local workflow. The platform must not scan arbitrary hosts, discover targets, or transmit challenge material unless the user explicitly configures an authorized endpoint.

The runtime records executed commands and sandbox policy. Host secrets, SSH agents, cloud credentials, Docker sockets, and broad filesystem mounts are never exposed to jobs.

## 7. Failure Semantics

Backend crashes are isolated and reported before alternate strategies run. Memory exhaustion preserves already validated facts and may trigger repartitioning. Timeouts return partial results and recommendations. Translation mismatches identify the smallest known source/UCIR/backend discrepancy. Unsupported semantics carry exact locations. Worker loss requeues idempotent work. Conflicting facts remain separate until validation identifies invalid scope, assumptions, translation, or producer output.

No internal exception may be presented as `UNSAT`, and no unvalidated candidate may be presented as `VALIDATED_SAT`.

## 8. Repository Structure

The repository is a Cargo workspace with focused crates for UCIR, protocol, fact storage, orchestration, scheduling, validation, sandboxing, frontends, and backend clients. The Python package uses maturin/PyO3 for stable core bindings while network/process protocols remain language-neutral. Backend services, plugins, benchmarks, test fixtures, documentation, examples, notebook UI, deployment assets, and GPU code occupy separate top-level directories.

Public contracts live in versioned schemas. Generated code is reproducible and checked for drift. Architectural boundaries prevent Python SDK code, UI code, or solver-specific types from leaking into the UCIR core.

## 9. Testing and Verification

### Unit and property tests

Tests cover every UCIR type and operation, signedness, overflow, endianness, memory, serialization, hashing, provenance, simplification equivalence, fact trust transitions, scheduling budgets, cancellation, caching, and protocol compatibility. Property tests assert that sound simplification preserves models and that heuristic facts never become mandatory without validation.

### Differential tests

Generated bounded problems compare the UCIR evaluator with native/reference execution, compatible solver backends with one another, lifted instructions with an emulator, and CPU search with SIMD/GPU search. Disagreement is a test failure with a minimized fixture.

### Integration and security tests

End-to-end fixtures cover each frontend, solver family, validation path, crash/timeout/resource failure, cache lifecycle, plugin isolation, worker disconnect/retry, authentication, malicious archives, hostile checkers, and redaction.

### Benchmark corpus

The repository includes only authorized, redistributable fixtures grouped by bit-vectors, XOR systems, modular algebra, RSA weaknesses, polynomial systems, lattices, path exploration, and bounded search. Metrics include first validated result, total time, peak memory, solver calls, preprocessing reduction, fact-sharing benefit, validation rate, acceleration break-even, and distributed overhead.

Performance claims require pinned benchmark manifests, hardware/software metadata, baselines, multiple samples, and machine-readable results.

## 10. Release Gates

Each release track must pass formatting, linting, unit/property/differential/integration tests, protocol compatibility, security tests, documentation checks, and its benchmark acceptance thresholds on supported Linux CI. Cross-platform SDK and CLI tests run on Windows, macOS, and Linux.

The complete product is achieved only when:

- All public interfaces and component contracts in this specification exist and are documented.
- Modular mathematics and exact fixed-width program semantics round-trip without loss.
- Required solver families operate through the common adapter protocol when their dependencies are installed.
- All four declared binary architectures have tested lifting/execution coverage for the supported instruction subset.
- Fact sharing demonstrably improves at least one representative benchmark without compromising correctness.
- Every reported terminal SAT result follows the validation contract.
- CPU SIMD, GPU, and authenticated distributed workers pass differential and failure tests.
- Learned ranking is benchmark-trained, optional, explainable, and unable to assert correctness.
- Notebook visualization consumes the production event/provenance model.
- Isolation tests show that supplied artifacts cannot access host secrets or networking under default policy.
- Reproducible end-to-end examples cover every release track.
- No placeholders, stub implementations, ignored required tests, or undocumented completion exceptions remain.

## 11. Initial Compatibility Policy

The first stable release supports the latest stable Rust toolchain at release time, Python 3.11 through 3.13, and current supported Linux distributions through OCI images. Protocol and cache schemas use semantic versions. Patch releases preserve schema compatibility; breaking contract changes require a major version and migration tooling where persisted data is affected.

## 12. Source Design Relationship

This document makes `design.md` implementation-ready by resolving its delivery boundary and deployment model. `design.md` remains the product rationale and capability catalog. If wording conflicts, this approved specification controls implementation scope and acceptance.
