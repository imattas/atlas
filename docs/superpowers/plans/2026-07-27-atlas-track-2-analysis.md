# AtlasCTF Track 2 Program Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe x86-64 reversing, symbolic/concolic execution, and specialized mathematical solvers through Track 1 contracts.

**Architecture:** Frontends lower source, traces, and lifted machine semantics into UCIR. Execution and specialized strategies publish scoped facts and candidates through the existing backend protocol, and concrete replay remains the final authority.

**Tech Stack:** Rust, LLVM-based lifter or stable disassembly/lifting library selected by an adapter spike, Unicorn/QEMU reference emulation, Z3, SageMath, fplll/flatter.

## Global Constraints

- Challenge artifacts are read-only and execute in the Track 1 sandbox with networking disabled.
- Unsupported instructions produce located errors or explicit controlled concretization.
- Every specialized solver result is translated into facts/candidates and validated through Track 1.

---

### Task 1: Binary, source, and trace intake

**Files:** Create `frontends/{binary,source,trace}`, `crates/atlas-program`, and `tests/fixtures/frontends`.

**Interfaces:** Produces `Program`, `Architecture`, `InstructionSemantics`, `Trace`, and `Frontend::lower(Artifact) -> LoweringResult`.

- [ ] Add fixtures covering ELF metadata, x86-64 registers/flags/calls/stack, restricted C fixed-width arithmetic, restricted Python checkers, and concrete trace bounds.
- [ ] Run frontend tests; expect missing frontend registrations.
- [ ] Implement content/type detection, safe parser limits, source locations, and UCIR lowering without executing intake-time code.
- [ ] Differentially compare each supported lifted instruction fixture against the reference emulator.
- [ ] Run frontend and differential suites; expect all fixtures to pass and malformed artifacts to fail cleanly.
- [ ] Commit with message `feat(frontends): lower binary source and trace artifacts`.

### Task 2: Symbolic and concolic execution

**Files:** Create `crates/atlas-executor/src/{state,memory,path,merge,loops,summaries,concolic,coverage}.rs` and tests.

**Interfaces:** Produces `Executor::explore(Program, Inputs, ExecutionBudget) -> EventStream`, `SymbolicState`, and `PathCandidate`.

- [ ] Test symbolic registers/memory, aliases, branch constraints, bounded loops, state merge soundness, concolic seed mutation, coverage prioritization, and path/process limits.
- [ ] Run `cargo test -p atlas-executor`; expect missing executor failures.
- [ ] Implement copy-on-write states, explicit path predicates, solver-assisted feasibility, loop policies, summaries, and deterministic coverage scheduling.
- [ ] Test that exhausted path budgets return `PARTIAL`, never `UNSAT`, and preserve validated facts.
- [ ] Run executor tests and replay each candidate in the Track 1 sandbox.
- [ ] Commit with message `feat(executor): add symbolic and concolic exploration`.

### Task 3: Taint slicing and program simplification

**Files:** Create `crates/atlas-program-analysis/src/{taint,slice,branches,loops,calls}.rs` and tests.

**Interfaces:** Produces `slice_for_target(&Program, Target, Inputs) -> ProgramSlice` with provenance mapping back to instructions.

- [ ] Add checkers containing irrelevant logging, initialization, constant branches, bounded loops, and summarized library calls.
- [ ] Run tests; expect unsliced instruction counts.
- [ ] Implement backward target slicing, forward input taint, branch pruning, loop-bound inference, and versioned summaries for pure library functions.
- [ ] Differentially execute original and sliced bounded fixtures for every input; outputs and acceptance must match.
- [ ] Run program-analysis and differential suites.
- [ ] Commit with message `feat(analysis): add taint-guided program slicing`.

### Task 4: Specialized algebra and crypto strategies

**Files:** Create `plugins/strategies/{gf2,modular-matrix,lattice,crypto-recognizers}` and `benchmarks/track2`.

**Interfaces:** Each strategy implements backend conformance and publishes assignments, residues, factors, bounds, or candidates with exact assumptions.

- [ ] Add fixtures for XOR linear systems, modular matrices, RSA small-private-exponent weakness, low-degree polynomial systems, and bounded lattice small-root patterns.
- [ ] Run strategy conformance tests; expect missing plugin manifests.
- [ ] Implement Gaussian elimination over GF(2), modular row reduction with non-invertible pivot handling, recorded lattice basis variants, and structural recognizers that require confirmation before launching attacks.
- [ ] Differentially compare small cases with exhaustive search and validate every candidate through UCIR.
- [ ] Run conformance, differential, and benchmark correctness suites.
- [ ] Commit with message `feat(strategies): add specialized mathematical solvers`.

### Task 5: Track 2 end-to-end release gate

**Files:** Create `tests/e2e/track2`, `benchmarks/track2/manifest.toml`, and `docs/guides/reversing.md`.

**Interfaces:** Extends `atlas solve` with binary/source/trace targets and strategy diagnostics.

- [ ] Add authorized end-to-end challenges for bit-vector verification, symbolic paths, XOR, modular algebra, RSA weakness, polynomial solving, and lattices.
- [ ] Run `./scripts/verify.ps1 -Profile analysis`; expect failures until all challenge reports reach their declared result levels.
- [ ] Wire frontend, slicing, execution, specialized strategies, fact exchange, replay, and reports through the orchestrator.
- [ ] Record fact-sharing enabled/disabled benchmarks and assert at least one statistically repeatable improvement with identical validated results.
- [ ] Run the analysis profile; expect all correctness, security, docs, and benchmark gates to pass.
- [ ] Commit with message `feat: deliver Atlas program-analysis release`.

