# AtlasCTF Full Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete AtlasCTF platform defined by the approved full-platform specification.

**Architecture:** A Rust Linux runtime owns typed constraints, orchestration, validation, isolation, and distributed scheduling. Cross-platform CLI/Python clients and notebook UI communicate through versioned contracts; solver, execution, and accelerator backends remain isolated processes.

**Tech Stack:** Rust stable, Python 3.11-3.13, PyO3/maturin, Protobuf/tonic, SQLite, OCI/Docker, Z3, SageMath, LLVM-based lifting, CUDA, Jupyter widgets, pytest, cargo-nextest, proptest.

## Global Constraints

- The authoritative runtime targets Linux; Windows uses WSL2 or Docker.
- Python support is 3.11 through 3.13.
- Every final SAT result must be validated against UCIR and the original checker when available.
- Untrusted artifacts run without networking and without host secrets under explicit resource limits.
- Learned and heuristic output may rank work but may never assert correctness or become mandatory without validation.
- Persisted schemas and public protocols are versioned; generated code is reproducible and drift-checked.
- No release gate may be waived through ignored required tests or undocumented exceptions.

---

## Plan Set and Dependency Order

- [ ] **Track 1:** Execute [core orchestration](2026-07-27-atlas-track-1-core.md). Deliver one API that solves, validates, explains, and benchmarks equation/SMT challenges through Z3 and SageMath.
- [ ] **Track 2:** Execute [program analysis and specialized solving](2026-07-27-atlas-track-2-analysis.md). Deliver x86-64 reversing plus GF(2), modular, lattice, and crypto strategies through the Track 1 contracts.
- [ ] **Track 3:** Execute [acceleration and distribution](2026-07-27-atlas-track-3-distributed.md). Deliver differential-tested SIMD/GPU search and authenticated isolated remote workers.
- [ ] **Track 4:** Execute [advanced automation and experience](2026-07-27-atlas-track-4-advanced.md). Deliver learned ranking, added architectures, function summaries, and the notebook debugger.
- [ ] **Completion audit:** Run `./scripts/verify.ps1 -Profile full` on Windows and `./scripts/verify.sh --profile full` in Linux CI. Expected: all formatting, lint, unit, property, differential, integration, security, schema, docs, and benchmark gates pass with no skipped required tests.

## Final Evidence Map

| Specification claim | Authoritative evidence |
|---|---|
| Exact UCIR semantics | UCIR property and differential suites from Track 1 |
| Solver coordination and fact sharing | Orchestrator integration tests and benchmark delta report |
| Concrete validation and explanations | Validator replay tests and golden report fixtures |
| Reversing and specialized mathematics | Track 2 end-to-end corpus |
| SIMD/GPU correctness | Track 3 CPU-versus-accelerator differential suite |
| Secure distributed workers | Mutual-auth, isolation, retry, and tamper integration tests |
| Learned planner safety | Track 4 invariants proving ranking cannot create facts/results |
| Four architecture frontends | Per-instruction emulator differential suites |
| Notebook production integration | Browser tests consuming recorded production event streams |
| Complete release readiness | Signed full verification manifest and reproducible benchmark report |

