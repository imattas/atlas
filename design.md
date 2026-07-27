# CTF Solver Platform Design

## 1. Overview

This document specifies a defensive, competition-focused solver platform that combines the strongest ideas from SageMath, Z3, symbolic execution, program analysis, and hardware-accelerated search.

Working name: **AtlasCTF**.

AtlasCTF is not intended to replace SageMath or Z3. It coordinates them, selects suitable solving methods automatically, shares facts between solvers, and uses CPU, GPU, FPGA, or remote workers only when a workload matches the hardware.

The central design principle is:

> Normalize a challenge into a common intermediate representation, identify exploitable structure, route each subproblem to the best solver, and continuously exchange deductions between solvers.

The system targets authorized CTF challenges, educational labs, and locally supplied binaries or challenge data.

---

## 2. Goals

### 2.1 Primary goals

- Solve common CTF crypto, reversing, constraint, and algebra challenges faster than manually coordinating separate tools.
- Expose a single Python and command-line interface.
- Automatically detect whether a problem is best handled by SAT/SMT, computer algebra, lattices, symbolic execution, concrete execution, brute force, or a hybrid strategy.
- Preserve exact machine semantics such as bit width, overflow, signedness, endianness, and memory aliasing.
- Run multiple solving strategies in parallel and stop losing strategies when one succeeds.
- Produce an explainable solution trace showing which deductions and attacks worked.
- Support reproducible local runs and isolated distributed workers.

### 2.2 Non-goals

- Automatically defeating correctly implemented modern cryptography.
- General-purpose offensive exploitation of real systems.
- Replacing mature backend engines such as Z3, cvc5, Boolector, SageMath, PARI/GP, FLINT, or fplll.
- Treating GPUs as universally faster than CPUs.
- Using machine learning as an unverified source of mathematical truth.

---

## 3. Why a Combined Tool Can Be Better

SageMath and Z3 are individually strong but have different internal models.

- SageMath understands mathematical domains such as rings, fields, polynomials, matrices, elliptic curves, and lattices.
- Z3 understands logical constraints over integers, reals, arrays, floating point, and fixed-width bit-vectors.
- Symbolic-execution engines understand program paths, registers, memory, and branch conditions.
- GPU kernels are effective for regular, massively parallel candidate testing.

A combined platform gains speed by avoiding a single universal strategy. It can split one challenge into components and assign each component to a specialized engine.

Example:

1. Symbolic execution extracts a verifier's branch conditions.
2. A bit-vector solver handles XOR, rotations, and overflow.
3. A computer-algebra engine simplifies modular polynomial equations.
4. A lattice engine attempts a bounded-small-root attack.
5. A GPU worker tests the remaining independent 24-bit suffix.
6. A concrete runner validates the final candidate against the original binary.

---

## 4. High-Level Architecture

```text
+------------------------------+
| CLI / Python API / Notebook  |
+---------------+--------------+
                |
                v
+------------------------------+
| Challenge Intake             |
| binary, source, equations,   |
| traces, packets, artifacts   |
+---------------+--------------+
                |
                v
+------------------------------+
| Frontends and Lifters        |
| machine code, Python, C,     |
| SMT-LIB, math expressions    |
+---------------+--------------+
                |
                v
+------------------------------+
| Unified Constraint IR        |
| values, memory, equations,   |
| domains, provenance          |
+---------------+--------------+
                |
                v
+------------------------------+
| Analyzer and Strategy Planner|
| classify, simplify, partition|
| estimate cost, choose engines|
+------+----------+------------+
       |          | 
       v          v
+-----------+  +----------------+
| Solver    |  | Execution      |
| Portfolio |  | and Search     |
+-----------+  +----------------+
       |          |
       +-----+----+
             v
+------------------------------+
| Shared Fact Store            |
| bounds, substitutions,       |
| lemmas, models, conflicts    |
+---------------+--------------+
                |
                v
+------------------------------+
| Candidate Validator          |
| concrete replay and proof    |
+---------------+--------------+
                |
                v
+------------------------------+
| Result, Script, Explanation  |
+------------------------------+
```

---

## 5. Unified Constraint Intermediate Representation

The unified intermediate representation, or **UCIR**, is the most important component. It allows frontends and solvers to communicate without depending on each other's internal data structures.

### 5.1 Value types

UCIR supports:

- Booleans
- Arbitrary-precision integers
- Fixed-width bit-vectors
- Signed and unsigned interpretations
- Rational and real values
- Floating-point values
- Byte strings
- Arrays and symbolic memory
- Modular integers
- Finite-field elements
- Polynomials
- Matrices and vectors
- Elliptic-curve points

### 5.2 Operations

- Arithmetic: addition, subtraction, multiplication, division, remainder
- Bitwise: XOR, AND, OR, NOT, shifts, rotates, extraction, concatenation
- Comparisons with explicit signedness
- Modular reduction and inversion
- Polynomial evaluation and factorization requests
- Memory load/store with explicit endianness
- Conditional expressions
- Function applications
- Path predicates

### 5.3 Semantic metadata

Every expression records:

- Bit width or mathematical domain
- Signedness
- Endianness
- Source location
- Original instruction or equation
- Transformation history
- Confidence level
- Whether an approximation was introduced

This metadata prevents common mistakes such as sending wrapping 32-bit multiplication to an unbounded-integer solver.

### 5.4 Provenance

Every derived fact records its origin. A final answer can therefore show:

```text
flag[7] = 0x61
  derived from verifier block 0x4012A0
  simplified by affine bit-vector pass
  confirmed by Z3 model
  validated by concrete execution
```

---

## 6. Frontends

### 6.1 Mathematical frontend

Accepts Python-like expressions and explicit domains:

```python
p = prime(bits=512)
x = bitvec(32)
y = modint(n)
require((x ^ 0x1337) + 9 == 0x41424344)
```

It can also import:

- SageMath expressions
- SymPy expressions
- SMT-LIB
- JSON constraints
- Matrix and polynomial files

### 6.2 Binary frontend

The binary frontend lifts machine code to UCIR through an existing disassembly and lifting framework.

Initial architecture support:

- x86-64
- x86-32
- ARM64
- WebAssembly

The lifter models:

- Registers
- Flags
- Memory
- Calling conventions
- Stack frames
- Branches
- Common library functions

### 6.3 Source frontend

A restricted C or Python frontend is useful for challenge checkers supplied as source. It should preserve fixed-width types and avoid translating all values into unbounded integers.

### 6.4 Trace frontend

Imports concrete traces and converts observed values into constraints or bounds. This allows concolic execution: concrete execution guides symbolic exploration.

---

## 7. Normalization and Simplification

Before invoking an expensive solver, AtlasCTF performs domain-aware simplification.

### 7.1 General simplifications

- Constant folding
- Dead-expression elimination
- Common-subexpression elimination
- Equality propagation
- Range propagation
- Substitution of fixed variables
- Contradiction detection
- Constraint deduplication

### 7.2 Bit-vector simplifications

- XOR cancellation
- Rotation normalization
- Mask propagation
- Bit slicing
- Carry-chain isolation
- Affine transformations over GF(2)
- Detection of independent bit lanes
- Conversion of suitable operations into Boolean circuits

### 7.3 Algebraic simplifications

- Polynomial normalization
- Greatest-common-divisor computation
- Modular factorization
- Chinese remainder decomposition
- Linear algebra over finite fields
- Detection of low-degree systems
- Detection of small-root structure

### 7.4 Program simplifications

- Constant branch pruning
- Function summaries
- Loop-bound inference
- Taint-guided slicing
- Removal of instructions unrelated to the target output

A good simplifier can provide a larger speedup than faster hardware because it reduces the actual problem sent to every backend.

---

## 8. Strategy Planner

The strategy planner classifies subproblems and creates a solver portfolio.

### 8.1 Feature extraction

The planner measures:

- Variable count
- Constraint count
- Bit widths
- Operation distribution
- Polynomial degree
- Modulus size
- Sparsity
- Memory usage
- Branch count
- Estimated independence between components
- Presence of known crypto structures

### 8.2 Rule-based first version

The first version should use transparent rules rather than machine learning.

Examples:

- Mostly XOR and linear bit operations: Gaussian elimination over GF(2), then SAT.
- Fixed-width arithmetic with carries: bit-vector SMT or SAT bit-blasting.
- Linear modular system: modular matrix solver.
- Low-degree polynomial equations: Gröbner basis or resultant methods.
- Approximate integer relation: lattice reduction.
- Small bounded unknown suffix: CPU SIMD or GPU enumeration.
- Many program branches: concolic execution with coverage guidance.
- Independent components: solve in parallel.

### 8.3 Learned planner as a later feature

A learned planner may rank strategies based on prior benchmark results. It must never decide correctness. It only chooses ordering, time budgets, and hardware placement.

---

## 9. Solver Portfolio

AtlasCTF uses backend adapters with a common interface.

```text
prepare(problem)
solve(time_budget, assumptions)
stream_facts()
accept_facts(facts)
extract_model()
explain_result()
cancel()
```

### 9.1 SMT and SAT backends

Potential backends:

- Z3
- cvc5
- Boolector or Bitwuzla for bit-vectors
- Kissat or CaDiCaL for SAT
- CryptoMiniSat for XOR-heavy Boolean systems

The portfolio may launch several engines with different tactics. The first validated model wins.

### 9.2 Computer-algebra backends

- SageMath adapter
- PARI/GP
- FLINT
- Singular
- NTL

Used for:

- Modular arithmetic
- Polynomial systems
- Finite fields
- Number theory
- Matrices
- Elliptic curves

### 9.3 Lattice backends

- fplll
- flatter
- Sage lattice wrappers

The lattice adapter generates multiple basis layouts and reduction settings because lattice attack performance is highly sensitive to formulation.

### 9.4 Symbolic-execution backend

The engine should support:

- Symbolic registers and memory
- Path constraints
- State merging
- Loop handling
- Function summaries
- Concolic execution
- Coverage-guided path selection

It can initially integrate an existing framework rather than creating a new lifter and executor from scratch.

### 9.5 Concrete validation backend

Every candidate from an approximate, heuristic, or translated solver must be checked against the original challenge.

Validation methods:

- Execute the supplied checker in a sandbox
- Re-evaluate the original UCIR constraints
- Compare expected output
- Record a deterministic replay trace

---

## 10. Shared Fact Store

Most solver portfolios waste work because engines run independently. AtlasCTF adds a shared fact store.

Facts include:

- Exact assignments
- Variable bounds
- Equalities
- Disequalities
- Modular residues
- Independent variable groups
- Learned conflict clauses
- Partial models
- Algebraic factors
- Candidate prefixes

### 10.1 Fact exchange examples

- The algebra engine determines `x mod 257 = 91`; the SMT solver receives the new constraint.
- The SMT solver proves the upper byte of `x` is zero; the lattice engine reduces its bound.
- Concrete execution rejects a prefix; the search engine adds a blocking clause.
- Gaussian elimination fixes 40 XOR variables before SAT begins.

### 10.2 Trust levels

Facts are labeled:

- Proven
- Solver-certified
- Concrete-observed
- Heuristic
- Approximate

Only proven or validated facts may be treated as mandatory constraints. Heuristic facts influence search order but cannot remove valid solutions.

---

## 11. Parallelism Model

### 11.1 Task-level parallelism

The largest practical gain comes from running different approaches concurrently:

- Different solvers
- Different tactic settings
- Different path-exploration strategies
- Different lattice embeddings
- Different variable partitions

### 11.2 Data-level parallelism

Use SIMD, GPU, or FPGA when many candidates execute the same operations independently.

Good GPU workloads:

- Hashing candidate suffixes
- Testing independent keys
- Evaluating simple arithmetic verifiers
- Batch modular exponentiation with shared sizes
- Large Boolean batches

Poor GPU workloads:

- Branch-heavy symbolic execution
- Irregular Gröbner-basis computation
- Small SAT instances
- Pointer-heavy graph algorithms
- Workloads with frequent CPU-GPU synchronization

### 11.3 Cooperative cancellation

All tasks receive cancellation tokens. Once a candidate is validated, remaining speculative workers stop and return their useful learned facts for caching.

---

## 12. Hardware Design

## 12.1 Recommended developer workstation

A balanced system is better than a GPU-only machine.

- CPU: 16 to 32 high-performance cores
- RAM: 64 GB minimum; 128 GB preferred for large symbolic states and algebra workloads
- Storage: 2 TB or larger NVMe SSD
- GPU: modern CUDA- or ROCm-capable GPU with 16 GB or more VRAM
- Optional FPGA: useful only for specialized repeated kernels
- Operating system: Linux

### 12.2 CPU responsibilities

CPUs handle:

- SAT and SMT control logic
- Symbolic execution
- Computer algebra
- Lattice reduction
- Branch-heavy workloads
- Task scheduling
- Final validation

Important CPU characteristics:

- Strong single-thread performance for solver bottlenecks
- Large caches
- High memory bandwidth
- Many cores for portfolio execution
- AVX2 or AVX-512 where available

### 12.3 Memory

Memory capacity often matters more than peak arithmetic throughput.

The runtime should use:

- Arena allocation for short-lived expressions
- Hash-consing to reuse identical expression nodes
- Compressed expression graphs
- Copy-on-write symbolic states
- Memory-mapped caches
- NUMA-aware worker placement on multi-socket systems

### 12.4 GPU responsibilities

The GPU service accepts compiled kernels for regular search tasks. It must include a cost model that accounts for transfer and compilation overhead.

GPU execution pipeline:

1. Identify an independent bounded search region.
2. Lower the verifier slice into a restricted GPU IR.
3. Compile or retrieve a cached kernel.
4. Run candidates in large batches.
5. Return matches or partial reductions.
6. Validate matches on the CPU.

### 12.5 FPGA option

An FPGA can outperform GPUs for a fixed repeated verifier, but compilation is slow and development is expensive. It belongs in a later distributed-worker tier, not the MVP.

Potential FPGA workloads:

- Fixed hash pipelines
- Stream ciphers
- Repeated CRC or checksum circuits
- Stable bitwise verifiers

### 12.6 Distributed workers

Workers advertise capabilities:

```json
{
  "cpu_cores": 32,
  "ram_gb": 128,
  "gpu": "supported",
  "gpu_vram_gb": 24,
  "solvers": ["z3", "cvc5", "sage", "fplll"],
  "architectures": ["x86_64", "aarch64"]
}
```

The scheduler sends self-contained, signed jobs. Workers run inside isolated containers or microVMs and return facts, models, logs, and validation artifacts.

---

## 13. Runtime and Language Choices

### 13.1 Core runtime

Recommended split:

- Rust for orchestration, UCIR, scheduling, caching, sandbox coordination, and backend adapters
- Python for user scripting, notebooks, challenge plugins, and SageMath interoperability
- C or C++ only where required by existing solver APIs
- CUDA, HIP, or a portable GPU layer for accelerator kernels

Rust is suitable for the core because the runtime handles untrusted challenge files, concurrent jobs, and complex resource ownership.

### 13.2 Process isolation

Backends should run out of process by default. Benefits:

- A crashing solver does not crash the orchestrator.
- Memory limits can be enforced.
- Different dependency stacks remain isolated.
- Solvers can be upgraded independently.

Communication can use a compact binary protocol over local sockets.

---

## 14. Plugin System

Plugins add domain recognizers, transformations, attacks, and validators.

Plugin categories:

- Frontend
- Recognizer
- Simplifier
- Solver adapter
- Attack strategy
- Search kernel
- Validator
- Reporter

A plugin declares:

- Supported UCIR operations
- Required dependencies
- Expected input features
- Soundness level
- Estimated cost
- Hardware requirements
- Output fact types

Example attack plugin metadata:

```yaml
name: rsa-small-private-exponent
category: attack-strategy
recognizes:
  - rsa-public-key
requires:
  - continued-fractions
soundness: exact
hardware: cpu
outputs:
  - private-exponent
  - plaintext-candidate
```

---

## 15. User Experience

### 15.1 Command line

```bash
atlas solve checker.bin --target success --input-length 32
atlas solve challenge.py --strategy auto
atlas inspect constraints.json
atlas benchmark corpus/
```

### 15.2 Python API

```python
from atlasctf import Project, BitVec

project = Project()
flag = project.bytes("flag", 32)
project.require_printable(flag)
project.load_checker("checker.bin", stdin=flag)

result = project.solve(strategy="auto", timeout=120)
print(result.model[flag])
print(result.explanation)
```

### 15.3 Notebook integration

The notebook view should display:

- Extracted constraints
- Variable domains
- Recognized structures
- Active solver strategies
- Progress and resource usage
- Partial assignments
- Final replay trace

---

## 16. Caching

Caching prevents repeated work during iterative challenge solving.

Cache keys include:

- Canonical UCIR hash
- Solver version
- Tactic configuration
- Architecture semantics
- Plugin version
- Relevant assumptions

Cached artifacts:

- Simplified expressions
- Lifted functions
- Function summaries
- SAT clauses
- Polynomial factorizations
- Lattice bases
- Compiled GPU kernels
- Validated models

Unsound heuristic results must not be cached as proofs.

---

## 17. Resource Management

Each task receives budgets for:

- Wall-clock time
- CPU time
- Memory
- GPU memory
- Process count
- Output size
- Path count

The planner uses staged budgets:

1. Cheap simplification and recognizers
2. Fast specialized solvers
3. Parallel general solvers
4. Expensive algebra or lattice strategies
5. Bounded accelerator search

This prevents a single poor strategy from consuming the entire run.

---

## 18. Correctness and Explainability

A solver result is not considered complete until it passes validation.

Result levels:

- `PROVEN_UNSAT`: a trusted backend proves no solution under the model
- `VALIDATED_SAT`: a candidate satisfies UCIR and the original checker
- `MODEL_ONLY`: a backend produced a model, but original validation is unavailable
- `PARTIAL`: useful deductions were found without a complete answer
- `UNKNOWN`: resource limit or unsupported theory

The report includes:

- Assumptions
- Transformations
- Solvers invoked
- Facts exchanged
- Candidate source
- Validation method
- Reproduction command

---

## 19. Safety and Isolation

CTF artifacts can contain hostile code. The platform must assume every binary and script is untrusted.

Controls:

- Run challenge code in a container or microVM
- Disable network access by default
- Mount the challenge directory read-only
- Use temporary writable storage
- Apply CPU, memory, process, and file-size limits
- Block privileged syscalls
- Do not expose host secrets or environment variables
- Record all executed commands
- Require explicit authorization for remote targets

The default workflow should focus on local files and competition-provided endpoints.

---

## 20. Failure Handling

Common failures and responses:

- Backend crash: isolate, record, retry with another backend
- Out of memory: cancel task, preserve partial facts, repartition problem
- Timeout: return partial deductions and next recommended strategies
- Translation mismatch: compare backend model against UCIR and original checker
- Unsupported instruction: concretize, emulate, summarize, or report exact location
- Conflicting facts: retain provenance and run a fact-validation pass
- Worker disconnect: requeue idempotent job

---

## 21. Testing Strategy

### 21.1 Unit tests

- UCIR typing and semantics
- Signed and unsigned comparisons
- Overflow behavior
- Endianness
- Simplification equivalence
- Serialization
- Fact provenance

### 21.2 Differential tests

Generate random small problems and compare:

- UCIR evaluator against native execution
- Multiple SMT backends against each other
- GPU verifier against CPU verifier
- Lifted instructions against an emulator

### 21.3 Property tests

Examples:

- Simplification preserves satisfying assignments.
- Serialization round trips preserve expression identity.
- A validated candidate always satisfies original constraints.
- Heuristic facts never become mandatory without validation.

### 21.4 Benchmark corpus

Maintain authorized challenges grouped by technique:

- Bit-vector verifier
- XOR-linear system
- Modular linear algebra
- RSA weakness
- Polynomial system
- Lattice attack
- Symbolic path exploration
- GPU-suitable bounded search

Metrics:

- Time to first valid result
- Peak memory
- Number of solver calls
- Constraints removed by preprocessing
- Fact-sharing benefit
- Validation success rate

---

## 22. Performance Strategy

The performance priority order should be:

1. Better modeling
2. Smaller problem slices
3. Domain-specific simplification
4. Correct specialized algorithm
5. Parallel solver portfolio
6. Incremental solving and caching
7. SIMD and GPU acceleration
8. Distributed execution

This order matters. Hardware cannot compensate for a badly modeled problem.

### 22.1 Portfolio scheduling

Use historical benchmark data to assign short initial budgets. Extend only strategies showing progress, such as:

- Reduced variable count
- New bounds
- New conflicts
- Better lattice vectors
- Increased path coverage

### 22.2 Solver-specific optimization

- Reuse incremental solver contexts.
- Partition independent constraint components.
- Preserve XOR clauses for XOR-aware SAT engines.
- Delay bit-blasting until algebraic simplification finishes.
- Use assumptions instead of rebuilding solver states.
- Generate several lattice formulations concurrently.
- Compile repeated concrete checks into native code.

---

## 23. MVP Scope

The first useful release should not attempt every feature.

### Phase 1: Constraint orchestrator

- Rust UCIR core
- Python API
- Z3 adapter
- SageMath adapter
- Basic simplifier
- Shared fact store
- Concrete UCIR validator
- Portfolio scheduler

### Phase 2: Reversing support

- x86-64 lifting
- Symbolic registers and flat memory
- Taint-guided slicing
- Basic path exploration
- Concrete replay

### Phase 3: Specialized solving

- GF(2) linear solver
- Modular matrix solver
- Lattice adapter
- Crypto structure recognizers
- Solver benchmark database

### Phase 4: Acceleration

- Native compiled candidate checker
- CPU SIMD search
- GPU kernel backend
- Remote worker protocol

### Phase 5: Advanced automation

- Learned strategy ranking
- Function-summary library
- Expanded architectures
- Visual notebook debugger

---

## 24. Suggested Repository Layout

```text
atlasctf/
├── crates/
│   ├── ucir/
│   ├── orchestrator/
│   ├── scheduler/
│   ├── fact-store/
│   ├── sandbox/
│   ├── validator/
│   └── protocol/
├── python/
│   └── atlasctf/
├── backends/
│   ├── z3/
│   ├── cvc5/
│   ├── sage/
│   ├── sat/
│   ├── lattice/
│   └── gpu/
├── frontends/
│   ├── equations/
│   ├── smtlib/
│   ├── binary/
│   └── traces/
├── plugins/
│   ├── recognizers/
│   ├── attacks/
│   └── simplifiers/
├── benchmarks/
├── tests/
├── docs/
└── examples/
```

---

## 25. Example End-to-End Run

A checker accepts a 24-byte input and performs XORs, rotations, additions, and one modular polynomial check.

1. The binary frontend lifts only instructions influenced by stdin.
2. Taint slicing removes logging and initialization code.
3. UCIR records every operation as an 8- or 32-bit value.
4. The simplifier detects four independent byte groups.
5. GF(2) elimination solves XOR-only relationships.
6. The algebra backend factors the modular polynomial.
7. The remaining bit-vector constraints are sent to Z3 and Bitwuzla.
8. Both engines receive bounds discovered by the algebra backend.
9. A candidate is returned.
10. The checker runs in a network-disabled sandbox with that candidate.
11. AtlasCTF emits the input, a replay command, and an explanation of each solved component.

---

## 26. Key Design Decisions

- Build an orchestrator, not a new universal solver.
- Use a typed common IR with exact machine and mathematical semantics.
- Prefer transparent rule-based planning for the MVP.
- Exchange facts between engines instead of running isolated races only.
- Validate every final model against the original challenge.
- Use GPUs only for regular bounded workloads.
- Keep untrusted binaries isolated from the host.
- Optimize preprocessing and problem decomposition before adding exotic hardware.

---

## 27. Success Criteria

The design is successful when the MVP can:

- Express both modular mathematics and fixed-width program constraints without semantic loss.
- Solve a benchmark suite using at least Z3 and SageMath through one API.
- Automatically choose a reasonable initial strategy for common challenge classes.
- Share at least bounds and assignments between backends.
- Validate every reported solution concretely.
- Demonstrate measurable speedups from preprocessing or portfolio execution on representative authorized CTF tasks.
- Produce a reproducible explanation rather than only printing a candidate flag.

---

## 28. Recommended First Prototype

The smallest convincing prototype is:

1. A Python-facing expression API.
2. A Rust or Python UCIR implementation.
3. Z3 and SageMath adapters.
4. A simplifier for constants, bounds, XOR-linear relations, and modular equations.
5. A planner with approximately ten explicit routing rules.
6. Parallel execution with cancellation.
7. A validator that re-evaluates the original UCIR.
8. A benchmark containing 20 to 30 small, authorized challenge patterns.

This prototype tests the core hypothesis: coordinated specialization and fact sharing can outperform using either tool alone, without requiring a new theorem prover or expensive custom hardware.
