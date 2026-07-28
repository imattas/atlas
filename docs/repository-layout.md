# Repository layout

AtlasCTF is currently a multi-track workspace rather than a single CLI crate.
That is why the repository has several top-level folders: each one maps to a
separate part of the system or release evidence.

## Tracked top-level folders

- `.github/` — CI and GitHub Release automation.
- `crates/` — Rust workspace crates, including the CLI, math, search, GPU
  adapters, planner, executor, reports, and protocol crates.
- `python/`, `notebook/`, `backends/` — Python SDK/helper surfaces and backend
  integration tests.
- `gpu/` — CUDA, HIP, OpenCL, Vulkan, and WGPU kernel sources used by the GPU
  adapter crates.
- `benchmarks/` — CTF and backend benchmark manifests plus generated benchmark
  evidence.
- `frontends/` — source, binary, and trace frontend notes.
- `plugins/` — strategy manifests for GF(2), lattice, modular-matrix, and
  crypto recognizer experiments.
- `schemas/`, `summaries/` — protocol schemas and reusable analysis summaries.
- `tests/` — end-to-end fixtures and release validation tests.
- `docs/` — architecture, installation, hardware, security, and workflow docs.
- `scripts/` — verification, release-manifest, and benchmark helper scripts.
- `examples/`, `models/`, `deploy/` — examples, model card, and worker deploy
  notes.

## Ignored local-only folders

These may exist in a developer checkout but are not part of the GitHub source
tree:

- `target/` — Rust build output.
- `release/` — legacy/local release scratch output. Real releases are GitHub
  Releases.
- `.worktrees/`, `.venv/`, `dist/`, `build/`, `tmp/`, and cache folders.

## Cleanup direction

The current layout favors explicit subsystem boundaries over a compact tree. If
the repo needs to become easier to browse, the safest next cleanup is to group
documentation-only surfaces under `docs/` first, then migrate experimental
strategy/backend folders behind workspace-aware path updates. Moving `crates/`,
`gpu/`, `benchmarks/`, or `tests/` should be done as a separate mechanical PR
with CI coverage because those paths are referenced by Cargo manifests, release
metadata, tests, and benchmark manifests.
