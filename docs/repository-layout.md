# Repository layout

AtlasCTF is a multi-crate workspace with a deliberately small set of canonical
top-level boundaries. New production Rust code belongs under `crates/`; new
user-facing documentation under `docs/`; challenge-independent benchmark
evidence under `benchmarks/`; and automation under `scripts/` or `.github/`.
Do not add another top-level folder for a single feature.

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

## Layout rules

- Root files are limited to workspace manifests, release metadata, and primary
  project documentation.
- `crates/` is the only production Rust source boundary.
- `gpu/` contains kernels only; host adapters remain in `crates/`.
- `backends/`, `plugins/`, `frontends/`, and `models/` are integration or
  experiment inputs, not alternate production source trees.
- `target/`, `release/`, and benchmark scratch output are ignored local state.

This formalizes the existing paths without a risky mass move. Any future move
should be a separate mechanical change with Cargo, workflow, manifest, and
installer validation in the same change.
