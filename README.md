# AtlasCTF

AtlasCTF is a from-scratch CTF math, symbolic-solving, and hardware-accelerated bounded-search workspace. It is built around auditable Rust crates, explicit benchmark evidence, and local-first tooling for reversing, crypto, serial/keygen, and GPU-search experiments.

## What is included

- Typed UCIR and program-analysis crates for deterministic reasoning over challenge-like inputs.
- Native and SIMD bounded-search engines for bit-vector, checksum, rotate/XOR, modular arithmetic, and serial-byte constraints.
- From-scratch math routines for common CTF crypto tasks such as CRT, modular exponentiation, finite-field linear solving, modular square roots, discrete logs, LFSR recovery, and exact rational arithmetic.
- Native CTF crypto utilities for hex/base64, repeating-key XOR, Caesar shifts,
  PKCS#7 validation, and SHA-256 known-vector workflows. This is an auditable
  CTF toolkit, not a claim to implement every production cryptosystem or to
  replace audited cryptographic libraries.
- Hardware acceleration through explicit OpenCL, Vulkan, WGPU, CUDA, and HIP adapter boundaries.
- Release validation, CTF benchmark manifests, and reproducible evidence under `benchmarks/`.

## Quick start

Install the CLI from GitHub:

```bash
curl -fsSL https://raw.githubusercontent.com/imattas/atlas/main/install.sh | sh
```

Installers use the latest GitHub Release tag and verified platform binary by
default, then fall back to a locked Cargo source build when that asset is not
available. Set `ATLAS_BINARY=off` to force Cargo or `ATLAS_BINARY=always` to
fail instead of falling back.

On Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/imattas/atlas/main/install.ps1 | iex
```

Install optional GPU adapter binaries too:

```bash
curl -fsSL https://raw.githubusercontent.com/imattas/atlas/main/install.sh | ATLAS_INSTALL_GPU=1 sh
```

On Windows PowerShell:

```powershell
$env:ATLAS_INSTALL_GPU='1'; irm https://raw.githubusercontent.com/imattas/atlas/main/install.ps1 | iex
```

```powershell
cargo test --workspace --all-targets
cargo run -p atlas-cli -- solve --fixture xor --start 0 --end 256
cargo run -p atlas-cli -- benchmark --fixture xor --start 0 --end 100000 --samples 3
```

For a hardware check on a machine with supported GPU drivers:

```powershell
.\scripts\verify.ps1 -Profile hardware
```

On Unix-like runners:

```bash
./scripts/verify.sh --profile hardware
```

## Hardware acceleration

Atlas keeps GPU execution behind SDK-specific adapter binaries instead of pretending CPU fallback is hardware acceleration. Runtime reports include `DeviceValidated` only when the adapter ran successfully and reported concrete hardware identity.

Supported adapter families:

- OpenCL
- Vulkan
- WGPU
- CUDA
- HIP

CUDA requires an NVIDIA driver/runtime on the host; unavailable backends are reported and skipped by capability probes.

## Benchmarks

CTF-relevant benchmark evidence lives in:

- `benchmarks/ctf/manifest.toml`
- `benchmarks/results/ctf-benchmarks.json`
- `benchmarks/results/ctf-benchmarks.md`
- `benchmarks/results/external-comparison.json`
- `benchmarks/results/external-comparison.md`

Run the benchmark refresh with:

```powershell
python benchmarks\compare_external.py --write
```

The suite covers XOR constraints, checksum residue search, rotate/XOR checks, modular multiply/add checks, serial-byte constraints, modular square roots, and finite-field discrete logs. Optional Z3 and Sage rows are included when those tools are installed; Atlas itself does not depend on a Z3 backend for its native math/search path.

## Release process

This repository uses GitHub Releases rather than a checked-in `release/` directory. Release metadata is tracked in:

- `RELEASE_MANIFEST.toml`
- `schemas/release-manifest.schema.json`
- `.github/workflows/release.yml`

Create a release by pushing a tag like `v1.0.0-rc.1`. The workflow validates
release metadata, runs release tests, builds x86_64 Windows/Linux/macOS CLI
binaries plus a source archive, writes checksums, and publishes the GitHub
Release.

You can also run the release workflow manually with a version input such as
`v0.1.0`; it will create/update the tag and publish the GitHub Release.

## Repository layout

The repo is a multi-track workspace, so the top-level folders separate Rust
crates, GPU kernels, Python helpers, benchmark evidence, schemas, docs, and
release tests. See `docs/repository-layout.md` for the current map and cleanup
direction. Local `target/` and `release/` folders are ignored build/scratch
output, not GitHub source folders.

The v1 contracts are documented in `docs/cli-contract.md`,
`docs/support-matrix.md`, and `docs/compatibility.md`.

## Verification profiles

```powershell
.\scripts\verify.ps1 -Profile core
.\scripts\verify.ps1 -Profile analysis
.\scripts\verify.ps1 -Profile distributed
.\scripts\verify.ps1 -Profile advanced
.\scripts\verify.ps1 -Profile full
```

Use `hardware` for real-device GPU validation when the host has supported SDKs and drivers.
