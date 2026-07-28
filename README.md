# AtlasCTF

AtlasCTF is a from-scratch CTF math, symbolic-solving, and hardware-accelerated bounded-search workspace. It is built around auditable Rust crates, explicit benchmark evidence, and local-first tooling for reversing, crypto, serial/keygen, and GPU-search experiments.

## What is included

- Typed UCIR and program-analysis crates for deterministic reasoning over challenge-like inputs.
- Native and SIMD bounded-search engines for bit-vector, checksum, rotate/XOR, modular arithmetic, and serial-byte constraints.
- From-scratch math routines for common CTF crypto tasks such as CRT, modular exponentiation, finite-field linear solving, modular square roots, discrete logs, LFSR recovery, and exact rational arithmetic.
- Hardware acceleration through explicit OpenCL, Vulkan, WGPU, CUDA, and HIP adapter boundaries.
- Release validation, CTF benchmark manifests, and reproducible evidence under `benchmarks/`.

## Quick start

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

Create a release by pushing a tag like `v0.1.0`. The workflow validates release metadata, runs release tests, builds a source archive, writes checksums, and publishes the GitHub Release.

## Verification profiles

```powershell
.\scripts\verify.ps1 -Profile core
.\scripts\verify.ps1 -Profile analysis
.\scripts\verify.ps1 -Profile distributed
.\scripts\verify.ps1 -Profile advanced
.\scripts\verify.ps1 -Profile full
```

Use `hardware` for real-device GPU validation when the host has supported SDKs and drivers.
