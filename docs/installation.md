# AtlasCTF installation

AtlasCTF is a Rust workspace with Python helper packages. A reproducible local
installation requires:

1. Rust toolchain compatible with workspace `rust-version`.
2. Python 3 with stdlib `unittest` and `tomllib`.
3. Optional CUDA tooling for compiling external GPU kernels; the checked tests
   validate GPU code generation without requiring hardware.

Recommended validation:

```powershell
.\scripts\verify.ps1 -Profile full
```

The CLI entry point is `atlas`. Python helpers are importable from `python/` and
`notebook/atlas_widget/python/` during development.
