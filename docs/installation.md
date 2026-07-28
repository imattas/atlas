# AtlasCTF installation

AtlasCTF is a Rust workspace with Python helper packages. A reproducible local
installation requires:

1. Rust toolchain compatible with workspace `rust-version`.
2. Python 3 with stdlib `unittest` and `tomllib`.
3. Optional Z3 Python bindings for full SMT solving through the Z3 adapter.
4. Optional SageMath CLI for broad algebra, number theory, and polynomial-ring
   workflows through the Sage adapter.
5. Optional GPU SDK/runtime. Prefer OpenCL or Vulkan for portable GPU compute;
   use CUDA/HIP for vendor-specific deployments. The checked tests validate GPU
   code generation and SDK planning without requiring hardware.

Recommended validation:

```powershell
.\scripts\verify.ps1 -Profile full
```

The CLI entry point is `atlas`. Python helpers are importable from `python/` and
`notebook/atlas_widget/python/` during development.
