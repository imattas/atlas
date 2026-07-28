# AtlasCTF installation

AtlasCTF is a Rust workspace with Python helper packages. A reproducible local
installation requires:

1. Rust toolchain compatible with workspace `rust-version`.
2. Python 3 with stdlib `unittest` and `tomllib`.
3. Native Atlas math is built from source in `crates/atlas-math`; no Z3 or
   Sage runtime is required.
4. Optional GPU SDK/runtime. Prefer OpenCL or Vulkan for portable GPU compute;
   use CUDA/HIP for vendor-specific deployments. The checked tests validate GPU
   code generation and SDK planning without requiring hardware.

Install the CLI directly from GitHub:

```bash
curl -fsSL https://raw.githubusercontent.com/imattas/atlas/main/install.sh | sh
```

On Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/imattas/atlas/main/install.ps1 | iex
```

Install GPU adapter binaries as well:

```bash
curl -fsSL https://raw.githubusercontent.com/imattas/atlas/main/install.sh | ATLAS_INSTALL_GPU=1 sh
```

On Windows PowerShell:

```powershell
$env:ATLAS_INSTALL_GPU='1'; irm https://raw.githubusercontent.com/imattas/atlas/main/install.ps1 | iex
```

Version/ref overrides are supported:

```bash
curl -fsSL https://raw.githubusercontent.com/imattas/atlas/main/install.sh | ATLAS_TAG=v0.1.0 sh
curl -fsSL https://raw.githubusercontent.com/imattas/atlas/main/install.sh | ATLAS_REV=<commit-sha> sh
```

On Windows PowerShell:

```powershell
$env:ATLAS_TAG='v0.1.0'; irm https://raw.githubusercontent.com/imattas/atlas/main/install.ps1 | iex
$env:ATLAS_REV='<commit-sha>'; irm https://raw.githubusercontent.com/imattas/atlas/main/install.ps1 | iex
```

Recommended validation:

```powershell
.\scripts\verify.ps1 -Profile full
```

The CLI entry point is `atlas`. Python helpers are importable from `python/` and
`notebook/atlas_widget/python/` during development.
