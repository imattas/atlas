param(
    [ValidateSet("core", "analysis", "distributed", "advanced", "full", "hardware")]
    [string]$Profile = "core"
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param([string]$Name, [scriptblock]$Command)
    Write-Host "==> $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Invoke-Step "cargo fmt" { cargo fmt --all -- --check }
Invoke-Step "cargo clippy" { cargo clippy --workspace --all-targets -- -D warnings }
Invoke-Step "cargo test" { cargo test --workspace --all-targets }

if ($Profile -in @("analysis", "distributed", "advanced", "full")) {
    foreach ($RequiredPath in @(
        "tests/e2e/track2/manifest.toml",
        "benchmarks/track2/manifest.toml",
        "docs/guides/reversing.md",
        "plugins/strategies/gf2/manifest.toml",
        "plugins/strategies/modular-matrix/manifest.toml",
        "plugins/strategies/lattice/manifest.toml",
        "plugins/strategies/crypto-recognizers/manifest.toml"
    )) {
        if (!(Test-Path $RequiredPath)) {
            Write-Error "missing analysis release artifact: $RequiredPath"
            exit 1
        }
    }
}

$HardwareFailures = @()
function Invoke-HardwareStep {
    param([string]$Name, [scriptblock]$Command)
    Write-Host "==> $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        $script:HardwareFailures += "$Name exited with $LASTEXITCODE"
    }
}

if ($Profile -in @("distributed", "advanced", "full")) {
    foreach ($RequiredPath in @(
        "tests/e2e/track3/manifest.toml",
        "benchmarks/track3/manifest.toml",
        "benchmarks/track3/calibration.toml",
        "docs/guides/workers.md",
        "deploy/worker/README.md",
        "crates/atlas-gpu-opencl-adapter/src/lib.rs",
        "crates/atlas-gpu-opencl-adapter/src/main.rs",
        "crates/atlas-gpu-cuda-adapter/src/lib.rs",
        "crates/atlas-gpu-cuda-adapter/src/main.rs",
        "crates/atlas-gpu-hip-adapter/src/lib.rs",
        "crates/atlas-gpu-hip-adapter/src/main.rs",
        "crates/atlas-gpu-vulkan-adapter/src/lib.rs",
        "crates/atlas-gpu-vulkan-adapter/src/main.rs",
        "gpu/cuda/atlas_search.cu",
        "gpu/hip/atlas_search.hip",
        "gpu/opencl/atlas_search.cl",
        "gpu/vulkan/atlas_search.comp"
    )) {
        if (!(Test-Path $RequiredPath)) {
            Write-Error "missing distributed release artifact: $RequiredPath"
            exit 1
        }
    }
}

if ($Profile -in @("advanced", "full")) {
    foreach ($RequiredPath in @(
        "notebook/atlas_widget/python/atlas_widget/__init__.py",
        "notebook/atlas_widget/tests/test_event_store.py",
        "notebook/atlas_widget/src/README.md",
        "tests/fixtures/events/track1_stream.toml",
        "tests/e2e/track4/manifest.toml"
    )) {
        if (!(Test-Path $RequiredPath)) {
            Write-Error "missing advanced release artifact: $RequiredPath"
            exit 1
        }
    }
}

if ($Profile -eq "full") {
    foreach ($RequiredPath in @(
        "release/manifest.schema.json",
        "release/manifest.toml",
        "release/write-manifest.sh",
        "release/write_manifest.py",
        "crates/atlas-math/src/lib.rs",
        "backends/native-math/atlas_native_math_backend.py",
        "docs/installation.md",
        "docs/security.md",
        "docs/plugins.md",
        "docs/architecture.md",
        "docs/hardware-acceleration.md",
        "tests/release/test_manifest.py"
    )) {
        if (!(Test-Path $RequiredPath)) {
            Write-Error "missing full release artifact: $RequiredPath"
            exit 1
        }
    }
    Invoke-Step "release manifest validation" { python release/write_manifest.py --validate release/manifest.toml }
    Invoke-Step "release manifest tests" { python -m unittest discover tests/release }
}

if ($Profile -eq "hardware") {
    Invoke-HardwareStep "GPU doctor diagnostics" { cargo run -q -p atlas-cli -- doctor }
    Invoke-HardwareStep "OpenCL real-device search" {
        cargo test -p atlas-gpu-opencl-adapter --test adapter generated_opencl_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
    }
    Invoke-HardwareStep "CUDA real-device search" {
        cargo test -p atlas-gpu-cuda-adapter --test adapter generated_cuda_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
    }
    Invoke-HardwareStep "HIP real-device search" {
        cargo test -p atlas-gpu-hip-adapter --test adapter generated_hip_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
    }
    Invoke-HardwareStep "Vulkan real-device search" {
        cargo test -p atlas-gpu-vulkan-adapter --test adapter generated_vulkan_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
    }
    Invoke-HardwareStep "Vulkan shaderInt64 real-device search" {
        cargo test -p atlas-gpu-vulkan-adapter --test adapter generated_vulkan_64_bit_kernel_runs_on_device -- --ignored --nocapture
    }
    if ($HardwareFailures.Count -ne 0) {
        Write-Error "Hardware verification failed after attempting every backend: $($HardwareFailures -join '; ')"
        exit 1
    }
}

if (Test-Path "python/tests") {
    $pytestAvailable = $false
    try {
        python -m pytest --version *> $null
        $pytestAvailable = ($LASTEXITCODE -eq 0)
    } catch {
        $pytestAvailable = $false
    }
    if ($pytestAvailable) {
        Invoke-Step "pytest python/tests" { python -m pytest python/tests }
    } else {
        Invoke-Step "unittest python/tests" { python -m unittest discover python/tests }
    }
}
if (Test-Path "notebook/atlas_widget/tests") {
    Invoke-Step "unittest notebook widget" { python -m unittest discover notebook/atlas_widget/tests }
}
if (Test-Path "backends") {
    $pytestAvailable = $false
    try {
        python -m pytest --version *> $null
        $pytestAvailable = ($LASTEXITCODE -eq 0)
    } catch {
        $pytestAvailable = $false
    }
    if ($pytestAvailable) {
        Invoke-Step "pytest backends" { python -m pytest backends }
    } else {
        Invoke-Step "unittest backends" { python -m unittest discover backends/tests }
    }
}

Write-Host "Verification profile '$Profile' passed."
