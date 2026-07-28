param(
    [ValidateSet("core", "analysis", "distributed", "advanced", "full")]
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

if ($Profile -in @("distributed", "advanced", "full")) {
    foreach ($RequiredPath in @(
        "tests/e2e/track3/manifest.toml",
        "benchmarks/track3/manifest.toml",
        "benchmarks/track3/calibration.toml",
        "docs/guides/workers.md",
        "deploy/worker/README.md",
        "gpu/cuda/atlas_search.cu"
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
