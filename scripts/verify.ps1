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
