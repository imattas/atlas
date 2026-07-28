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

if (Test-Path "python/tests") {
    Invoke-Step "pytest python/tests" { python -m pytest python/tests }
}
if (Test-Path "backends") {
    Invoke-Step "pytest backends" { python -m pytest backends }
}

Write-Host "Verification profile '$Profile' passed."
