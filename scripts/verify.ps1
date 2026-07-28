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
$BenchmarkSamples = 3
function Invoke-HardwareStep {
    param([string]$Name, [scriptblock]$Command)
    Write-Host "==> $Name"
    try {
        & $Command
        if ($LASTEXITCODE -ne 0) {
            $script:HardwareFailures += "$Name exited with $LASTEXITCODE"
        }
    } catch {
        $script:HardwareFailures += "$Name failed: $($_.Exception.Message)"
        $global:LASTEXITCODE = 0
    }
}

function Skip-HardwareStep {
    param([string]$Name, [string]$Reason)
    Write-Host "==> $Name"
    Write-Host "skipped: $Reason"
}

function Get-GpuFeatureProbeOk {
    param($Doctor, [string]$Name)
    if ($null -eq $Doctor -or $null -eq $Doctor.gpu_feature_probes) {
        return $false
    }
    foreach ($Probe in $Doctor.gpu_feature_probes) {
        if ($Probe.name -eq $Name) {
            return [bool]$Probe.ok
        }
    }
    return $false
}

function Get-GpuFeatureProbeHasFeature {
    param($Doctor, [string]$Name, [string]$Feature)
    if ($null -eq $Doctor -or $null -eq $Doctor.gpu_feature_probes) {
        return $false
    }
    foreach ($Probe in $Doctor.gpu_feature_probes) {
        if ($Probe.name -eq $Name) {
            return [bool]$Probe.ok -and $Probe.features -contains $Feature
        }
    }
    return $false
}

function Get-AnyGpuFeatureProbeHasInt64 {
    param($Doctor)
    return (Get-GpuFeatureProbeHasFeature $Doctor "OpenCL" "int64") `
        -or (Get-GpuFeatureProbeHasFeature $Doctor "Vulkan" "shaderInt64") `
        -or (Get-GpuFeatureProbeHasFeature $Doctor "CUDA" "int64") `
        -or (Get-GpuFeatureProbeHasFeature $Doctor "HIP" "int64")
}

function Assert-GpuFeatureProbeHasLaunchAbi {
    param($Doctor)
    if ($null -eq $Doctor -or $null -eq $Doctor.gpu_feature_probes) {
        throw "GPU doctor did not report feature probes"
    }
    foreach ($Probe in $Doctor.gpu_feature_probes) {
        if (![bool]$Probe.ok) {
            continue
        }
        foreach ($RequiredFeature in @("launchAbiU32", "launchAbiU64")) {
            if ($Probe.features -notcontains $RequiredFeature) {
                throw "GPU feature probe $($Probe.name) missing $RequiredFeature"
            }
        }
    }
}

function Invoke-ForcedGpuBenchmark {
    param(
        [string]$Name,
        [string]$Sdk,
        [string]$ExpectedActualGpuSdk,
        [string]$Fixture = "xor",
        [string]$Start = "0x50",
        [string]$End = "0x60",
        [int]$MinRetainedMatches = 0
    )
    Invoke-HardwareStep $Name {
        $BenchmarkArgs = @("run", "-q", "-p", "atlas-cli", "--", "benchmark", "--fixture", $Fixture, "--start", $Start, "--end", $End, "--force-gpu", "--samples", $BenchmarkSamples)
        if (![string]::IsNullOrEmpty($Sdk)) {
            $BenchmarkArgs += @("--gpu-sdk", $Sdk)
        }
        $Output = cargo @BenchmarkArgs
        $Status = $LASTEXITCODE
        if ($Status -ne 0) {
            $global:LASTEXITCODE = $Status
            return
        }
        Write-Host $Output
        $Benchmark = $Output | ConvertFrom-Json
        if ($Benchmark.accelerator.mode -ne "DeviceValidated") {
            throw "expected DeviceValidated, got $($Benchmark.accelerator.mode)"
        }
        if ($Benchmark.sample_count -ne $BenchmarkSamples) {
            throw "expected sample_count $BenchmarkSamples, got $($Benchmark.sample_count)"
        }
        if (![string]::IsNullOrEmpty($ExpectedActualGpuSdk) -and $Benchmark.accelerator.actual_gpu_sdk -ne $ExpectedActualGpuSdk) {
            throw "expected actual_gpu_sdk $ExpectedActualGpuSdk, got $($Benchmark.accelerator.actual_gpu_sdk)"
        }
        if ($MinRetainedMatches -gt 0) {
            if ($Benchmark.accelerator.launch.max_matches -lt $MinRetainedMatches) {
                throw "expected benchmark max_matches at least $MinRetainedMatches, got $($Benchmark.accelerator.launch.max_matches)"
            }
            $MinOutputBufferBytes = $MinRetainedMatches * 8
            if ($Benchmark.accelerator.launch.output_buffer_bytes -lt $MinOutputBufferBytes) {
                throw "expected benchmark output_buffer_bytes at least $MinOutputBufferBytes, got $($Benchmark.accelerator.launch.output_buffer_bytes)"
            }
            $ExpectedCanonicalMatches = $MinRetainedMatches
            if (@($Benchmark.accelerator.matches).Count -lt $ExpectedCanonicalMatches) {
                throw "expected at least $ExpectedCanonicalMatches returned matches, got $(@($Benchmark.accelerator.matches).Count)"
            }
        }
        $Telemetry = [string]$Benchmark.accelerator.telemetry
        foreach ($RequiredTelemetry in @("driver exit 0", "driver launches", "launch abi")) {
            if (!$Telemetry.Contains($RequiredTelemetry)) {
                throw "expected benchmark telemetry to include '$RequiredTelemetry', got '$Telemetry'"
            }
        }
        $global:LASTEXITCODE = 0
    }
}

function Invoke-PlacementSelectedGpuBenchmark {
    param([string]$Name)
    Invoke-HardwareStep $Name {
        $BenchmarkArgs = @("run", "-q", "-p", "atlas-cli", "--", "benchmark", "--fixture", "xor", "--start", "0", "--end", "1000000", "--samples", $BenchmarkSamples)
        $Output = cargo @BenchmarkArgs
        $Status = $LASTEXITCODE
        if ($Status -ne 0) {
            $global:LASTEXITCODE = $Status
            return
        }
        Write-Host $Output
        $Benchmark = $Output | ConvertFrom-Json
        if ($Benchmark.accelerator.requested_gpu_sdk -ne $null) {
            throw "expected placement-selected benchmark to omit requested_gpu_sdk, got $($Benchmark.accelerator.requested_gpu_sdk)"
        }
        if ($Benchmark.accelerator.mode -ne "DeviceValidated") {
            throw "expected DeviceValidated placement-selected GPU benchmark, got $($Benchmark.accelerator.mode)"
        }
        if ($Benchmark.sample_count -ne $BenchmarkSamples) {
            throw "expected sample_count $BenchmarkSamples, got $($Benchmark.sample_count)"
        }
        if ([string]::IsNullOrEmpty([string]$Benchmark.accelerator.actual_gpu_sdk)) {
            throw "expected placement-selected benchmark to report actual_gpu_sdk"
        }
        if ($Benchmark.accelerator.launch.global_size -lt 1000000) {
            throw "expected placement-selected benchmark global_size to cover 1000000 candidates, got $($Benchmark.accelerator.launch.global_size)"
        }
        $Telemetry = [string]$Benchmark.accelerator.telemetry
        foreach ($RequiredTelemetry in @("driver exit 0", "driver launches", "launch abi")) {
            if (!$Telemetry.Contains($RequiredTelemetry)) {
                throw "expected placement-selected benchmark telemetry to include '$RequiredTelemetry', got '$Telemetry'"
            }
        }
        $global:LASTEXITCODE = 0
    }
}

function Invoke-WarmCachePlacementGpuBenchmark {
    param([string]$WarmName, [string]$AutoName)
    Invoke-HardwareStep $WarmName {
        $BenchmarkArgs = @("run", "-q", "-p", "atlas-cli", "--", "benchmark", "--fixture", "xor", "--start", "0", "--end", "100000", "--force-gpu", "--samples", $BenchmarkSamples)
        $Output = cargo @BenchmarkArgs
        $Status = $LASTEXITCODE
        if ($Status -ne 0) {
            $global:LASTEXITCODE = $Status
            return
        }
        Write-Host $Output
        $Benchmark = $Output | ConvertFrom-Json
        if ($Benchmark.accelerator.mode -ne "DeviceValidated") {
            throw "expected DeviceValidated warm-cache GPU benchmark, got $($Benchmark.accelerator.mode)"
        }
        if ($Benchmark.sample_count -ne $BenchmarkSamples) {
            throw "expected sample_count $BenchmarkSamples, got $($Benchmark.sample_count)"
        }
        if ([string]::IsNullOrEmpty([string]$Benchmark.accelerator.actual_gpu_sdk)) {
            throw "expected warm-cache benchmark to report actual_gpu_sdk"
        }
        $Telemetry = [string]$Benchmark.accelerator.telemetry
        foreach ($RequiredTelemetry in @("driver exit 0", "driver launches", "launch abi")) {
            if (!$Telemetry.Contains($RequiredTelemetry)) {
                throw "expected warm-cache benchmark telemetry to include '$RequiredTelemetry', got '$Telemetry'"
            }
        }
        $global:LASTEXITCODE = 0
    }
    Invoke-HardwareStep $AutoName {
        $BenchmarkArgs = @("run", "-q", "-p", "atlas-cli", "--", "benchmark", "--fixture", "xor", "--start", "0", "--end", "100000", "--samples", $BenchmarkSamples)
        $Output = cargo @BenchmarkArgs
        $Status = $LASTEXITCODE
        if ($Status -ne 0) {
            $global:LASTEXITCODE = $Status
            return
        }
        Write-Host $Output
        $Benchmark = $Output | ConvertFrom-Json
        if ($Benchmark.accelerator.requested_gpu_sdk -ne $null) {
            throw "expected warm-cache auto-placement benchmark to omit requested_gpu_sdk, got $($Benchmark.accelerator.requested_gpu_sdk)"
        }
        if ($Benchmark.accelerator.mode -ne "DeviceValidated") {
            throw "expected DeviceValidated warm-cache auto-placement GPU benchmark, got $($Benchmark.accelerator.mode)"
        }
        if ($Benchmark.sample_count -ne $BenchmarkSamples) {
            throw "expected sample_count $BenchmarkSamples, got $($Benchmark.sample_count)"
        }
        if ([string]::IsNullOrEmpty([string]$Benchmark.accelerator.actual_gpu_sdk)) {
            throw "expected warm-cache auto-placement benchmark to report actual_gpu_sdk"
        }
        if ($Benchmark.accelerator.launch.global_size -lt 100000) {
            throw "expected warm-cache auto-placement benchmark global_size to cover 100000 candidates, got $($Benchmark.accelerator.launch.global_size)"
        }
        $Telemetry = [string]$Benchmark.accelerator.telemetry
        foreach ($RequiredTelemetry in @("driver exit 0", "driver launches", "launch abi")) {
            if (!$Telemetry.Contains($RequiredTelemetry)) {
                throw "expected warm-cache auto-placement telemetry to include '$RequiredTelemetry', got '$Telemetry'"
            }
        }
        $global:LASTEXITCODE = 0
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
        "scripts/verify_hardware_doctor.py",
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
    $HardwareDoctor = $null
    Invoke-HardwareStep "GPU doctor diagnostics" {
        $Output = cargo run -q -p atlas-cli -- doctor
        $Status = $LASTEXITCODE
        if ($Status -ne 0) {
            $global:LASTEXITCODE = $Status
            return
        }
        Write-Host $Output
        $script:HardwareDoctor = $Output | ConvertFrom-Json
        $DoctorJsonFile = New-TemporaryFile
        try {
            Set-Content -LiteralPath $DoctorJsonFile -Value $Output -NoNewline -Encoding utf8
            python scripts/verify_hardware_doctor.py --input $DoctorJsonFile --require-launch-abi
            if ($LASTEXITCODE -ne 0) {
                throw "GPU doctor diagnostics failed hardware validation"
            }
        } finally {
            Remove-Item -LiteralPath $DoctorJsonFile -Force -ErrorAction SilentlyContinue
        }
        $global:LASTEXITCODE = 0
    }
    Invoke-PlacementSelectedGpuBenchmark "Placement-selected GPU benchmark"
    Invoke-ForcedGpuBenchmark "Forced-GPU benchmark" $null $null
    Invoke-ForcedGpuBenchmark "Forced-GPU dense benchmark" $null $null "dense" "0" "1500" 1500
    if (Get-AnyGpuFeatureProbeHasInt64 $HardwareDoctor) {
        Invoke-ForcedGpuBenchmark "Forced-GPU int64 benchmark" $null $null "xor64" "0x8000000000000000" "0x8000000000000002"
    } else {
        Skip-HardwareStep "Forced-GPU int64 benchmark" "No GPU int64 feature probe available"
    }
    Invoke-WarmCachePlacementGpuBenchmark "Warm-cache placement GPU benchmark" "Warm-cache auto-placement GPU benchmark"
    if (Get-GpuFeatureProbeOk $HardwareDoctor "OpenCL") {
        Invoke-ForcedGpuBenchmark "Forced-GPU OpenCL benchmark" "opencl" "OpenCL"
        Invoke-ForcedGpuBenchmark "Forced-GPU OpenCL dense benchmark" "opencl" "OpenCL" "dense" "0" "1500" 1500
        Invoke-HardwareStep "OpenCL real-device search" {
            cargo test -p atlas-gpu-opencl-adapter --test adapter generated_opencl_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
        }
        if (Get-GpuFeatureProbeHasFeature $HardwareDoctor "OpenCL" "int64") {
            Invoke-ForcedGpuBenchmark "Forced-GPU OpenCL int64 benchmark" "opencl" "OpenCL" "xor64" "0x8000000000000000" "0x8000000000000002"
            Invoke-HardwareStep "OpenCL int64 real-device search" {
                cargo test -p atlas-gpu-opencl-adapter --test adapter generated_opencl_64_bit_kernel_runs_on_device -- --ignored --nocapture
            }
        } else {
            Skip-HardwareStep "Forced-GPU OpenCL int64 benchmark" "OpenCL int64 feature unavailable"
            Skip-HardwareStep "OpenCL int64 real-device search" "OpenCL int64 feature unavailable"
        }
    } else {
        Skip-HardwareStep "Forced-GPU OpenCL benchmark" "OpenCL runtime feature probe unavailable"
        Skip-HardwareStep "Forced-GPU OpenCL dense benchmark" "OpenCL runtime feature probe unavailable"
        Skip-HardwareStep "Forced-GPU OpenCL int64 benchmark" "OpenCL runtime feature probe unavailable"
        Skip-HardwareStep "OpenCL real-device search" "OpenCL runtime feature probe unavailable"
        Skip-HardwareStep "OpenCL int64 real-device search" "OpenCL runtime feature probe unavailable"
    }
    if (Get-GpuFeatureProbeOk $HardwareDoctor "Vulkan") {
        Invoke-ForcedGpuBenchmark "Forced-GPU Vulkan benchmark" "vulkan" "Vulkan"
        Invoke-ForcedGpuBenchmark "Forced-GPU Vulkan dense benchmark" "vulkan" "Vulkan" "dense" "0" "1500" 1500
        Invoke-HardwareStep "Vulkan real-device search" {
            cargo test -p atlas-gpu-vulkan-adapter --test adapter generated_vulkan_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
        }
        if (Get-GpuFeatureProbeHasFeature $HardwareDoctor "Vulkan" "shaderInt64") {
            Invoke-ForcedGpuBenchmark "Forced-GPU Vulkan int64 benchmark" "vulkan" "Vulkan" "xor64" "0x8000000000000000" "0x8000000000000002"
            Invoke-HardwareStep "Vulkan shaderInt64 real-device search" {
                cargo test -p atlas-gpu-vulkan-adapter --test adapter generated_vulkan_64_bit_kernel_runs_on_device -- --ignored --nocapture
            }
        } else {
            Skip-HardwareStep "Forced-GPU Vulkan int64 benchmark" "Vulkan shaderInt64 feature unavailable"
            Skip-HardwareStep "Vulkan shaderInt64 real-device search" "Vulkan shaderInt64 feature unavailable"
        }
    } else {
        Skip-HardwareStep "Forced-GPU Vulkan benchmark" "Vulkan runtime feature probe unavailable"
        Skip-HardwareStep "Forced-GPU Vulkan dense benchmark" "Vulkan runtime feature probe unavailable"
        Skip-HardwareStep "Forced-GPU Vulkan int64 benchmark" "Vulkan runtime feature probe unavailable"
        Skip-HardwareStep "Vulkan real-device search" "Vulkan runtime feature probe unavailable"
        Skip-HardwareStep "Vulkan shaderInt64 real-device search" "Vulkan runtime feature probe unavailable"
    }
    if (Get-GpuFeatureProbeOk $HardwareDoctor "CUDA") {
        Invoke-ForcedGpuBenchmark "Forced-GPU CUDA benchmark" "cuda" "CUDA"
        Invoke-ForcedGpuBenchmark "Forced-GPU CUDA dense benchmark" "cuda" "CUDA" "dense" "0" "1500" 1500
        Invoke-HardwareStep "CUDA real-device search" {
            cargo test -p atlas-gpu-cuda-adapter --test adapter generated_cuda_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
        }
        if (Get-GpuFeatureProbeHasFeature $HardwareDoctor "CUDA" "int64") {
            Invoke-ForcedGpuBenchmark "Forced-GPU CUDA int64 benchmark" "cuda" "CUDA" "xor64" "0x8000000000000000" "0x8000000000000002"
            Invoke-HardwareStep "CUDA int64 real-device search" {
                cargo test -p atlas-gpu-cuda-adapter --test adapter generated_cuda_64_bit_kernel_runs_on_device -- --ignored --nocapture
            }
        } else {
            Skip-HardwareStep "Forced-GPU CUDA int64 benchmark" "CUDA int64 feature unavailable"
            Skip-HardwareStep "CUDA int64 real-device search" "CUDA int64 feature unavailable"
        }
    } else {
        Skip-HardwareStep "Forced-GPU CUDA benchmark" "CUDA runtime feature probe unavailable"
        Skip-HardwareStep "Forced-GPU CUDA dense benchmark" "CUDA runtime feature probe unavailable"
        Skip-HardwareStep "Forced-GPU CUDA int64 benchmark" "CUDA runtime feature probe unavailable"
        Skip-HardwareStep "CUDA real-device search" "CUDA runtime feature probe unavailable"
        Skip-HardwareStep "CUDA int64 real-device search" "CUDA runtime feature probe unavailable"
    }
    if (Get-GpuFeatureProbeOk $HardwareDoctor "HIP") {
        Invoke-ForcedGpuBenchmark "Forced-GPU HIP benchmark" "hip" "HIP"
        Invoke-ForcedGpuBenchmark "Forced-GPU HIP dense benchmark" "hip" "HIP" "dense" "0" "1500" 1500
        Invoke-HardwareStep "HIP real-device search" {
            cargo test -p atlas-gpu-hip-adapter --test adapter generated_hip_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
        }
        if (Get-GpuFeatureProbeHasFeature $HardwareDoctor "HIP" "int64") {
            Invoke-ForcedGpuBenchmark "Forced-GPU HIP int64 benchmark" "hip" "HIP" "xor64" "0x8000000000000000" "0x8000000000000002"
            Invoke-HardwareStep "HIP int64 real-device search" {
                cargo test -p atlas-gpu-hip-adapter --test adapter generated_hip_64_bit_kernel_runs_on_device -- --ignored --nocapture
            }
        } else {
            Skip-HardwareStep "Forced-GPU HIP int64 benchmark" "HIP int64 feature unavailable"
            Skip-HardwareStep "HIP int64 real-device search" "HIP int64 feature unavailable"
        }
    } else {
        Skip-HardwareStep "Forced-GPU HIP benchmark" "HIP runtime feature probe unavailable"
        Skip-HardwareStep "Forced-GPU HIP dense benchmark" "HIP runtime feature probe unavailable"
        Skip-HardwareStep "Forced-GPU HIP int64 benchmark" "HIP runtime feature probe unavailable"
        Skip-HardwareStep "HIP real-device search" "HIP runtime feature probe unavailable"
        Skip-HardwareStep "HIP int64 real-device search" "HIP runtime feature probe unavailable"
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
