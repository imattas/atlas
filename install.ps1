[CmdletBinding()]
param(
    [switch]$Help,
    [switch]$InstallGpu,
    [switch]$DryRun,
    [string]$Repo = $(if ($env:ATLAS_REPO) { $env:ATLAS_REPO } else { "imattas/atlas" }),
    [string]$RepoUrl = $env:ATLAS_REPO_URL,
    [string]$Branch = $(if ($env:ATLAS_BRANCH) { $env:ATLAS_BRANCH } else { "main" }),
    [string]$Tag = $env:ATLAS_TAG,
    [string]$Rev = $env:ATLAS_REV,
    [string]$Release = $(if ($env:ATLAS_RELEASE) { $env:ATLAS_RELEASE } else { "latest" }),
    [string]$Root = $env:ATLAS_ROOT,
    [string]$CargoArgs = $env:ATLAS_CARGO_ARGS
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $RepoUrl) {
    $RepoUrl = "https://github.com/$Repo.git"
}

if (-not $InstallGpu -and $env:ATLAS_INSTALL_GPU -match '^(1|true|yes|on)$') {
    $InstallGpu = $true
}

function Show-Usage {
    @"
AtlasCTF Windows installer

Usage:
  irm https://raw.githubusercontent.com/imattas/atlas/main/install.ps1 | iex

Options:
  -InstallGpu      Also install GPU adapter binaries.
  -DryRun          Print cargo commands without executing them.

Environment:
  ATLAS_BRANCH=main       Git branch to install when ATLAS_TAG/ATLAS_REV are unset.
  ATLAS_TAG=v0.1.0        Git tag to install.
  ATLAS_REV=<sha>         Git revision to install.
  ATLAS_RELEASE=latest    Install latest GitHub Release tag by default. Use "off" for ATLAS_BRANCH.
  ATLAS_INSTALL_GPU=1     Also install GPU adapter binaries.
  ATLAS_ROOT=$HOME\.cargo Cargo install root. Defaults to Cargo's install root.
  ATLAS_CARGO_ARGS="..."  Extra arguments appended to each cargo install command.
"@
}

if ($Help) {
    Show-Usage
    exit 0
}

function Resolve-Cargo {
    $cargo = Get-Command cargo.exe -ErrorAction SilentlyContinue
    if (-not $cargo) {
        $cargo = Get-Command cargo -ErrorAction SilentlyContinue
    }
    if (-not $cargo) {
        throw "missing required command: cargo; install Rust from https://rustup.rs/ or add Cargo to PATH"
    }
    return $cargo.Source
}

function Resolve-ReleaseTag {
    if ($Tag -or $Rev -or $Release -eq "off") {
        return
    }
    if ($Release -ne "latest") {
        $script:Tag = $Release
        return
    }
    try {
        $apiUrl = "https://api.github.com/repos/$Repo/releases/latest"
        $latest = Invoke-RestMethod -Uri $apiUrl -UseBasicParsing
        if ($latest.tag_name) {
            $script:Tag = [string]$latest.tag_name
            Write-Host "==> Installing latest GitHub Release $script:Tag"
        }
    }
    catch {
        Write-Warning "latest GitHub Release not found; falling back to branch $Branch"
    }
}

function Get-RefArgs {
    if ($Tag) {
        return @("--tag", $Tag)
    }
    if ($Rev) {
        return @("--rev", $Rev)
    }
    return @("--branch", $Branch)
}

function Split-ExtraCargoArgs([string]$Text) {
    if (-not $Text) {
        return @()
    }
    return $Text -split '\s+' | Where-Object { $_ }
}

function Format-Command([string]$Exe, [string[]]$ArgList) {
    $quoted = @($Exe) + ($ArgList | ForEach-Object {
        if ($_ -match '\s') {
            '"' + ($_ -replace '"', '\"') + '"'
        }
        else {
            $_
        }
    })
    return ($quoted -join " ")
}

function Install-Package([string]$Package, [string]$Bin) {
    Write-Host "==> Installing $Bin from $RepoUrl"

    $cargoInstallArgs = @("install", "--git", $RepoUrl, "-p", $Package, "--bin", $Bin, "--locked", "--force")
    if ($Root) {
        $cargoInstallArgs += @("--root", $Root)
    }
    $cargoInstallArgs += Get-RefArgs
    $cargoInstallArgs += Split-ExtraCargoArgs $CargoArgs

    if ($DryRun) {
        Write-Host (Format-Command $script:CargoCommand $cargoInstallArgs)
        return
    }

    & $script:CargoCommand @cargoInstallArgs
}

$script:CargoCommand = Resolve-Cargo
Resolve-ReleaseTag

Install-Package "atlas-cli" "atlas"

if ($InstallGpu) {
    Install-Package "atlas-gpu-opencl-adapter" "atlas-gpu-opencl-run"
    Install-Package "atlas-gpu-vulkan-adapter" "atlas-gpu-vulkan-run"
    Install-Package "atlas-gpu-wgpu-adapter" "atlas-gpu-wgpu-run"
    Install-Package "atlas-gpu-cuda-adapter" "atlas-gpu-cuda-run"
    Install-Package "atlas-gpu-hip-adapter" "atlas-gpu-hip-run"
}

@"
==> AtlasCTF install complete
Run:
  atlas --help

GPU adapters are optional. To install them:
  `$env:ATLAS_INSTALL_GPU='1'; irm https://raw.githubusercontent.com/imattas/atlas/main/install.ps1 | iex
"@
