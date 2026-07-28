#!/bin/sh
set -eu

repo="${ATLAS_REPO:-imattas/atlas}"
repo_url="${ATLAS_REPO_URL:-https://github.com/${repo}.git}"
branch="${ATLAS_BRANCH:-main}"
tag="${ATLAS_TAG:-}"
rev="${ATLAS_REV:-}"
release="${ATLAS_RELEASE:-latest}"
install_gpu="${ATLAS_INSTALL_GPU:-0}"
install_root="${ATLAS_ROOT:-}"
extra_cargo_args="${ATLAS_CARGO_ARGS:-}"

append_path_if_dir() {
  if [ -d "$1" ]; then
    PATH="${PATH}:$1"
  fi
}

append_path_if_dir "${HOME:-}/.cargo/bin"
append_path_if_dir "/mnt/c/Users/${USER:-}/.cargo/bin"
append_path_if_dir "/mnt/c/Users/${USERNAME:-}/.cargo/bin"
append_path_if_dir "/c/Users/${USER:-}/.cargo/bin"
append_path_if_dir "/c/Users/${USERNAME:-}/.cargo/bin"
for cargo_dir in /mnt/c/Users/*/.cargo/bin /c/Users/*/.cargo/bin; do
  append_path_if_dir "$cargo_dir"
done
export PATH

usage() {
  cat <<'EOF'
AtlasCTF installer

Usage:
  curl -fsSL https://raw.githubusercontent.com/imattas/atlas/main/install.sh | sh

Environment:
  ATLAS_BRANCH=main       Git branch to install when ATLAS_TAG/ATLAS_REV are unset.
  ATLAS_TAG=v0.1.0        Git tag to install.
  ATLAS_REV=<sha>         Git revision to install.
  ATLAS_RELEASE=latest    Install latest GitHub Release tag by default. Use "off" for ATLAS_BRANCH.
  ATLAS_INSTALL_GPU=1     Also install GPU adapter binaries.
  ATLAS_ROOT=$HOME/.local Cargo install root. Defaults to Cargo's install root.
  ATLAS_CARGO_ARGS="..."  Extra arguments appended to each cargo install command.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    return 127
  fi
}

resolve_cargo() {
  if command -v cargo >/dev/null 2>&1 && cargo --version >/dev/null 2>&1; then
    printf '%s\n' cargo
  elif command -v cargo.exe >/dev/null 2>&1 && cargo.exe --version >/dev/null 2>&1; then
    printf '%s\n' cargo.exe
  else
    echo "missing required command: cargo; install Rust from https://rustup.rs/ or add Cargo to PATH" >&2
    return 127
  fi
}

resolve_release_tag() {
  if [ -n "$tag" ] || [ -n "$rev" ] || [ "$release" = "off" ]; then
    return 0
  fi
  if [ "$release" != "latest" ]; then
    tag="$release"
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl not found for latest release lookup; falling back to branch ${branch}" >&2
    return 0
  fi
  api_url="https://api.github.com/repos/${repo}/releases/latest"
  latest_tag=$(
    curl -fsSL "$api_url" 2>/dev/null \
      | sed -n 's/^[[:space:]]*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
      | head -n 1
  ) || latest_tag=""
  if [ -n "$latest_tag" ]; then
    tag="$latest_tag"
    echo "==> Installing latest GitHub Release ${tag}"
  else
    echo "latest GitHub Release not found; falling back to branch ${branch}" >&2
  fi
}

cargo_ref_args() {
  if [ -n "$tag" ]; then
    printf '%s\n%s\n' --tag "$tag"
  elif [ -n "$rev" ]; then
    printf '%s\n%s\n' --rev "$rev"
  else
    printf '%s\n%s\n' --branch "$branch"
  fi
}

install_package() {
  package="$1"
  bin="$2"
  echo "==> Installing ${bin} from ${repo_url}"

  set -- install --git "$repo_url" -p "$package" --bin "$bin" --locked --force
  if [ -n "$install_root" ]; then
    set -- "$@" --root "$install_root"
  fi
  ref_file="$(mktemp)"
  cargo_ref_args >"$ref_file"
  while IFS= read -r arg; do
    set -- "$@" "$arg"
  done <"$ref_file"
  rm -f "$ref_file"

  # shellcheck disable=SC2086
  "$cargo_cmd" "$@" $extra_cargo_args
}

require_command mktemp
cargo_cmd="$(resolve_cargo)"
resolve_release_tag

install_package atlas-cli atlas

case "$install_gpu" in
  1|true|yes|on)
    install_package atlas-gpu-opencl-adapter atlas-gpu-opencl-run
    install_package atlas-gpu-vulkan-adapter atlas-gpu-vulkan-run
    install_package atlas-gpu-wgpu-adapter atlas-gpu-wgpu-run
    install_package atlas-gpu-cuda-adapter atlas-gpu-cuda-run
    install_package atlas-gpu-hip-adapter atlas-gpu-hip-run
    ;;
  0|false|no|off)
    ;;
  *)
    echo "invalid ATLAS_INSTALL_GPU value: ${install_gpu}" >&2
    exit 2
    ;;
esac

cat <<'EOF'
==> AtlasCTF install complete
Run:
  atlas --help

GPU adapters are optional. To install them:
  curl -fsSL https://raw.githubusercontent.com/imattas/atlas/main/install.sh | ATLAS_INSTALL_GPU=1 sh
EOF
