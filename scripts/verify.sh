#!/usr/bin/env bash
set -euo pipefail

profile="core"
if [[ "${1:-}" == "--profile" ]]; then
  profile="${2:-core}"
fi

case "$profile" in
  core|analysis|distributed|advanced|full|hardware) ;;
  *)
    echo "unknown profile: $profile" >&2
    exit 2
    ;;
esac

hardware_failures=()
run_hardware_step() {
  local name="$1"
  shift
  echo "==> $name"
  set +e
  "$@"
  local status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    hardware_failures+=("$name exited with $status")
  fi
}

cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets

if [[ "$profile" == "analysis" || "$profile" == "distributed" || "$profile" == "advanced" || "$profile" == "full" ]]; then
  for required_path in \
    tests/e2e/track2/manifest.toml \
    benchmarks/track2/manifest.toml \
    docs/guides/reversing.md \
    plugins/strategies/gf2/manifest.toml \
    plugins/strategies/modular-matrix/manifest.toml \
    plugins/strategies/lattice/manifest.toml \
    plugins/strategies/crypto-recognizers/manifest.toml
  do
    if [[ ! -e "$required_path" ]]; then
      echo "missing analysis release artifact: $required_path" >&2
      exit 1
    fi
  done
fi

if [[ "$profile" == "distributed" || "$profile" == "advanced" || "$profile" == "full" ]]; then
  for required_path in \
    tests/e2e/track3/manifest.toml \
    benchmarks/track3/manifest.toml \
    benchmarks/track3/calibration.toml \
    docs/guides/workers.md \
    deploy/worker/README.md \
    crates/atlas-gpu-opencl-adapter/src/lib.rs \
    crates/atlas-gpu-opencl-adapter/src/main.rs \
    crates/atlas-gpu-cuda-adapter/src/lib.rs \
    crates/atlas-gpu-cuda-adapter/src/main.rs \
    crates/atlas-gpu-hip-adapter/src/lib.rs \
    crates/atlas-gpu-hip-adapter/src/main.rs \
    crates/atlas-gpu-vulkan-adapter/src/lib.rs \
    crates/atlas-gpu-vulkan-adapter/src/main.rs \
    gpu/cuda/atlas_search.cu \
    gpu/hip/atlas_search.hip \
    gpu/opencl/atlas_search.cl \
    gpu/vulkan/atlas_search.comp
  do
    if [[ ! -e "$required_path" ]]; then
      echo "missing distributed release artifact: $required_path" >&2
      exit 1
    fi
  done
fi

if [[ "$profile" == "advanced" || "$profile" == "full" ]]; then
  for required_path in \
    notebook/atlas_widget/python/atlas_widget/__init__.py \
    notebook/atlas_widget/tests/test_event_store.py \
    notebook/atlas_widget/src/README.md \
    tests/fixtures/events/track1_stream.toml \
    tests/e2e/track4/manifest.toml
  do
    if [[ ! -e "$required_path" ]]; then
      echo "missing advanced release artifact: $required_path" >&2
      exit 1
    fi
  done
fi

if [[ "$profile" == "full" ]]; then
  for required_path in \
    release/manifest.schema.json \
    release/manifest.toml \
    release/write-manifest.sh \
    release/write_manifest.py \
    crates/atlas-math/src/lib.rs \
    backends/native-math/atlas_native_math_backend.py \
    docs/installation.md \
    docs/security.md \
    docs/plugins.md \
    docs/architecture.md \
    docs/hardware-acceleration.md \
    tests/release/test_manifest.py
  do
    if [[ ! -e "$required_path" ]]; then
      echo "missing full release artifact: $required_path" >&2
      exit 1
    fi
  done
  python release/write_manifest.py --validate release/manifest.toml
  python -m unittest discover tests/release
fi

if [[ "$profile" == "hardware" ]]; then
  run_hardware_step "GPU doctor diagnostics" cargo run -q -p atlas-cli -- doctor
  run_hardware_step "Forced-GPU benchmark" cargo run -q -p atlas-cli -- benchmark --fixture xor --start 0x50 --end 0x60 --force-gpu
  run_hardware_step "OpenCL real-device search" cargo test -p atlas-gpu-opencl-adapter --test adapter generated_opencl_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
  run_hardware_step "CUDA real-device search" cargo test -p atlas-gpu-cuda-adapter --test adapter generated_cuda_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
  run_hardware_step "HIP real-device search" cargo test -p atlas-gpu-hip-adapter --test adapter generated_hip_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
  run_hardware_step "Vulkan real-device search" cargo test -p atlas-gpu-vulkan-adapter --test adapter generated_vulkan_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
  run_hardware_step "Vulkan shaderInt64 real-device search" cargo test -p atlas-gpu-vulkan-adapter --test adapter generated_vulkan_64_bit_kernel_runs_on_device -- --ignored --nocapture
  if [[ "${#hardware_failures[@]}" -ne 0 ]]; then
    echo "Hardware verification failed after attempting every backend:" >&2
    printf '  - %s\n' "${hardware_failures[@]}" >&2
    exit 1
  fi
fi

if [[ -d python/tests ]]; then
  if python -m pytest --version >/dev/null 2>&1; then
    python -m pytest python/tests
  else
    python -m unittest discover python/tests
  fi
fi
if [[ -d notebook/atlas_widget/tests ]]; then
  python -m unittest discover notebook/atlas_widget/tests
fi
if [[ -d backends ]]; then
  if python -m pytest --version >/dev/null 2>&1; then
    python -m pytest backends
  else
    python -m unittest discover backends/tests
  fi
fi

echo "Verification profile '${profile}' passed."
