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

resolve_cargo_command() {
  if command -v cargo >/dev/null 2>&1 && cargo --version >/dev/null 2>&1; then
    printf "%s\n" cargo
  elif command -v cargo.exe >/dev/null 2>&1 && cargo.exe --version >/dev/null 2>&1; then
    printf "%s\n" cargo.exe
  else
    echo "cargo command not found or not executable; install Rust or add cargo to PATH" >&2
    return 127
  fi
}

cargo_cmd=$(resolve_cargo_command)
benchmark_samples=3

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

skip_hardware_step() {
  local name="$1"
  local reason="$2"
  echo "==> $name"
  echo "skipped: $reason"
}

gpu_feature_probe_ok() {
  local doctor_json="$1"
  local name="$2"
  DOCTOR_JSON="$doctor_json" GPU_NAME="$name" python - <<'PY'
import json
import os
import sys

document = json.loads(os.environ["DOCTOR_JSON"])
name = os.environ["GPU_NAME"]
for probe in document.get("gpu_feature_probes", []):
    if probe.get("name") == name:
        raise SystemExit(0 if probe.get("ok") else 1)
raise SystemExit(1)
PY
}

gpu_feature_probe_has_feature() {
  local doctor_json="$1"
  local name="$2"
  local feature="$3"
  DOCTOR_JSON="$doctor_json" GPU_NAME="$name" GPU_FEATURE="$feature" python - <<'PY'
import json
import os
import sys

document = json.loads(os.environ["DOCTOR_JSON"])
name = os.environ["GPU_NAME"]
feature = os.environ["GPU_FEATURE"]
for probe in document.get("gpu_feature_probes", []):
    if probe.get("name") == name:
        raise SystemExit(0 if probe.get("ok") and feature in probe.get("features", []) else 1)
raise SystemExit(1)
PY
}

gpu_any_feature_probe_has_int64() {
  gpu_feature_probe_has_feature "$1" OpenCL int64 \
    || gpu_feature_probe_has_feature "$1" Vulkan shaderInt64 \
    || gpu_feature_probe_has_feature "$1" CUDA int64 \
    || gpu_feature_probe_has_feature "$1" HIP int64
}

assert_gpu_feature_probes_have_launch_abi() {
  local doctor_json="$1"
  DOCTOR_JSON="$doctor_json" python - <<'PY'
import json
import os

document = json.loads(os.environ["DOCTOR_JSON"])
for probe in document.get("gpu_feature_probes", []):
    if not probe.get("ok"):
        continue
    features = set(probe.get("features", []))
    for required in ["launchAbiU32", "launchAbiU64"]:
        if required not in features:
            raise SystemExit(f"GPU feature probe {probe.get('name')} missing {required}")
PY
}

run_forced_gpu_benchmark() {
  local name="$1"
  local sdk="$2"
  local expected_actual_gpu_sdk="$3"
  local fixture="${4:-xor}"
  local start="${5:-0x50}"
  local end="${6:-0x60}"
  run_hardware_step "$name" bash -c '
    set -euo pipefail
    command=("$3" run -q -p atlas-cli -- benchmark --fixture "$4" --start "$5" --end "$6" --force-gpu --samples "$7")
    if [[ -n "$1" ]]; then
      command+=(--gpu-sdk "$1")
    fi
    output=$("${command[@]}")
    printf "%s\n" "$output"
    EXPECTED_ACTUAL_GPU_SDK="$2" BENCHMARK_JSON="$output" BENCHMARK_SAMPLES="$7" python - <<'"'"'PY'"'"'
import json
import os

expected_actual_gpu_sdk = os.environ["EXPECTED_ACTUAL_GPU_SDK"]
benchmark_samples = int(os.environ["BENCHMARK_SAMPLES"])
document = json.loads(os.environ["BENCHMARK_JSON"])
accelerator = document["accelerator"]
if accelerator["mode"] != "DeviceValidated":
    raise SystemExit(f"expected DeviceValidated, got {accelerator['mode']}")
if document.get("sample_count") != benchmark_samples:
    raise SystemExit(f"expected sample_count {benchmark_samples}, got {document.get('sample_count')}")
actual_gpu_sdk = accelerator.get("actual_gpu_sdk")
if expected_actual_gpu_sdk and actual_gpu_sdk != expected_actual_gpu_sdk:
    raise SystemExit(f"expected actual_gpu_sdk {expected_actual_gpu_sdk}, got {actual_gpu_sdk}")
telemetry = accelerator.get("telemetry") or ""
for required in ["driver exit 0", "driver launches", "launch abi"]:
    if required not in telemetry:
        raise SystemExit(f"expected benchmark telemetry to include {required!r}, got {telemetry!r}")
PY
  ' _ "$sdk" "$expected_actual_gpu_sdk" "$cargo_cmd" "$fixture" "$start" "$end" "$benchmark_samples"
}

"$cargo_cmd" fmt --all -- --check
"$cargo_cmd" clippy --workspace --all-targets -- -D warnings
"$cargo_cmd" test --workspace --all-targets

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
    scripts/verify_hardware_doctor.py \
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
  echo "==> GPU doctor diagnostics"
  set +e
  hardware_doctor_json=$("$cargo_cmd" run -q -p atlas-cli -- doctor)
  status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    hardware_failures+=("GPU doctor diagnostics exited with $status")
    hardware_doctor_json='{"gpu_feature_probes":[]}'
  else
    printf "%s\n" "$hardware_doctor_json"
    printf "%s\n" "$hardware_doctor_json" | python scripts/verify_hardware_doctor.py --require-launch-abi
  fi
  run_forced_gpu_benchmark "Forced-GPU benchmark" "" ""
  if gpu_any_feature_probe_has_int64 "$hardware_doctor_json"; then
    run_forced_gpu_benchmark "Forced-GPU int64 benchmark" "" "" xor64 0x8000000000000000 0x8000000000000002
  else
    skip_hardware_step "Forced-GPU int64 benchmark" "No GPU int64 feature probe available"
  fi
  if gpu_feature_probe_ok "$hardware_doctor_json" OpenCL; then
    run_forced_gpu_benchmark "Forced-GPU OpenCL benchmark" opencl OpenCL
    run_hardware_step "OpenCL real-device search" "$cargo_cmd" test -p atlas-gpu-opencl-adapter --test adapter generated_opencl_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
    if gpu_feature_probe_has_feature "$hardware_doctor_json" OpenCL int64; then
      run_forced_gpu_benchmark "Forced-GPU OpenCL int64 benchmark" opencl OpenCL xor64 0x8000000000000000 0x8000000000000002
      run_hardware_step "OpenCL int64 real-device search" "$cargo_cmd" test -p atlas-gpu-opencl-adapter --test adapter generated_opencl_64_bit_kernel_runs_on_device -- --ignored --nocapture
    else
      skip_hardware_step "Forced-GPU OpenCL int64 benchmark" "OpenCL int64 feature unavailable"
      skip_hardware_step "OpenCL int64 real-device search" "OpenCL int64 feature unavailable"
    fi
  else
    skip_hardware_step "Forced-GPU OpenCL benchmark" "OpenCL runtime feature probe unavailable"
    skip_hardware_step "Forced-GPU OpenCL int64 benchmark" "OpenCL runtime feature probe unavailable"
    skip_hardware_step "OpenCL real-device search" "OpenCL runtime feature probe unavailable"
    skip_hardware_step "OpenCL int64 real-device search" "OpenCL runtime feature probe unavailable"
  fi
  if gpu_feature_probe_ok "$hardware_doctor_json" Vulkan; then
    run_forced_gpu_benchmark "Forced-GPU Vulkan benchmark" vulkan Vulkan
    run_hardware_step "Vulkan real-device search" "$cargo_cmd" test -p atlas-gpu-vulkan-adapter --test adapter generated_vulkan_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
    if gpu_feature_probe_has_feature "$hardware_doctor_json" Vulkan shaderInt64; then
      run_forced_gpu_benchmark "Forced-GPU Vulkan int64 benchmark" vulkan Vulkan xor64 0x8000000000000000 0x8000000000000002
      run_hardware_step "Vulkan shaderInt64 real-device search" "$cargo_cmd" test -p atlas-gpu-vulkan-adapter --test adapter generated_vulkan_64_bit_kernel_runs_on_device -- --ignored --nocapture
    else
      skip_hardware_step "Forced-GPU Vulkan int64 benchmark" "Vulkan shaderInt64 feature unavailable"
      skip_hardware_step "Vulkan shaderInt64 real-device search" "Vulkan shaderInt64 feature unavailable"
    fi
  else
    skip_hardware_step "Forced-GPU Vulkan benchmark" "Vulkan runtime feature probe unavailable"
    skip_hardware_step "Forced-GPU Vulkan int64 benchmark" "Vulkan runtime feature probe unavailable"
    skip_hardware_step "Vulkan real-device search" "Vulkan runtime feature probe unavailable"
    skip_hardware_step "Vulkan shaderInt64 real-device search" "Vulkan runtime feature probe unavailable"
  fi
  if gpu_feature_probe_ok "$hardware_doctor_json" CUDA; then
    run_forced_gpu_benchmark "Forced-GPU CUDA benchmark" cuda CUDA
    run_hardware_step "CUDA real-device search" "$cargo_cmd" test -p atlas-gpu-cuda-adapter --test adapter generated_cuda_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
    if gpu_feature_probe_has_feature "$hardware_doctor_json" CUDA int64; then
      run_forced_gpu_benchmark "Forced-GPU CUDA int64 benchmark" cuda CUDA xor64 0x8000000000000000 0x8000000000000002
      run_hardware_step "CUDA int64 real-device search" "$cargo_cmd" test -p atlas-gpu-cuda-adapter --test adapter generated_cuda_64_bit_kernel_runs_on_device -- --ignored --nocapture
    else
      skip_hardware_step "Forced-GPU CUDA int64 benchmark" "CUDA int64 feature unavailable"
      skip_hardware_step "CUDA int64 real-device search" "CUDA int64 feature unavailable"
    fi
  else
    skip_hardware_step "Forced-GPU CUDA benchmark" "CUDA runtime feature probe unavailable"
    skip_hardware_step "Forced-GPU CUDA int64 benchmark" "CUDA runtime feature probe unavailable"
    skip_hardware_step "CUDA real-device search" "CUDA runtime feature probe unavailable"
    skip_hardware_step "CUDA int64 real-device search" "CUDA runtime feature probe unavailable"
  fi
  if gpu_feature_probe_ok "$hardware_doctor_json" HIP; then
    run_forced_gpu_benchmark "Forced-GPU HIP benchmark" hip HIP
    run_hardware_step "HIP real-device search" "$cargo_cmd" test -p atlas-gpu-hip-adapter --test adapter generated_hip_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture
    if gpu_feature_probe_has_feature "$hardware_doctor_json" HIP int64; then
      run_forced_gpu_benchmark "Forced-GPU HIP int64 benchmark" hip HIP xor64 0x8000000000000000 0x8000000000000002
      run_hardware_step "HIP int64 real-device search" "$cargo_cmd" test -p atlas-gpu-hip-adapter --test adapter generated_hip_64_bit_kernel_runs_on_device -- --ignored --nocapture
    else
      skip_hardware_step "Forced-GPU HIP int64 benchmark" "HIP int64 feature unavailable"
      skip_hardware_step "HIP int64 real-device search" "HIP int64 feature unavailable"
    fi
  else
    skip_hardware_step "Forced-GPU HIP benchmark" "HIP runtime feature probe unavailable"
    skip_hardware_step "Forced-GPU HIP int64 benchmark" "HIP runtime feature probe unavailable"
    skip_hardware_step "HIP real-device search" "HIP runtime feature probe unavailable"
    skip_hardware_step "HIP int64 real-device search" "HIP runtime feature probe unavailable"
  fi
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
