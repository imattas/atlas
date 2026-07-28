# AtlasCTF hardware acceleration

Atlas supports hardware acceleration behind an explicit SDK boundary. GPU
execution never bypasses CPU/IR validation.

Recommended SDK order:

1. OpenCL via the Khronos OpenCL SDK for portable heterogeneous compute.
2. Vulkan compute via the LunarG Vulkan SDK when shader-based compute is the
   best available cross-vendor path.
3. CUDA for NVIDIA-specific deployments.
4. HIP for AMD-specific deployments when available.

Runtime behavior:

- detected SDKs are ranked by `GpuSdkPlan`;
- host SDK discovery can scan `PATH` for adapter/compiler/runtime tools and
  route through the selected process-backed driver;
- missing SDKs produce a precise diagnostic and scalar/SIMD fallback;
- CUDA, HIP, OpenCL, and Vulkan compute sources are generated from the same
  restricted IR boundary;
- OpenCL code generation lowers the restricted search operations into device
  predicates for candidate filtering and bounded atomic match output;
- CUDA/HIP code generation lowers the same restricted search operations into
  device predicates with full-candidate output for CPU validation;
- Vulkan GLSL code generation lowers the same restricted search operations into
  64-bit shader predicates with bounded atomic match output;
- per-SDK command plans select checked-in kernel artifacts, `hipcc` where a HIP
  code object is needed, and launcher frontends
  (`atlas-gpu-opencl-run`, `atlas-gpu-vulkan-run`, `atlas-gpu-cuda-run`,
  `atlas-gpu-hip-run`);
- the OpenCL adapter uses dynamic OpenCL loading, build-checks generated source,
  launches the generated `atlas_search` kernel, and prints `match=<candidate>`
  lines for CPU validation by the runtime;
- the CUDA adapter uses NVRTC for in-process CUDA-source-to-PTX compilation,
  falls back to SDK-discovered `nvcc -ptx` when NVRTC is unavailable, uses
  dynamic CUDA Driver API loading, module validation, kernel launch, and
  `match=<candidate>` output for CPU validation by the runtime;
- the HIP adapter uses dynamic HIP runtime loading, validates generated code
  objects, launches the `atlas_search` kernel, and prints `match=<candidate>`
  lines for CPU validation by the runtime;
- the Vulkan adapter uses shaderc for in-process GLSL-to-SPIR-V compilation,
  dynamic Vulkan loading, shader module validation, compute dispatch, and
  `match=<candidate>` output for CPU validation by the runtime;
- production driver execution is isolated behind a runner boundary so host
  adapters can compile/launch kernels without changing search semantics;
- process-backed execution writes generated per-program kernel source into the
  build output directory before invoking the selected compiler or adapter
  compile-check;
- HIP compilation uses `hipcc --genco -O2` so the adapter receives a loadable
  `.hsaco` code object instead of a host executable;
- CUDA SDK discovery checks `CUDA_PATH`, `CUDA_HOME`, `CUDA_ROOT`, standard
  Toolkit install roots, SDK `bin` directories for `nvcc`, and dynamic-library
  locations for NVRTC/CUDA driver loading;
- launch configuration records global size, local size, output cap, and transfer
  bytes;
- accelerator placement can load Track 3 calibration thresholds for SIMD and
  GPU break-even decisions;
- generated kernels are cache-keyed by program, compiler, device, and options;
- every GPU-reported match is revalidated against CPU IR semantics;
- runtime telemetry records whether the result came from CPU fallback or from a
  CPU-validated device buffer.

Checked packaging fixtures:

- `gpu/cuda/atlas_search.cu`
- `gpu/hip/atlas_search.hip`
- `gpu/opencl/atlas_search.cl`
- `gpu/vulkan/atlas_search.comp`

Checked adapter artifacts:

- `crates/atlas-gpu-opencl-adapter/src/lib.rs`
- `crates/atlas-gpu-opencl-adapter/src/main.rs`
- `crates/atlas-gpu-cuda-adapter/src/lib.rs`
- `crates/atlas-gpu-cuda-adapter/src/main.rs`
- `crates/atlas-gpu-hip-adapter/src/lib.rs`
- `crates/atlas-gpu-hip-adapter/src/main.rs`
- `crates/atlas-gpu-vulkan-adapter/src/lib.rs`
- `crates/atlas-gpu-vulkan-adapter/src/main.rs`

Real-device validation commands:

- OpenCL:
  `cargo test -p atlas-gpu-opencl-adapter --test adapter generated_opencl_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture`
- CUDA:
  `cargo test -p atlas-gpu-cuda-adapter --test adapter generated_cuda_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture`
- HIP:
  `cargo test -p atlas-gpu-hip-adapter --test adapter generated_hip_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture`
- Vulkan:
  `cargo test -p atlas-gpu-vulkan-adapter --test adapter generated_vulkan_kernel_runs_on_device_and_preserves_full_candidates -- --ignored --nocapture`

Primary GPU references checked while implementing this boundary:

- Khronos OpenCL SDK/resources.
- LunarG/Khronos Vulkan SDK compute documentation.
