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
- missing SDKs produce a precise diagnostic and scalar/SIMD fallback;
- CUDA, HIP, OpenCL, and Vulkan compute sources are generated from the same
  restricted IR boundary;
- OpenCL code generation lowers the restricted search operations into device
  predicates for candidate filtering and bounded atomic match output;
- per-SDK command plans select checked-in kernel artifacts and compiler
  frontends (`opencl-clang`, `glslc`, `nvcc`, or `hipcc`);
- production driver execution is isolated behind a runner boundary so host
  adapters can compile/launch kernels without changing search semantics;
- process-backed execution writes generated per-program kernel source into the
  build output directory before invoking the selected SDK compiler;
- launch configuration records global size, local size, output cap, and transfer
  bytes;
- generated kernels are cache-keyed by program, compiler, device, and options;
- every GPU-reported match is revalidated against CPU IR semantics;
- runtime telemetry records whether the result came from CPU fallback or from a
  CPU-validated device buffer.

Checked packaging fixtures:

- `gpu/cuda/atlas_search.cu`
- `gpu/hip/atlas_search.hip`
- `gpu/opencl/atlas_search.cl`
- `gpu/vulkan/atlas_search.comp`

Primary GPU references checked while implementing this boundary:

- Khronos OpenCL SDK/resources.
- LunarG/Khronos Vulkan SDK compute documentation.
