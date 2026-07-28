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
- generated kernels are cache-keyed by program, compiler, device, and options;
- every GPU-reported match is revalidated against CPU IR semantics.

Primary references checked while implementing this boundary:

- Z3: official Z3Py API/tutorial and Z3 project documentation.
- SageMath: official Sage reference and command-line documentation.
- GPU compute: Khronos OpenCL SDK/resources and LunarG/Khronos Vulkan SDK
  compute documentation.
