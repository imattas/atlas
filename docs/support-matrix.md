# CPU/GPU support matrix

Support describes what Atlas can build and probe. A backend is only reported
as device-validated when its adapter successfully launches and returns hardware
identity; a compiled kernel alone is not hardware evidence.

| Platform | CPU/native | SIMD | WGPU | OpenCL | Vulkan | CUDA | HIP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Linux x86_64 | Supported | Supported | Supported when driver exists | Supported when ICD exists | Supported when loader/driver exists | NVIDIA driver required | ROCm required |
| macOS x86_64/arm64 | Supported | Supported | Supported when Metal backend is available | Host ICD required | Host loader/driver required | Not supported by NVIDIA on modern macOS | Not supported |
| Windows x86_64 | Supported | Supported | Supported when driver exists | Supported when ICD exists | Supported when loader/driver exists | NVIDIA driver/toolkit required | ROCm/AMD support required |

Release binaries contain the CPU CLI. GPU adapter binaries are optional and
are useful only when the matching SDK and driver are installed on the host.
The default runtime remains safe CPU execution when no accelerator is usable.
