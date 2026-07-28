extern "C" __global__ void atlas_search(unsigned long long start, unsigned long long end) {
  // Generated kernels use the same restricted IR as CPU search. This static
  // fixture exists for hardware-independent compile-path and packaging checks.
  (void)start;
  (void)end;
}
