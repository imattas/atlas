__kernel void atlas_search(
    ulong start,
    ulong end,
    __global ulong* out,
    __global uint* out_len
) {
    // Generated kernels use the same restricted IR as CPU search. This static
    // fixture exists for hardware-independent compile-path and packaging checks.
    size_t gid = get_global_id(0);
    (void)gid;
    (void)start;
    (void)end;
    (void)out;
    (void)out_len;
}
