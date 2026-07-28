# AtlasCTF Workers

Atlas workers execute signed, content-addressed bounded-search jobs for an
authenticated coordinator. A worker advertises only explicit capabilities. The
coordinator schedules jobs to the least-capable worker that satisfies the job,
which keeps specialized hardware available for workloads that need it.

Default job isolation requirements:

- non-root execution
- networking disabled
- read-only artifact mounts
- no host environment leakage
- no Docker socket exposure
- scoped, signed job and result envelopes
- lease expiry and cooperative cancellation
- duplicate result suppression

Accelerator results never bypass CPU validation. GPU execution may be absent on
a host; hardware-independent tests still compile the CUDA boundary and compare
GPU-fallback output against scalar native search.
