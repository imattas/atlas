# Stable CLI contract (v1)

The `atlas` executable writes one machine-readable JSON record to stdout for
`solve`, `inspect`, `benchmark`, `worker`, and `doctor`. Human-readable help is
the only normal non-JSON output. Diagnostics and errors go to stderr, with a
non-zero exit status.

Commands:

| Command | Purpose | Stable record kind |
| --- | --- | --- |
| `atlas solve` | Run a bounded fixture search | `solve` report with `schema_major: 1` |
| `atlas inspect` | Inspect an input/fixture envelope | `inspect` |
| `atlas benchmark` | Compare native, SIMD, and accelerator execution | `benchmark` |
| `atlas worker` | Exercise the worker protocol surface | `worker` |
| `atlas doctor` | Report SDK, adapter, and feature probes | `doctor` |

Common solve/benchmark flags are `--fixture`, `--start`, `--end`, `--samples`,
`--force-gpu`, and `--gpu-sdk`. `--format json` is accepted explicitly and is
the default. Unknown formats fail rather than silently changing output.

Use `atlas help` or `atlas --help` for the concise command list. JSON records
are line-delimited when a future command emits more than one record; current
commands emit exactly one record per invocation.
