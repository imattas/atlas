# Compatibility policy

AtlasCTF uses Semantic Versioning from `v1.0.0` onward.

The following are stable within a major version:

- The command names and documented flags in [the CLI contract](cli-contract.md).
- JSON records with `schema_major: 1`. New fields may be added; existing
  fields keep their meaning and types.
- Public Rust APIs in crates documented as stable in the support matrix.

The following are explicitly experimental and may change in a minor release:

- GPU kernel source and adapter launch ABIs.
- Undocumented Rust items, benchmark fixture names, and internal workspace
  crates.
- Optional external solver comparison rows. Atlas never requires Z3 or Sage as
  a runtime backend.

Breaking changes require a new `schema_major`, a migration note in this file,
and a major version. Release candidates can still change before `v1.0.0`.

Supported Rust and Python versions are the versions exercised by CI and listed
in `rust-toolchain.toml` and the release manifest. Older toolchains may work,
but are not a compatibility promise.
