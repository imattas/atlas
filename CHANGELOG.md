# Changelog

All notable AtlasCTF changes are recorded here. Versions follow Semantic
Versioning. The `v1.0.0` line means the documented CLI and JSON schemas are
stable; it does not mean every optional GPU SDK is available on every host.

## [Unreleased]

- Continued v1 stabilization and release automation work.

## [1.0.0-rc.1] - 2026-07-28

- Added cross-platform release binaries for Windows, Linux, and macOS.
- Added release-artifact verification tests and installer fallback behavior.
- Defined the stable CLI command and JSON output contract.
- Expanded native CTF math with encoding, XOR, classical-cipher, padding, and
  SHA-256 helpers, alongside the existing number-theory and GF(2) routines.
- Added CPU/GPU backend support documentation and compatibility policy.

## [0.1.0]

- Initial GitHub Release with source archive, checksums, release manifest, CTF
  benchmarks, native math, and optional GPU adapter crates.
