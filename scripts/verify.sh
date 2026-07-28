#!/usr/bin/env bash
set -euo pipefail

profile="core"
if [[ "${1:-}" == "--profile" ]]; then
  profile="${2:-core}"
fi

cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets

if [[ "$profile" == "analysis" || "$profile" == "distributed" || "$profile" == "advanced" || "$profile" == "full" ]]; then
  for required_path in \
    tests/e2e/track2/manifest.toml \
    benchmarks/track2/manifest.toml \
    docs/guides/reversing.md \
    plugins/strategies/gf2/manifest.toml \
    plugins/strategies/modular-matrix/manifest.toml \
    plugins/strategies/lattice/manifest.toml \
    plugins/strategies/crypto-recognizers/manifest.toml
  do
    if [[ ! -e "$required_path" ]]; then
      echo "missing analysis release artifact: $required_path" >&2
      exit 1
    fi
  done
fi

if [[ -d python/tests ]]; then
  if python -m pytest --version >/dev/null 2>&1; then
    python -m pytest python/tests
  else
    python -m unittest discover python/tests
  fi
fi
if [[ -d backends ]]; then
  if python -m pytest --version >/dev/null 2>&1; then
    python -m pytest backends
  else
    python -m unittest discover backends/tests
  fi
fi

echo "Verification profile '${profile}' passed."
