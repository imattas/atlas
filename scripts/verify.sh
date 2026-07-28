#!/usr/bin/env bash
set -euo pipefail

profile="core"
if [[ "${1:-}" == "--profile" ]]; then
  profile="${2:-core}"
fi

cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets

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
