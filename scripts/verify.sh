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
  python -m pytest python/tests
fi
if [[ -d backends ]]; then
  python -m pytest backends
fi

echo "Verification profile '${profile}' passed."
