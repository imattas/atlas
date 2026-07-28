#!/usr/bin/env bash
set -euo pipefail

python "$(dirname "$0")/write_manifest.py" --write release/manifest.toml
