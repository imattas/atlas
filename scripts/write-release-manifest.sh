#!/usr/bin/env bash
set -euo pipefail

python "$(dirname "$0")/write_release_manifest.py" --write RELEASE_MANIFEST.toml
