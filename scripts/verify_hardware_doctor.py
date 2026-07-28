"""Validate atlas doctor GPU diagnostics for hardware verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def validate_doctor(document: dict[str, Any], require_launch_abi: bool) -> list[str]:
    """Return validation errors for a doctor JSON document."""

    probes = document.get("gpu_feature_probes")
    if not isinstance(probes, list) or not probes:
        return ["GPU doctor must report at least one GPU feature probe"]

    successful_probes = [probe for probe in probes if isinstance(probe, dict) and probe.get("ok")]
    if not successful_probes:
        return ["GPU doctor must report at least one successful GPU feature probe"]

    errors: list[str] = []
    if require_launch_abi:
        for probe in successful_probes:
            name = probe.get("name", "<unknown>")
            features = probe.get("features")
            if not isinstance(features, list):
                errors.append(f"GPU feature probe {name} did not report a feature list")
                continue
            for required in ["launchAbiU32", "launchAbiU64"]:
                if required not in features:
                    errors.append(f"GPU feature probe {name} missing {required}")
    return errors


def main() -> int:
    """Validate doctor JSON from stdin."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="read doctor JSON from this file instead of stdin")
    parser.add_argument("--require-launch-abi", action="store_true")
    args = parser.parse_args()

    try:
        payload = (
            args.input.read_text(encoding="utf-8-sig") if args.input else sys.stdin.read()
        )
        document = json.loads(payload)
    except json.JSONDecodeError as error:
        print(f"invalid GPU doctor JSON: {error}", file=sys.stderr)
        return 1
    if not isinstance(document, dict):
        print("GPU doctor JSON must be an object", file=sys.stderr)
        return 1

    errors = validate_doctor(document, args.require_launch_abi)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
