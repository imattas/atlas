from __future__ import annotations

import json
import sys
from typing import Any


def summarize_matches(matches: Any) -> dict[str, Any]:
    if not isinstance(matches, list):
        matches = []
    return {
        "match_count": len(matches),
        "first_matches": matches[:5],
        "last_matches": matches[-5:] if len(matches) > 5 else matches[:],
    }


def summarize_benchmark(document: dict[str, Any]) -> dict[str, Any]:
    accelerator = document.get("accelerator") or {}
    native = document.get("native") or {}
    simd = document.get("simd") or {}
    summary: dict[str, Any] = {
        "schema_major": document.get("schema_major"),
        "kind": document.get("kind"),
        "fixture": document.get("fixture"),
        "domain": document.get("domain"),
        "sample_count": document.get("sample_count"),
        "native_samples_ns": document.get("native_samples_ns"),
        "simd_samples_ns": document.get("simd_samples_ns"),
        "accelerator_samples_ns": document.get("accelerator_samples_ns"),
        "native": {
            "elapsed_ns": native.get("elapsed_ns"),
            **summarize_matches(native.get("matches")),
        },
        "simd": {
            "elapsed_ns": simd.get("elapsed_ns"),
            **summarize_matches(simd.get("matches")),
        },
        "accelerator": {
            "elapsed_ns": accelerator.get("elapsed_ns"),
            "requested_gpu_sdk": accelerator.get("requested_gpu_sdk"),
            "actual_gpu_sdk": accelerator.get("actual_gpu_sdk"),
            "hardware": accelerator.get("hardware"),
            "mode": accelerator.get("mode"),
            "launch": accelerator.get("launch"),
            "telemetry": accelerator.get("telemetry"),
            **summarize_matches(accelerator.get("matches")),
        },
    }
    if "speedup_ratio" in accelerator:
        summary["accelerator"]["speedup_ratio"] = accelerator.get("speedup_ratio")
    return summary


def main() -> int:
    input_path = None
    if len(sys.argv) == 3 and sys.argv[1] == "--input":
        input_path = sys.argv[2]
    elif len(sys.argv) != 1:
        print("usage: summarize_hardware_benchmark.py [--input PATH]", file=sys.stderr)
        return 2

    try:
        if input_path is None:
            document = json.load(sys.stdin)
        else:
            with open(input_path, encoding="utf-8-sig") as handle:
                document = json.load(handle)
    except json.JSONDecodeError as error:
        print(f"invalid benchmark JSON: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"failed to read benchmark JSON: {error}", file=sys.stderr)
        return 1
    if not isinstance(document, dict):
        print("benchmark JSON must be an object", file=sys.stderr)
        return 1
    print(json.dumps(summarize_benchmark(document), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
