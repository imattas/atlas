"""Compare Atlas bounded search against optional external solvers."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmarks" / "results"
JSON_RESULT = RESULT_DIR / "external-comparison.json"
MD_RESULT = RESULT_DIR / "external-comparison.md"


def run_atlas() -> list[dict[str, Any]]:
    """Run the Rust native benchmark and return parsed results."""

    output = subprocess.check_output(
        ["cargo", "run", "--release", "-p", "atlas-search-native", "--example", "bench_native"],
        cwd=ROOT,
        text=True,
    )
    return json.loads(output)


def z3_available() -> bool:
    """Return whether the Python Z3 bindings are importable."""

    try:
        import z3  # noqa: F401
    except ImportError:
        return False
    return True


def run_z3() -> list[dict[str, Any]]:
    """Run Z3 model-finding baselines for the same constraints."""

    import z3

    cases = [
        ("xor_width20", lambda x: x ^ 0xAAAAA == 0xFFFFF),
        ("add_width20", lambda x: x + 1 == 424_242),
        ("checksum_width20", lambda x: z3.URem(x, z3.BitVecVal(997, 20)) == 313),
    ]
    results: list[dict[str, Any]] = []
    for name, build_constraint in cases:
        samples: list[int] = []
        model_value = None
        iterations = 100
        for _ in range(iterations):
            x = z3.BitVec("x", 20)
            solver = z3.Solver()
            solver.add(build_constraint(x))
            start = time.perf_counter_ns()
            status = solver.check()
            if status == z3.sat:
                model_value = solver.model()[x].as_long()
            samples.append(time.perf_counter_ns() - start)
        results.append(
            {
                "name": name,
                "engine": "z3-python-first-model",
                "iterations": iterations,
                "mean_ns": int(mean(samples)),
                "matches": 1 if model_value is not None else 0,
                "candidates_evaluated": None,
                "used_closed_form": False,
            }
        )
    return results


def run_python_scalar() -> list[dict[str, Any]]:
    """Run a dependency-free scalar enumeration baseline."""

    cases = [
        ("xor_width20", lambda candidate: (candidate ^ 0xAAAAA) & ((1 << 20) - 1) == 0xFFFFF),
        ("add_width20", lambda candidate: (candidate + 1) & ((1 << 20) - 1) == 424_242),
        ("checksum_width20", lambda candidate: candidate % 997 == 313),
    ]
    results: list[dict[str, Any]] = []
    for name, accepts in cases:
        samples: list[int] = []
        match_count = 0
        iterations = 3
        for _ in range(iterations):
            start = time.perf_counter_ns()
            matches = []
            for candidate in range(1 << 20):
                if accepts(candidate):
                    matches.append(candidate)
                    if len(matches) >= 1024:
                        break
            samples.append(time.perf_counter_ns() - start)
            match_count = len(matches)
        results.append(
            {
                "name": name,
                "engine": "python-scalar",
                "iterations": iterations,
                "mean_ns": int(mean(samples)),
                "matches": match_count,
                "candidates_evaluated": None,
                "used_closed_form": False,
            }
        )
    return results


def sage_result() -> list[dict[str, Any]]:
    """Return Sage availability metadata."""

    sage = shutil.which("sage")
    if sage is None:
        return [
            {
                "name": "all",
                "engine": "sage",
                "available": False,
                "reason": "sage CLI not found on PATH",
            }
        ]
    version = subprocess.check_output([sage, "--version"], text=True).strip()
    return [{"name": "all", "engine": "sage", "available": True, "version": version}]


def write_reports(results: list[dict[str, Any]]) -> None:
    """Write JSON and Markdown benchmark reports."""

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_RESULT.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# External solver comparison",
        "",
        "Lower `mean_ns` is better. Atlas rows produce the bounded match stream up to the existing 1024-result cap; Z3 rows measure first-model solving through Python bindings.",
        "",
        "| case | engine | mean ns | iterations | matches | notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in results:
        notes = []
        if row.get("available") is False:
            notes.append(str(row.get("reason", "unavailable")))
        if row.get("used_closed_form"):
            notes.append("closed-form")
        if row.get("candidates_evaluated") is not None:
            notes.append(f"evaluated={row['candidates_evaluated']}")
        lines.append(
            f"| {row.get('name')} | {row.get('engine')} | {row.get('mean_ns', 'n/a')} | {row.get('iterations', 'n/a')} | {row.get('matches', 'n/a')} | {'; '.join(notes)} |"
        )
    MD_RESULT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run benchmark comparison."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write benchmark result files")
    args = parser.parse_args()

    results = run_atlas()
    results.extend(run_python_scalar())
    if z3_available():
        results.extend(run_z3())
    else:
        results.append({"name": "all", "engine": "z3-python", "available": False, "reason": "z3 module unavailable"})
    results.extend(sage_result())
    if args.write:
        write_reports(results)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
