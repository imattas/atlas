"""Compare Atlas bounded search against optional external solvers."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "benchmarks" / "results"
JSON_RESULT = RESULT_DIR / "external-comparison.json"
MD_RESULT = RESULT_DIR / "external-comparison.md"
CTF_JSON_RESULT = RESULT_DIR / "ctf-benchmarks.json"
CTF_MD_RESULT = RESULT_DIR / "ctf-benchmarks.md"

CTF_RELEVANCE = {
    "xor_width20": "XOR masks and linear bit-vector checks common in crackmes and crypto warmups",
    "add_width20": "modular integer equality used in keygen and checksum gates",
    "checksum_width20": "bounded checksum residue search for license and firmware puzzles",
    "rotxor_width24": "rotate/XOR mixing used in reversing and obfuscation challenges",
    "muladd_width24": "LCG-style modular arithmetic used in PRNG and serial checks",
    "serial_bytes_width32": "byte-constrained serial-prefix search",
    "mod_sqrt_prime_101": "quadratic residue step used in CTF crypto",
    "discrete_log_prime_29": "small finite-field discrete logarithm baseline",
}


def with_ctf_relevance(row: dict[str, Any]) -> dict[str, Any]:
    """Attach CTF relevance metadata for known benchmark cases."""

    enriched = dict(row)
    enriched["ctf_relevance"] = CTF_RELEVANCE.get(str(row.get("name")), "availability or comparison metadata")
    if "iterations" not in enriched:
        enriched["iterations"] = 0
    if "mean_ns" not in enriched:
        enriched["mean_ns"] = 0
    return enriched


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

    cases: list[tuple[str, int, Any]] = [
        ("xor_width20", 20, lambda x: x ^ 0xAAAAA == 0xFFFFF),
        ("add_width20", 20, lambda x: x + 1 == 424_242),
        ("checksum_width20", 20, lambda x: z3.URem(x, z3.BitVecVal(997, 20)) == 313),
        ("rotxor_width24", 24, lambda x: z3.RotateLeft(x, 7) ^ 0xA5_A5_A5 == 0x12_34_56),
        ("muladd_width24", 24, lambda x: x * 65_537 + 0x1337 == 0xC0_FF_EE),
        (
            "serial_bytes_width32",
            32,
            lambda x: z3.And(
                z3.Extract(7, 0, x) == ord("C"),
                z3.Extract(15, 8, x) == ord("T"),
                z3.Extract(23, 16, x) == ord("F"),
            ),
        ),
    ]
    results: list[dict[str, Any]] = []
    for name, width, build_constraint in cases:
        samples: list[int] = []
        model_value = None
        iterations = 100
        for _ in range(iterations):
            x = z3.BitVec("x", width)
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
        ("xor_width20", 20, lambda candidate: (candidate ^ 0xAAAAA) & ((1 << 20) - 1) == 0xFFFFF),
        ("add_width20", 20, lambda candidate: (candidate + 1) & ((1 << 20) - 1) == 424_242),
        ("checksum_width20", 20, lambda candidate: candidate % 997 == 313),
        (
            "rotxor_width24",
            24,
            lambda candidate: ((((candidate << 7) | (candidate >> (24 - 7))) & ((1 << 24) - 1)) ^ 0xA5_A5_A5)
            == 0x12_34_56,
        ),
        ("muladd_width24", 24, lambda candidate: (candidate * 65_537 + 0x1337) & ((1 << 24) - 1) == 0xC0_FF_EE),
        ("serial_bytes_width32", 32, lambda candidate: candidate & 0x00FF_FFFF == 0x0046_5443),
    ]
    results: list[dict[str, Any]] = []
    for name, width, accepts in cases:
        samples: list[int] = []
        match_count = 0
        iterations = 3
        scan_end = min(1 << width, 1 << 24)
        for _ in range(iterations):
            start = time.perf_counter_ns()
            matches = []
            for candidate in range(scan_end):
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
                "candidates_evaluated": scan_end,
                "used_closed_form": False,
            }
        )
    return results


def run_native_math_backend() -> list[dict[str, Any]]:
    """Run Atlas' dependency-free native math backend on CTF-style kernels."""

    sys.path.insert(0, str(ROOT / "backends" / "native-math"))
    from atlas_native_math_backend import NativeMathBackend

    cases: list[tuple[str, dict[str, Any], int]] = [
        (
            "mod_sqrt_prime_101",
            {
                "kind": "mod_sqrt_prime",
                "value": 56,
                "modulus": 101,
            },
            2000,
        ),
        (
            "discrete_log_prime_29",
            {
                "kind": "discrete_log_prime",
                "base": 2,
                "target": 22,
                "modulus": 29,
            },
            2000,
        ),
    ]
    results: list[dict[str, Any]] = []
    backend = NativeMathBackend()
    for name, problem, iterations in cases:
        handle = backend.prepare(json.dumps(problem).encode())
        samples: list[int] = []
        last_result: dict[str, Any] = {}
        for _ in range(iterations):
            start = time.perf_counter_ns()
            last_result = json.loads(backend.solve(handle, 1000))
            samples.append(time.perf_counter_ns() - start)
        backend.cancel(handle)
        results.append(
            {
                "name": name,
                "engine": "atlas-native-math",
                "iterations": iterations,
                "mean_ns": int(mean(samples)),
                "matches": result_count(last_result),
                "candidates_evaluated": None,
                "used_closed_form": False,
            }
        )
    return results


def result_count(result: dict[str, Any]) -> int:
    """Return a comparable solution count from a backend response."""

    if result.get("status") not in {"sat", "ok"}:
        return 0
    if isinstance(result.get("roots"), list):
        return len(result["roots"])
    if result.get("exponent") is not None:
        return 1
    if isinstance(result.get("matches"), list):
        return len(result["matches"])
    return 1


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
    ctf_results = [with_ctf_relevance(row) for row in results if row.get("name") in CTF_RELEVANCE]
    CTF_JSON_RESULT.write_text(
        json.dumps(
            {
                "schema_major": 1,
                "kind": "ctf-benchmarks",
                "source": "benchmarks/compare_external.py --write",
                "cases": ctf_results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
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
    ctf_lines = [
        "# CTF benchmark results",
        "",
        "Measured cases cover bounded-search kernels and exact math operations that show up in reversing, crypto, and serial/keygen CTF tasks.",
        "",
        "| case | engine | mean ns | iterations | matches | CTF relevance |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in ctf_results:
        ctf_lines.append(
            f"| {row.get('name')} | {row.get('engine')} | {row.get('mean_ns')} | {row.get('iterations')} | {row.get('matches', 'n/a')} | {row.get('ctf_relevance')} |"
        )
    CTF_MD_RESULT.write_text("\n".join(ctf_lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run benchmark comparison."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write benchmark result files")
    args = parser.parse_args()

    results = run_atlas()
    results.extend(run_python_scalar())
    results.extend(run_native_math_backend())
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
