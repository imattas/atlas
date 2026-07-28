"""Native-from-scratch Atlas math backend."""

from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SDK_ROOT = Path(__file__).resolve().parents[1] / "sdk-python"
sys.path.insert(0, str(SDK_ROOT))

from atlas_backend_sdk import Backend, BackendHealth


class NativeMathBackend(Backend):
    """Pure Atlas math backend with no Z3/Sage dependency."""

    def __init__(self) -> None:
        self._prepared: dict[str, dict[str, Any]] = {}

    def health(self) -> BackendHealth:
        return BackendHealth(
            "atlas-native-math",
            "0.1.0",
            (
                "prepare",
                "solve",
                "cancel",
                "bitvector",
                "exact-rational",
                "modular-linear",
                "polynomial-gcd",
            ),
        )

    def prepare(self, problem: bytes) -> str:
        document = json.loads(problem.decode())
        if document.get("kind") not in {"u8_xor_eq", "modular_linear", "polynomial_gcd"}:
            raise ValueError("unsupported native math problem kind")
        handle = str(uuid.uuid4())
        self._prepared[handle] = document
        return handle

    def solve(self, handle: str, time_budget_ms: int) -> bytes:
        if time_budget_ms <= 0:
            return json.dumps({"status": "timeout"}).encode()
        problem = self._prepared[handle]
        kind = problem["kind"]
        if kind == "u8_xor_eq":
            return json.dumps(
                {"status": "sat", "matches": [int(problem["mask"]) ^ int(problem["target"])]}
            ).encode()
        if kind == "modular_linear":
            solution = solve_modular_linear(problem["modulus"], problem["matrix"], problem["rhs"])
            return json.dumps({"status": "sat" if solution else "unsat", "solution": solution}).encode()
        gcd = polynomial_gcd(problem["modulus"], problem["left"], problem["right"])
        return json.dumps({"status": "ok", "gcd": gcd}).encode()

    def cancel(self, handle: str) -> None:
        self._prepared.pop(handle, None)


def solve_modular_linear(modulus: int, matrix: list[list[int]], rhs: list[int]) -> list[int] | None:
    """Solve a square linear system over a prime field."""

    size = len(rhs)
    rows = [[value % modulus for value in row] + [rhs[index] % modulus] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = next((row for row in range(column, size) if rows[row][column] % modulus), None)
        if pivot is None:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        inverse = pow(rows[column][column], -1, modulus)
        rows[column] = [(value * inverse) % modulus for value in rows[column]]
        for row_index in range(size):
            if row_index == column:
                continue
            factor = rows[row_index][column]
            rows[row_index] = [
                (value - factor * rows[column][cell_index]) % modulus
                for cell_index, value in enumerate(rows[row_index])
            ]
    return [row[size] for row in rows]


def polynomial_gcd(modulus: int, left: list[int], right: list[int]) -> list[int]:
    """Compute monic polynomial GCD over a prime field."""

    left = trim([value % modulus for value in left])
    right = trim([value % modulus for value in right])
    while right != [0]:
        left, right = right, poly_remainder(modulus, left, right)
    leading_inverse = pow(left[-1], -1, modulus)
    return trim([(value * leading_inverse) % modulus for value in left])


def poly_remainder(modulus: int, left: list[int], divisor: list[int]) -> list[int]:
    """Compute polynomial remainder over a prime field."""

    remainder = left[:]
    divisor_inverse = pow(divisor[-1], -1, modulus)
    while remainder != [0] and len(remainder) >= len(divisor):
        degree_delta = len(remainder) - len(divisor)
        factor = remainder[-1] * divisor_inverse % modulus
        for index, coefficient in enumerate(divisor):
            remainder[index + degree_delta] = (remainder[index + degree_delta] - factor * coefficient) % modulus
        remainder = trim(remainder)
    return remainder


def trim(polynomial: list[int]) -> list[int]:
    """Remove leading zero coefficients."""

    while len(polynomial) > 1 and polynomial[-1] == 0:
        polynomial.pop()
    return polynomial
