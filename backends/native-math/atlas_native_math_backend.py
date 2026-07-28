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
                "modular-sqrt",
                "discrete-log",
            ),
        )

    def prepare(self, problem: bytes) -> str:
        document = json.loads(problem.decode())
        if document.get("kind") not in {
            "u8_xor_eq",
            "modular_linear",
            "polynomial_gcd",
            "mod_sqrt_prime",
            "discrete_log_prime",
        }:
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
        if kind == "polynomial_gcd":
            gcd = polynomial_gcd(problem["modulus"], problem["left"], problem["right"])
            return json.dumps({"status": "ok", "gcd": gcd}).encode()
        if kind == "mod_sqrt_prime":
            roots = mod_sqrt_prime(problem["value"], problem["modulus"])
            return json.dumps({"status": "sat" if roots is not None else "unsat", "roots": roots}).encode()
        exponent = discrete_log_prime(problem["base"], problem["target"], problem["modulus"])
        return json.dumps({"status": "sat" if exponent is not None else "unsat", "exponent": exponent}).encode()

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


def mod_sqrt_prime(value: int, modulus: int) -> list[int] | None:
    """Compute modular square roots over a prime field with Tonelli-Shanks."""

    if not is_prime(modulus):
        return None
    value %= modulus
    if value == 0:
        return [0]
    if modulus == 2:
        return [value]
    if pow(value, (modulus - 1) // 2, modulus) != 1:
        return None
    if modulus % 4 == 3:
        return sorted_roots(pow(value, (modulus + 1) // 4, modulus), modulus)

    q = modulus - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1

    non_residue = 2
    while pow(non_residue, (modulus - 1) // 2, modulus) != modulus - 1:
        non_residue += 1

    c = pow(non_residue, q, modulus)
    x = pow(value, (q + 1) // 2, modulus)
    t = pow(value, q, modulus)
    m = s
    while t != 1:
        i = 1
        t_power = t * t % modulus
        while i < m and t_power != 1:
            t_power = t_power * t_power % modulus
            i += 1
        if i == m:
            return None
        b = pow(c, 1 << (m - i - 1), modulus)
        x = x * b % modulus
        b_squared = b * b % modulus
        t = t * b_squared % modulus
        c = b_squared
        m = i
    return sorted_roots(x, modulus)


def discrete_log_prime(base: int, target: int, modulus: int) -> int | None:
    """Solve base**x == target mod prime modulus with baby-step/giant-step."""

    if not is_prime(modulus):
        return None
    base %= modulus
    target %= modulus
    if target == 1:
        return 0
    if base == 0:
        return None

    order = modulus - 1
    step = 0
    while step * step < order:
        step += 1

    baby_steps: dict[int, int] = {}
    value = 1
    for exponent in range(step):
        baby_steps.setdefault(value, exponent)
        value = value * base % modulus

    giant_stride = pow(pow(base, step, modulus), -1, modulus)
    gamma = target
    for giant in range(step + 1):
        if gamma in baby_steps:
            exponent = giant * step + baby_steps[gamma]
            if exponent < order:
                return exponent
        gamma = gamma * giant_stride % modulus
    return None


def sorted_roots(root: int, modulus: int) -> list[int]:
    """Return the two prime-field square roots in stable order."""

    other = (-root) % modulus
    return [root] if root == other else sorted([root, other])


def is_prime(value: int) -> bool:
    """Return whether value is prime by trial division."""

    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True
