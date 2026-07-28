import json
import sys
import unittest
from pathlib import Path

BACKENDS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKENDS_ROOT / "native-math"))


class MathBackendParityTest(unittest.TestCase):
    def test_native_backend_solves_bitvector_problem_without_z3(self) -> None:
        from atlas_native_math_backend import NativeMathBackend

        backend = NativeMathBackend()
        self.assertIn("bitvector", backend.health().capabilities)
        handle = backend.prepare(
            json.dumps(
                {
                    "kind": "u8_xor_eq",
                    "mask": 0xAA,
                    "target": 0xFF,
                }
            ).encode()
        )

        result = json.loads(backend.solve(handle, 1000))

        self.assertEqual(result["status"], "sat")
        self.assertEqual(result["matches"], [0x55])

    def test_native_backend_solves_modular_linear_system_without_sage(self) -> None:
        from atlas_native_math_backend import NativeMathBackend

        backend = NativeMathBackend()
        handle = backend.prepare(
            json.dumps(
                {
                    "kind": "modular_linear",
                    "modulus": 7,
                    "matrix": [[2, 3], [4, 1]],
                    "rhs": [1, 6],
                }
            ).encode()
        )

        result = json.loads(backend.solve(handle, 1000))

        self.assertEqual(result["status"], "sat")
        self.assertEqual(result["solution"], [1, 2])

    def test_native_backend_solves_polynomial_gcd_without_sage(self) -> None:
        from atlas_native_math_backend import NativeMathBackend

        backend = NativeMathBackend()
        handle = backend.prepare(
            json.dumps(
                {
                    "kind": "polynomial_gcd",
                    "modulus": 5,
                    "left": [2, 3, 1],
                    "right": [3, 4, 1],
                }
            ).encode()
        )

        result = json.loads(backend.solve(handle, 1000))

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["gcd"], [1, 1])

    def test_native_backend_solves_modular_square_root_without_sage(self) -> None:
        from atlas_native_math_backend import NativeMathBackend

        backend = NativeMathBackend()
        handle = backend.prepare(
            json.dumps(
                {
                    "kind": "mod_sqrt_prime",
                    "value": 56,
                    "modulus": 101,
                }
            ).encode()
        )

        result = json.loads(backend.solve(handle, 1000))

        self.assertEqual(result["status"], "sat")
        self.assertEqual(result["roots"], [37, 64])

    def test_native_backend_solves_discrete_log_without_sage(self) -> None:
        from atlas_native_math_backend import NativeMathBackend

        backend = NativeMathBackend()
        handle = backend.prepare(
            json.dumps(
                {
                    "kind": "discrete_log_prime",
                    "base": 2,
                    "target": 22,
                    "modulus": 29,
                }
            ).encode()
        )

        result = json.loads(backend.solve(handle, 1000))

        self.assertEqual(result["status"], "sat")
        self.assertEqual(result["exponent"], 26)


if __name__ == "__main__":
    unittest.main()
