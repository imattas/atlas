import json
import sys
import unittest
from pathlib import Path

BACKENDS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKENDS_ROOT / "z3"))
sys.path.insert(0, str(BACKENDS_ROOT / "sage"))


class MathBackendParityTest(unittest.TestCase):
    def test_z3_backend_solves_raw_smtlib_bitvector_problem(self) -> None:
        from atlas_z3_backend import Z3Backend

        backend = Z3Backend()
        self.assertIn("smtlib2", backend.health().capabilities)
        handle = backend.prepare(
            json.dumps(
                {
                    "kind": "smtlib2",
                    "script": """
                    (set-logic QF_BV)
                    (declare-fun x () (_ BitVec 8))
                    (assert (= (bvxor x #xaa) #xff))
                    (check-sat)
                    (get-model)
                    """,
                }
            ).encode()
        )

        result = json.loads(backend.solve(handle, 1000))

        self.assertEqual(result["status"], "sat")
        self.assertIn("x", result["model"])

    def test_z3_backend_supports_optimize_for_theory_parity(self) -> None:
        from atlas_z3_backend import Z3Backend

        backend = Z3Backend()
        handle = backend.prepare(
            json.dumps(
                {
                    "kind": "z3py",
                    "variables": [{"name": "x", "sort": "int"}],
                    "constraints": [{"op": "ge", "left": "x", "right": 7}],
                    "objective": {"direction": "minimize", "term": "x"},
                }
            ).encode()
        )

        result = json.loads(backend.solve(handle, 1000))

        self.assertEqual(result["status"], "sat")
        self.assertEqual(result["model"]["x"], "7")

    def test_sage_backend_reports_missing_cli_precisely(self) -> None:
        from atlas_sage_backend import SageBackend

        backend = SageBackend()
        health = backend.health()
        self.assertIn("sage-cli", health.capabilities)
        if health.available:
            handle = backend.prepare(json.dumps({"kind": "sage", "code": "print(factor(91))"}).encode())
            result = json.loads(backend.solve(handle, 1000))
            self.assertEqual(result["status"], "ok")
            self.assertIn("7", result["stdout"])
        else:
            with self.assertRaisesRegex(RuntimeError, "sage CLI not found"):
                backend.prepare(json.dumps({"kind": "sage", "code": "print(2+2)"}).encode())


if __name__ == "__main__":
    unittest.main()
