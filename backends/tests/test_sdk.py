import unittest
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).resolve().parents[1] / "sdk-python"
sys.path.insert(0, str(SDK_ROOT))

from atlas_backend_sdk import Backend, BackendHealth


class EchoBackend(Backend):
    def health(self) -> BackendHealth:
        return BackendHealth("echo", "0.1.0", ("prepare", "solve", "cancel"))

    def prepare(self, problem: bytes) -> str:
        assert problem
        return "handle"

    def solve(self, handle: str, time_budget_ms: int) -> bytes:
        assert handle == "handle"
        assert time_budget_ms > 0
        return b"facts"

    def cancel(self, handle: str) -> None:
        assert handle == "handle"


class BackendSdkContractTest(unittest.TestCase):
    def test_backend_sdk_contract(self) -> None:
        backend = EchoBackend()

        self.assertEqual(backend.health().capabilities, ("prepare", "solve", "cancel"))
        handle = backend.prepare(b"problem")
        self.assertEqual(backend.solve(handle, 10), b"facts")
        backend.cancel(handle)


if __name__ == "__main__":
    unittest.main()
