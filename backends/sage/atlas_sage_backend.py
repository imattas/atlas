"""SageMath backend adapter for Atlas math parity."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

SDK_ROOT = Path(__file__).resolve().parents[1] / "sdk-python"
sys.path.insert(0, str(SDK_ROOT))

from atlas_backend_sdk import Backend, BackendHealth


class SageBackend(Backend):
    """Sage command-line adapter.

    Sage is intentionally invoked as an external process so Atlas can use Sage's
    broad algebra/number-theory stack when installed while reporting precise
    unavailability otherwise.
    """

    def __init__(self) -> None:
        self._prepared: dict[str, dict[str, Any]] = {}
        self._sage = shutil.which("sage")

    def health(self) -> BackendHealth:
        if self._sage is None:
            return BackendHealth(
                "sage",
                "unavailable",
                ("sage-cli", "algebra", "number-theory", "polynomial-rings", "linear-algebra"),
                False,
                "sage CLI not found on PATH",
            )
        version = subprocess.check_output([self._sage, "--version"], text=True).strip()
        return BackendHealth(
            "sage",
            version,
            ("prepare", "solve", "cancel", "sage-cli", "algebra", "number-theory", "polynomial-rings", "linear-algebra"),
        )

    def prepare(self, problem: bytes) -> str:
        if self._sage is None:
            raise RuntimeError("sage CLI not found on PATH")
        document = json.loads(problem.decode())
        if document.get("kind") != "sage":
            raise ValueError("unsupported sage problem kind")
        handle = str(uuid.uuid4())
        self._prepared[handle] = document
        return handle

    def solve(self, handle: str, time_budget_ms: int) -> bytes:
        if self._sage is None:
            raise RuntimeError("sage CLI not found on PATH")
        problem = self._prepared[handle]
        with tempfile.NamedTemporaryFile("w", suffix=".sage", delete=False, encoding="utf-8") as script:
            script.write(problem["code"])
            script_path = script.name
        try:
            completed = subprocess.run(
                [self._sage, script_path],
                text=True,
                capture_output=True,
                timeout=max(time_budget_ms / 1000.0, 0.001),
                check=False,
            )
            return json.dumps(
                {
                    "status": "ok" if completed.returncode == 0 else "error",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            ).encode()
        finally:
            Path(script_path).unlink(missing_ok=True)

    def cancel(self, handle: str) -> None:
        self._prepared.pop(handle, None)
