"""Python backend adapter SDK contracts for AtlasCTF."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendHealth:
    """Backend capability advertisement."""

    name: str
    version: str
    capabilities: tuple[str, ...]


class Backend:
    """Base class for isolated Python backend adapters."""

    def health(self) -> BackendHealth:
        """Return backend health and capabilities."""
        raise NotImplementedError

    def prepare(self, problem: bytes) -> str:
        """Prepare an encoded problem and return a backend-local handle."""
        raise NotImplementedError

    def solve(self, handle: str, time_budget_ms: int) -> bytes:
        """Solve a prepared problem and return encoded facts or candidates."""
        raise NotImplementedError

    def cancel(self, handle: str) -> None:
        """Cancel a prepared problem."""
        raise NotImplementedError
