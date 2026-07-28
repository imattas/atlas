"""AtlasCTF Python SDK."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BitVec:
    """A Python SDK bit-vector value declaration."""

    name: str
    width: int


@dataclass(frozen=True)
class Result:
    """SDK solve result."""

    level: str
    explanation: str


@dataclass
class Project:
    """User-facing AtlasCTF project."""

    constraints: list[str] = field(default_factory=list)

    def bitvec(self, name: str, width: int) -> BitVec:
        """Declare a bit-vector."""
        if not name:
            raise ValueError("name must not be empty")
        if width <= 0:
            raise ValueError("width must be positive")
        return BitVec(name, width)

    def require(self, expression: str) -> None:
        """Record a textual constraint for the initial SDK boundary."""
        if not expression:
            raise ValueError("expression must not be empty")
        self.constraints.append(expression)

    def solve(self, strategy: str = "auto", timeout: int = 120) -> Result:
        """Solve through the local runtime boundary."""
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        return Result("UNKNOWN", f"strategy={strategy}; constraints={len(self.constraints)}")
