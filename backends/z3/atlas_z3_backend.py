"""Z3 backend adapter for Atlas math parity."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

SDK_ROOT = Path(__file__).resolve().parents[1] / "sdk-python"
sys.path.insert(0, str(SDK_ROOT))

from atlas_backend_sdk import Backend, BackendHealth


class Z3Backend(Backend):
    """Out-of-process compatible Z3 adapter.

    The adapter exposes raw SMT-LIB2 pass-through for broad Z3 theory coverage
    and a small structured z3py path used by Atlas tests and orchestrators.
    """

    def __init__(self) -> None:
        self._prepared: dict[str, dict[str, Any]] = {}

    def health(self) -> BackendHealth:
        try:
            import z3

            version = ".".join(str(part) for part in z3.get_version())
            return BackendHealth(
                "z3",
                version,
                (
                    "prepare",
                    "solve",
                    "cancel",
                    "smtlib2",
                    "z3py",
                    "bitvec",
                    "arrays",
                    "int-real-arithmetic",
                    "quantifiers",
                    "optimize",
                    "unsat-core",
                ),
            )
        except ImportError as exc:
            return BackendHealth("z3", "unavailable", ("z3-python",), False, str(exc))

    def prepare(self, problem: bytes) -> str:
        if not self.health().available:
            raise RuntimeError("z3 Python bindings not found")
        document = json.loads(problem.decode())
        if document.get("kind") not in {"smtlib2", "z3py"}:
            raise ValueError("unsupported z3 problem kind")
        handle = str(uuid.uuid4())
        self._prepared[handle] = document
        return handle

    def solve(self, handle: str, time_budget_ms: int) -> bytes:
        import z3

        problem = self._prepared[handle]
        if problem["kind"] == "smtlib2":
            return json.dumps(_solve_smtlib2(z3, problem["script"], time_budget_ms)).encode()
        return json.dumps(_solve_structured_z3py(z3, problem, time_budget_ms)).encode()

    def cancel(self, handle: str) -> None:
        self._prepared.pop(handle, None)


def _solve_smtlib2(z3: Any, script: str, time_budget_ms: int) -> dict[str, Any]:
    solver = z3.Solver()
    solver.set(timeout=time_budget_ms)
    solver.from_string(script)
    status = solver.check()
    result: dict[str, Any] = {"status": str(status)}
    if status == z3.sat:
        result["model"] = {decl.name(): str(solver.model()[decl]) for decl in solver.model().decls()}
    elif status == z3.unsat:
        result["unsat_core"] = [str(item) for item in solver.unsat_core()]
    return result


def _solve_structured_z3py(z3: Any, problem: dict[str, Any], time_budget_ms: int) -> dict[str, Any]:
    variables: dict[str, Any] = {}
    for variable in problem.get("variables", []):
        name = variable["name"]
        sort = variable["sort"]
        if sort == "int":
            variables[name] = z3.Int(name)
        elif sort == "real":
            variables[name] = z3.Real(name)
        elif sort == "bitvec":
            variables[name] = z3.BitVec(name, int(variable["width"]))
        else:
            raise ValueError(f"unsupported z3 sort: {sort}")

    objective = problem.get("objective")
    solver = z3.Optimize() if objective else z3.Solver()
    solver.set(timeout=time_budget_ms)
    for constraint in problem.get("constraints", []):
        solver.add(_term(z3, variables, constraint))
    if objective:
        term = _term(z3, variables, objective["term"])
        if objective["direction"] == "minimize":
            solver.minimize(term)
        elif objective["direction"] == "maximize":
            solver.maximize(term)
        else:
            raise ValueError("objective direction must be minimize or maximize")
    status = solver.check()
    result: dict[str, Any] = {"status": str(status)}
    if status == z3.sat:
        model = solver.model()
        result["model"] = {name: str(model.eval(value, model_completion=True)) for name, value in variables.items()}
    return result


def _term(z3: Any, variables: dict[str, Any], node: Any) -> Any:
    if isinstance(node, str):
        return variables[node]
    if isinstance(node, int | float):
        return node
    op = node["op"]
    if op == "ge":
        return _term(z3, variables, node["left"]) >= _term(z3, variables, node["right"])
    if op == "le":
        return _term(z3, variables, node["left"]) <= _term(z3, variables, node["right"])
    if op == "eq":
        return _term(z3, variables, node["left"]) == _term(z3, variables, node["right"])
    if op == "add":
        return _term(z3, variables, node["left"]) + _term(z3, variables, node["right"])
    if op == "sub":
        return _term(z3, variables, node["left"]) - _term(z3, variables, node["right"])
    if op == "mul":
        return _term(z3, variables, node["left"]) * _term(z3, variables, node["right"])
    raise ValueError(f"unsupported z3 term op: {op}")
