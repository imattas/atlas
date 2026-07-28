import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlasctf import Project


class PythonSdkTest(unittest.TestCase):
    def test_project_solve_returns_result(self) -> None:
        project = Project()
        x = project.bitvec("x", 8)
        project.require(f"{x.name} == 65")

        result = project.solve()

        self.assertEqual(result.level, "UNKNOWN")
        self.assertIn("constraints=1", result.explanation)


if __name__ == "__main__":
    unittest.main()
