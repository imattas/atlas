import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_ranker import RankRequest, rank


class RankerTest(unittest.TestCase):
    def test_ranker_response_is_allowlisted(self) -> None:
        response = rank(RankRequest(("general-smt", "gf2"), {"vars": 32.0}))

        self.assertEqual(response.ordered_strategy_ids, ("gf2", "general-smt"))
        self.assertEqual(set(response.budget_multipliers), {"gf2", "general-smt"})
        self.assertIn("baseline", response.explanation)


if __name__ == "__main__":
    unittest.main()
