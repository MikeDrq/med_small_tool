"""Regression tests for case-level bootstrap resampling."""

import unittest

from utils.bootstrap import grouped_bootstrap_ci


class GroupedBootstrapTest(unittest.TestCase):
    def test_redraws_whole_cases(self) -> None:
        values = [0.0, 0.0, 1.0, 1.0]
        case_ids = ["case_a", "case_a", "case_b", "case_b"]

        point, low, high = grouped_bootstrap_ci(
            values,
            case_ids,
            lambda items: sum(items) / len(items),
            n_boot=200,
            seed=7,
        )

        self.assertEqual(point, 0.5)
        self.assertEqual(low, 0.0)
        self.assertEqual(high, 1.0)

    def test_rejects_mismatched_group_ids(self) -> None:
        with self.assertRaises(ValueError):
            grouped_bootstrap_ci([1, 2], ["case_a"], lambda items: float(sum(items)))


if __name__ == "__main__":
    unittest.main()
