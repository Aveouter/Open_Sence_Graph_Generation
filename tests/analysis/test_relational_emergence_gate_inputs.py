"""The gate must refuse inputs that describe a different experiment.

The Phase IA driver folds the Level-2 oracle result into its verdict via
``--level2``. That result records the SHA-256 of the dataset it was measured
against, and the point of recording it is to be checked: a result measured on a
different dataset would put a number in the gate that looks valid and describes
an experiment the run did not perform.

This was not hypothetical. After a schema change to the episode file, the
artifact on disk still carried the *old* digest, and the difference was found by
comparing two hex strings by hand. The check below is what makes that automatic.
"""

from __future__ import annotations

import unittest

from tools.relational_emergence.experiments.run_phase1a import check_level2_dataset

CURRENT = "2e4bb8779fb8087265469e4e99e12bbfbfd5f8cab81d6bae0818194dd62590ae"
OTHER = "835b0830aeeb68a5157d00ced80cc83c3ebb6542070bae36799ee57f2dfbf4ed"


class TestGateRefusesStaleLevel2(unittest.TestCase):
    def test_a_matching_digest_is_accepted(self) -> None:
        check_level2_dataset({"dataset_sha256": CURRENT}, CURRENT)

    def test_a_result_from_another_dataset_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            check_level2_dataset({"dataset_sha256": OTHER}, CURRENT)
        message = str(caught.exception)
        self.assertIn(OTHER, message)
        self.assertIn(CURRENT, message)

    def test_a_missing_digest_is_refused(self) -> None:
        """An artifact that cannot be shown to describe this dataset is not usable."""
        with self.assertRaises(SystemExit) as caught:
            check_level2_dataset({"delta_M": 0.18}, CURRENT)
        self.assertIn("no dataset digest", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
