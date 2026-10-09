"""The config contract: the defaults and merge policy a run is actually built from.

Two things make this worth a module of its own.  The merge policy -- which CLI
arguments survive a config-file load -- is what decides a run's real settings,
and it was written out seven times across ``train.py`` and ``tools/analysis``,
including one inverted copy.  And the metric list was derived inline inside
``train.py``'s ``if __name__ == "__main__"`` block, where nothing could reach it.

Stdlib-only, deliberately: the existing test for this area guards on torch and
so never runs in the dependency-free CI job.
"""

from __future__ import annotations

import unittest


class MetricsForEvalModeTest(unittest.TestCase):
    """Replaces the derivation that lived inline in train.py.

    The values are pinned to what that code produced, so this is a move rather
    than a change: a different list here would silently change which metric a
    checkpoint is selected on.
    """

    def test_sgdet_produces_three_recall_and_three_mean_recall_metrics(self) -> None:
        from src.config_contract import metrics_for_eval_mode

        self.assertEqual(
            metrics_for_eval_mode("sgdet"),
            [
                "sgdet_R@20",
                "sgdet_R@50",
                "sgdet_R@100",
                "sgdet_mR@20",
                "sgdet_mR@50",
                "sgdet_mR@100",
            ],
        )


class MergePolicyTest(unittest.TestCase):
    """Which CLI arguments survive a config-file load.

    The policy decides a run's real settings, so it is tested through the merge
    rather than by pinning the two tuples: what matters is which value wins, not
    how the key list is spelled.

    ``update_config`` is imported from here rather than from ``utils.main_utils``
    because that module imports cv2 and torch at module scope, which makes it
    unreachable in the dependency-free CI job.
    """

    def test_protected_key_keeps_its_command_line_value(self) -> None:
        from src.config_contract import merge_exclude_keys, update_config

        merged = update_config(
            {"method": "EGTR", "val_batch_size": None},
            {"method": "Motifs", "val_batch_size": 32},
            exclude_keys=merge_exclude_keys(overwrite=False),
        )

        self.assertEqual(merged["method"], "EGTR")
        self.assertIsNone(
            merged["val_batch_size"],
            "a protected key must not pick up the config file's value",
        )

    def test_overwrite_narrows_protection_to_method_alone(self) -> None:
        from src.config_contract import merge_exclude_keys, update_config

        merged = update_config(
            {"method": "EGTR", "val_batch_size": None},
            {"method": "Motifs", "val_batch_size": 32},
            exclude_keys=merge_exclude_keys(overwrite=True),
        )

        self.assertEqual(merged["method"], "EGTR")
        self.assertEqual(
            merged["val_batch_size"],
            32,
            "--overwrite must let the config file supply an unprotected key",
        )


if __name__ == "__main__":
    unittest.main()
