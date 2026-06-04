# sgg/framework/base_evaluator.py
"""
Abstract base class for SGG evaluators.

Evaluators accumulate per-sample results and produce aggregated metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set

import numpy as np


class BaseEvaluator(ABC):
    """
    Protocol for evaluation systems.

    Evaluators are stateful between reset() calls: they accumulate
    per-sample results and produce aggregated metrics on demand.
    """

    @abstractmethod
    def supported_tasks(self) -> Set[str]:
        """Return supported task names (e.g., {'sgdet', 'predcls', 'sgcls'})."""

    @abstractmethod
    def evaluate_entry(
        self,
        gt_entry: Dict[str, np.ndarray],
        pred_entry: Dict[str, np.ndarray],
        task: str,
    ) -> None:
        """
        Evaluate a single image's predictions against ground truth.

        Args:
            gt_entry: Ground-truth dict with keys like 'boxes', 'labels',
                      'rel_annotations', 'orig_size'.
            pred_entry: Prediction dict with keys like 'sub_boxes', 'obj_boxes',
                        'sub_labels', 'obj_labels', 'rel_scores', 'rel_labels'.
            task: Task mode, one of 'sgdet', 'predcls', 'sgcls'.
        """

    @abstractmethod
    def aggregate(self, metrics: List[str]) -> Dict[str, float]:
        """
        Compute final metrics from all accumulated entries.

        Args:
            metrics: List of metric names like 'sgdet_R@50', 'sgdet_mR@20'.

        Returns:
            Dict mapping metric name to float value.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new evaluation round."""
