# ssgoos/data/datasets/action_genome.py
"""
Action Genome Dataset (future).

Video-based scene graph dataset with temporal relation annotations.
This is a skeleton implementation for future video SGG support.

Dataset statistics (to be filled):
  - Videos: ~10K
  - Object classes: 36
  - Predicate classes: 26
  - Relations: Temporal (with start/end timestamps)
"""

from typing import List, Tuple

from torch.utils.data import DataLoader

from ssgoos.registry import DATASET_REGISTRY
from ssgoos.framework.base_dataset import BaseSGGDataset


@DATASET_REGISTRY.register('action_genome')
class ActionGenomeDataset(BaseSGGDataset):
    """Action Genome video scene graph dataset (skeleton)."""

    def __init__(self, config=None):
        self._config = config

    # ------------------------------------------------------------------
    # BaseSGGDataset interface
    # ------------------------------------------------------------------

    @property
    def num_entity_classes(self) -> int:
        return 36  # 35 object classes + 1 background

    @property
    def num_predicate_classes(self) -> int:
        return 26  # 25 predicate classes + 1 background

    @property
    def default_metrics(self) -> List[str]:
        return [
            "vid_sgdet_R@10",
            "vid_sgdet_R@20",
            "vid_sgdet_R@50",
        ]

    def build_loaders(
        self,
        batch_size: int = 4,
        val_batch_size: int = 4,
        num_workers: int = 4,
        distributed: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Build video DataLoaders for Action Genome.

        TODO: Implement actual dataset loading.
        """
        raise NotImplementedError(
            "ActionGenome dataset not yet implemented. "
            "This is a placeholder for future video SGG support."
        )
