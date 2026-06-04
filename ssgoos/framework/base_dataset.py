# ssgoos/framework/base_dataset.py
"""
Abstract base class for SGG datasets.

Defines the protocol that all scene graph datasets must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from torch.utils.data import DataLoader


class BaseSGGDataset(ABC):
    """
    Protocol for scene graph datasets.

    Each dataset must produce (image, target) sample pairs where:
        image: Tensor [3, H, W] (image) or [T, 3, H, W] (video)
        target: dict with keys:
            - 'labels': LongTensor [num_objects]  -- object class IDs
            - 'boxes': FloatTensor [num_objects, 4]  -- (cx, cy, w, h), normalized [0,1]
            - 'rel_annotations': LongTensor [num_relations, 3]
                -- (subject_idx, object_idx, predicate_label)
            - 'orig_size': LongTensor [2]  -- (H, W)
            - 'image_id': int
    """

    @property
    @abstractmethod
    def num_entity_classes(self) -> int:
        """Number of entity/object classes (including background)."""

    @property
    @abstractmethod
    def num_predicate_classes(self) -> int:
        """Number of predicate/relation classes (including background)."""

    @property
    @abstractmethod
    def default_metrics(self) -> List[str]:
        """
        Default evaluation metrics for this dataset.

        Example: ['sgdet_R@10', 'sgdet_R@20', 'sgdet_R@50',
                   'sgdet_mR@10', 'sgdet_mR@20', 'sgdet_mR@50']
        """

    @abstractmethod
    def build_loaders(
        self,
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        distributed: bool,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Build train/val/test DataLoaders.

        Returns:
            (train_loader, val_loader, test_loader)
        """
