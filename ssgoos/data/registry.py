# ssgoos/data/registry.py
"""
Dataset registry for SGG framework.

Replaces the hardcoded string-matching dispatch in data/dataloaders/dataloader.py.
"""

from ssgoos.registry import DATASET_REGISTRY
from ssgoos.framework.base_dataset import BaseSGGDataset


def get_dataset(dataname: str, config):
    """
    Look up and build a dataset by name.

    Args:
        dataname: Dataset name, e.g. 'visual_genome', 'open_images', 'thyrotriples'.
        config: Configuration namespace or dict.

    Returns:
        (train_loader, val_loader, test_loader) tuple.

    Raises:
        KeyError: if dataname is not registered.
    """
    dataset_cls = DATASET_REGISTRY.get(dataname)
    dataset = dataset_cls(config)
    return dataset.build_loaders(
        batch_size=getattr(config, 'batch_size', 4),
        val_batch_size=getattr(config, 'val_batch_size', 4),
        num_workers=getattr(config, 'num_workers', 4),
        distributed=getattr(config, 'distributed', False),
    )
