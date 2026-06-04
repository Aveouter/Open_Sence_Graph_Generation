from functools import partial
from types import SimpleNamespace

import torch
from sgg.data.datasets.coco_base import CocoDetection
from torch.utils.data import DataLoader, DistributedSampler, Subset
import sgg.utils.misc as utils
from data.dataloaders import build_dataset, get_coco_api_from_dataset

# Fixed max image size for val/test collation: [C, H, W] = [3, 1344, 1344]
# Using a fixed size ensures all batches have identical tensor shapes,
# eliminating CUDA allocator fragmentation from per-batch variable sizes.
# 1344 covers the val transform's max_size=1333 plus rounding slack.
_VAL_FIXED_MAX_SIZE = [3, 1344, 1344]

def load_data(args=None, **kwargs):
    if args is None:
        args = SimpleNamespace(**kwargs)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # Debuging and testing code: 只取前 N 个样本，快速跑通流程
    debug_num_samples = None
    if debug_num_samples is not None:
        train_n = min(debug_num_samples, len(dataset_train))
        val_n = min(debug_num_samples, len(dataset_val))
        dataset_train = Subset(dataset_train, list(range(train_n)))
        dataset_val = Subset(dataset_val, list(range(val_n)))

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers
    )

    val_batch_size = getattr(args, 'val_batch_size', None) or args.batch_size
    val_collate = partial(utils.collate_fn, fixed_max_size=_VAL_FIXED_MAX_SIZE)

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=val_batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=val_collate,
        num_workers=args.num_workers
    )

    # 可选：保存评估用对象，挂到 dataset 上而非 loader
    base_ds = get_coco_api_from_dataset(dataset_val)
    data_loader_val.dataset.base_ds = base_ds

    return data_loader_train, data_loader_val, data_loader_val