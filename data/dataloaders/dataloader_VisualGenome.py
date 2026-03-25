from types import SimpleNamespace

import torch
from .coco import CocoDetection
from torch.utils.data import DataLoader, DistributedSampler, Subset
import utils.misc as utils
from data.dataloaders import build_dataset, get_coco_api_from_dataset

def load_data(args=None, **kwargs):
    if args is None:
        args = SimpleNamespace(**kwargs)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    debug_num_samples = 1000
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

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers
    )

    # 可选：保存评估用对象，但不参与 unpack
    base_ds = get_coco_api_from_dataset(dataset_val)

    # 如果你后面还需要 base_ds，可以挂到 loader 或 args 上
    data_loader_val.base_ds = base_ds

    return data_loader_train, data_loader_val, data_loader_val