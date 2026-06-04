import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

NUM_ENT, NUM_REL = 23, 29  # 实体/关系类别数

class ThyroTriplesDataset(Dataset):
    """时序输入 (C,T,H,W) + 灵活 targets（关系带时间戳，不按帧 list 返回）"""
    def __init__(self, data_root: str, pre_seq_length: int, aft_seq_length: int, in_shape: tuple,
                 split: str = "train", dataset_size: int = 1000,
                 min_obj: int = 6, max_obj: int = 16, min_rel: int = 12, max_rel: int = 40,
                 rel_time_coverage: float = 0.6, seed: int = 0, **kw):
        self.C, self.T, self.H, self.W = map(int, in_shape)              # 直接用 in_shape 的 T
        self.n = int(dataset_size)
        self.min_obj, self.max_obj = int(min_obj), int(max_obj)
        self.min_rel, self.max_rel = int(min_rel), int(max_rel)
        self.cov = float(rel_time_coverage)
        off = {"train": 0, "val": 12345, "test": 54321}.get(split, 999)
        self.seed = int(seed) + off

    def __len__(self): return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx * 9973)

        # 1) 随机生成序列图像： (C,T,H,W)
        images = rng.random((self.C, self.T, self.H, self.W), dtype=np.float32)

        # 2) 生成对象池（全序列共享，不按帧对齐）
        N = int(rng.integers(self.min_obj, self.max_obj + 1))
        labels = rng.integers(0, NUM_ENT, size=N, dtype=np.int64)

        # 随机框：xyxy 像素坐标
        x1 = rng.uniform(0, self.W * 0.8, size=N).astype(np.float32)
        y1 = rng.uniform(0, self.H * 0.8, size=N).astype(np.float32)
        x2 = (x1 + rng.uniform(self.W * 0.05, self.W * 0.25, size=N)).clip(0, self.W).astype(np.float32)
        y2 = (y1 + rng.uniform(self.H * 0.05, self.H * 0.25, size=N)).clip(0, self.H).astype(np.float32)
        boxes = np.stack([x1, y1, x2, y2], 1)

        # 3) 关系事件：(R,4) 每行 [t, sub_idx, obj_idx, rel_id]
        R = int(rng.integers(self.min_rel, self.max_rel + 1))
        ts = np.arange(self.T, dtype=np.int64); rng.shuffle(ts)
        active = ts[:max(1, int(np.ceil(self.T * self.cov)))]            # 只在部分时间步产生关系

        t = rng.choice(active, size=R, replace=True).astype(np.int64)
        s = rng.integers(0, N, size=R, dtype=np.int64)
        o = rng.integers(0, N, size=R, dtype=np.int64)
        o[s == o] = (o[s == o] + 1) % N                                  # 避免自环
        r = rng.integers(0, NUM_REL, size=R, dtype=np.int64)
        rel_triples = np.stack([t, s, o, r], 1)

        target = {
            "boxes": torch.from_numpy(boxes).float(),                    # (N,4)
            "labels": torch.from_numpy(labels).long(),                   # (N,)
            "rel_triples": torch.from_numpy(rel_triples).long(),         # (R,4)
            "image_id": torch.tensor([idx]).long(),
        }
        return torch.from_numpy(images), target

def collate_fn(batch):
    # images 可堆叠；targets 变长保留 list（最通用）
    images = torch.stack([b[0] for b in batch], 0)                       # (B,C,T,H,W)
    targets = [b[1] for b in batch]
    return images, targets

def load_data(data_root: str, batch_size: int, val_batch_size: int, num_workers: int, **kw):
    assert kw.get("in_shape"), "需要 in_shape=(C,T,H,W)"
    pre, aft = kw.get("pre_seq_length", 0), kw.get("aft_seq_length", 0)  # 保留接口但不强依赖
    train_ds = ThyroTriplesDataset(data_root, pre, aft, kw["in_shape"], split="train", dataset_size=kw.get("dataset_size", 1000))
    val_ds   = ThyroTriplesDataset(data_root, pre, aft, kw["in_shape"], split="val",   dataset_size=kw.get("val_dataset_size", 200))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=val_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader
