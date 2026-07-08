from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


BOX_SCALE = 1024.0


class OfficialVGH5EvalDataset(Dataset):
    """Visual Genome eval dataset using official PENET/SGB H5 semantics."""

    def __init__(
        self,
        img_folder,
        vg_root,
        split="test",
        transforms=None,
        filter_empty_rels=True,
    ):
        if split not in {"test", "val"}:
            raise ValueError(f"Official VG eval split must be test/val, got {split}")

        self.img_folder = Path(img_folder)
        self.vg_root = Path(vg_root)
        self.transforms = transforms
        self.split = split
        self.filter_empty_rels = filter_empty_rels

        self.roidb_file = self.vg_root / "VG-SGG-with-attri.h5"
        self.image_file = self.vg_root / "image_data.json"
        if not self.roidb_file.exists():
            raise FileNotFoundError(f"Official VG H5 not found: {self.roidb_file}")
        if not self.image_file.exists():
            raise FileNotFoundError(f"Official VG image_data not found: {self.image_file}")

        import json

        with self.image_file.open() as f:
            image_data = json.load(f)

        with h5py.File(self.roidb_file, "r") as roi_h5:
            split_flag = 2 if split == "test" else 0
            split_mask = roi_h5["split"][:] == split_flag
            split_mask &= roi_h5["img_to_first_box"][:] >= 0
            if filter_empty_rels:
                split_mask &= roi_h5["img_to_first_rel"][:] >= 0

            self.image_indices = np.where(split_mask)[0].astype(np.int64)
            self.image_data = [image_data[int(i)] for i in self.image_indices]
            self.ids = [int(row["image_id"]) for row in self.image_data]

            self.img_to_first_box = roi_h5["img_to_first_box"][self.image_indices]
            self.img_to_last_box = roi_h5["img_to_last_box"][self.image_indices]
            self.img_to_first_rel = roi_h5["img_to_first_rel"][self.image_indices]
            self.img_to_last_rel = roi_h5["img_to_last_rel"][self.image_indices]

            self.labels = roi_h5["labels"][:, 0].astype(np.int64)
            self.boxes_1024 = roi_h5["boxes_1024"][:].astype(np.float32)
            self.relationships = roi_h5["relationships"][:].astype(np.int64)
            self.predicates = roi_h5["predicates"][:, 0].astype(np.int64)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        info = self.image_data[idx]
        image_id = int(info["image_id"])
        width = int(info["width"])
        height = int(info["height"])
        image_path = self.img_folder / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        first_box = int(self.img_to_first_box[idx])
        last_box = int(self.img_to_last_box[idx])
        first_rel = int(self.img_to_first_rel[idx])
        last_rel = int(self.img_to_last_rel[idx])

        boxes = self._boxes_xyxy(first_box, last_box, width, height)
        labels = torch.as_tensor(
            self.labels[first_box:last_box + 1],
            dtype=torch.int64,
        )
        rel_annotations = self._relations(first_box, first_rel, last_rel)

        wh = boxes[:, 2:] - boxes[:, :2]
        area = wh[:, 0].clamp(min=0) * wh[:, 1].clamp(min=0)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.as_tensor([image_id], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
            "orig_size": torch.as_tensor([height, width], dtype=torch.int64),
            "size": torch.as_tensor([height, width], dtype=torch.int64),
            "rel_annotations": rel_annotations,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _boxes_xyxy(self, first_box, last_box, width, height):
        raw = self.boxes_1024[first_box:last_box + 1].copy()
        scale = float(max(width, height)) / BOX_SCALE
        boxes = np.zeros_like(raw, dtype=np.float32)
        boxes[:, 0] = (raw[:, 0] - raw[:, 2] / 2.0) * scale
        boxes[:, 1] = (raw[:, 1] - raw[:, 3] / 2.0) * scale
        boxes[:, 2] = (raw[:, 0] + raw[:, 2] / 2.0) * scale
        boxes[:, 3] = (raw[:, 1] + raw[:, 3] / 2.0) * scale
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, float(width))
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, float(height))
        return torch.as_tensor(boxes, dtype=torch.float32)

    def _relations(self, first_box, first_rel, last_rel):
        if first_rel < 0:
            return torch.zeros((0, 3), dtype=torch.int64)
        rel_pairs = self.relationships[first_rel:last_rel + 1] - int(first_box)
        predicates = self.predicates[first_rel:last_rel + 1, None]
        rels = np.concatenate([rel_pairs, predicates], axis=1)
        return torch.as_tensor(rels, dtype=torch.int64)
