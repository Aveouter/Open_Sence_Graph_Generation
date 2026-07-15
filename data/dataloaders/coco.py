# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

"""
dataset (COCO-like) which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import json
import os
import h5py
import numpy as np
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import utils.transforms as T

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks,
                 use_official_vg_h5_eval=False, vg_h5_file=None,
                 vg_image_data_file=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.data_name = Path(img_folder).parent.name  #TODO: 直接从路径中解析数据集名称，或者在 args 中指定
        # print(f"Initialized CocoDetection with data_name: {self.data_name}")


        # load relationship annotations
        ann_dir = os.path.dirname(ann_file)
        with open(os.path.join(ann_dir, 'rel.json'), 'r') as f:
            all_rels = json.load(f)
        if 'train' in ann_file:
            self.rel_annotations = all_rels['train']
        elif 'val' in ann_file:
            self.rel_annotations = all_rels['val']
        else:
            self.rel_annotations = all_rels['test']

        self.rel_categories = all_rels['rel_categories']

        # The historical COCO conversion rounds VG boxes and deduplicates exact
        # relation triplets. Official maskrcnn-benchmark evaluation instead
        # reads float boxes and raw (duplicate-preserving) relation tuples from
        # the Stanford H5 file. RA-SGG checkpoint evaluation opts into that
        # source explicitly so GT semantics match the official evaluator.
        self.use_official_vg_h5_eval = bool(use_official_vg_h5_eval)
        self._official_vg_eval_targets = None
        if self.use_official_vg_h5_eval:
            if vg_h5_file is None or vg_image_data_file is None:
                raise ValueError(
                    "official VG H5 evaluation requires vg_h5_file and "
                    "vg_image_data_file"
                )
            self._official_vg_eval_targets = self._load_official_vg_eval_targets(
                img_folder, vg_h5_file, vg_image_data_file
            )

    def _load_official_vg_eval_targets(self, img_folder, h5_file, image_data_file):
        """Load the exact official test targets, preserving duplicate GT rels."""
        with open(image_data_file, 'r') as f:
            image_data = json.load(f)

        corrupted = {1592, 1722, 4616, 4617}
        image_data = [
            item for item in image_data
            if int(item['image_id']) not in corrupted
            and os.path.exists(os.path.join(img_folder, f"{item['image_id']}.jpg"))
        ]

        with h5py.File(h5_file, 'r') as h5:
            split = h5['split'][:]
            first_box = h5['img_to_first_box'][:]
            last_box = h5['img_to_last_box'][:]
            first_rel = h5['img_to_first_rel'][:]
            last_rel = h5['img_to_last_rel'][:]
            selected = np.flatnonzero(
                (split == 2) & (first_box >= 0) & (first_rel >= 0)
            )

            if len(image_data) != len(split):
                raise RuntimeError(
                    "VG image_data/H5 alignment mismatch after removing the four "
                    f"known corrupt images: {len(image_data)} != {len(split)}"
                )
            official_ids = [int(image_data[idx]['image_id']) for idx in selected]
            local_ids = list(map(int, self.ids))
            if len(local_ids) != len(official_ids) or set(local_ids) != set(official_ids):
                raise RuntimeError(
                    "COCO test image set does not match official VG H5 test set: "
                    f"local_count={len(self.ids)}, "
                    f"official_count={len(official_ids)}"
                )
            if local_ids != official_ids:
                # torchvision sorts COCO image ids, whereas the official VG
                # loader preserves H5/image_data order. Restore that order.
                self.ids = official_ids

            # Keep the original int32 dtype through the in-place cxcywh->xyxy
            # conversion. This intentionally preserves maskrcnn-benchmark's
            # integer cast/truncation before the later image-scale conversion.
            all_boxes = h5['boxes_1024'][:]
            all_labels = h5['labels'][:, 0].astype(np.int64, copy=False)
            all_rel_pairs = h5['relationships'][:].astype(np.int64, copy=False)
            all_predicates = h5['predicates'][:, 0].astype(np.int64, copy=False)

            targets = []
            for h5_idx in selected:
                box_start = int(first_box[h5_idx])
                box_end = int(last_box[h5_idx]) + 1
                rel_start = int(first_rel[h5_idx])
                rel_end = int(last_rel[h5_idx]) + 1
                info = image_data[h5_idx]
                width, height = int(info['width']), int(info['height'])

                boxes = all_boxes[box_start:box_end].copy()
                # Official load_graphs conversion: cx,cy,w,h -> xyxy at 1024,
                # then get_groundtruth rescales by max(original width,height).
                boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2.0
                boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
                boxes = boxes.astype(np.float32)
                boxes *= float(max(width, height)) / 1024.0
                boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0.0, width - 1.0)
                boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0.0, height - 1.0)

                local_pairs = all_rel_pairs[rel_start:rel_end] - box_start
                relations = np.column_stack((
                    local_pairs,
                    all_predicates[rel_start:rel_end],
                )).astype(np.int64, copy=False)
                if (
                    np.any(local_pairs < 0)
                    or np.any(local_pairs >= (box_end - box_start))
                ):
                    raise RuntimeError(
                        f"official VG relation index outside image boxes: {info['image_id']}"
                    )

                targets.append({
                    'boxes': boxes,
                    'labels': all_labels[box_start:box_end].copy(),
                    'relations': relations,
                    'width': width,
                    'height': height,
                })

        print(
            "[VisualGenome] Using official H5 test targets: "
            f"{len(targets)} images, "
            f"{sum(len(t['relations']) for t in targets)} raw relations"
        )
        return targets

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]

        if self.use_official_vg_h5_eval:
            official = self._official_vg_eval_targets[idx]
            width, height = img.size
            if (width, height) != (official['width'], official['height']):
                raise RuntimeError(
                    f"VG image size mismatch for {image_id}: image={(width, height)}, "
                    f"official={(official['width'], official['height'])}"
                )
            eval_boxes = torch.from_numpy(official['boxes'].copy()).float()
            eval_labels = torch.from_numpy(official['labels'].copy()).long()
            eval_rel_annotations = torch.from_numpy(
                official['relations'].copy()
            ).long()

            # Official __getitem__ uses get_groundtruth(evaluation=False),
            # which removes clipped boxes with zero width/height. The evaluator
            # separately calls get_groundtruth(evaluation=True), which retains
            # those boxes and raw relation tuples. Keep both views.
            keep = (
                (eval_boxes[:, 3] > eval_boxes[:, 1])
                & (eval_boxes[:, 2] > eval_boxes[:, 0])
            )
            boxes = eval_boxes[keep]
            labels = eval_labels[keep]
            old_to_new = torch.full(
                (len(eval_boxes),), -1, dtype=torch.long
            )
            old_to_new[keep] = torch.arange(int(keep.sum()), dtype=torch.long)
            rel_keep = (
                keep[eval_rel_annotations[:, 0]]
                & keep[eval_rel_annotations[:, 1]]
            )
            rel_annotations = eval_rel_annotations[rel_keep].clone()
            rel_annotations[:, 0] = old_to_new[rel_annotations[:, 0]]
            rel_annotations[:, 1] = old_to_new[rel_annotations[:, 1]]
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([image_id]),
                'area': (
                    (boxes[:, 2] - boxes[:, 0] + 1.0)
                    * (boxes[:, 3] - boxes[:, 1] + 1.0)
                ),
                'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
                'orig_size': torch.as_tensor([height, width]),
                'size': torch.as_tensor([height, width]),
                'rel_annotations': rel_annotations,
                'eval_boxes': eval_boxes,
                'eval_labels': eval_labels,
                'eval_rel_annotations': eval_rel_annotations,
            }
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            return img, target

        # rel.json keys can be str or int depending on the split;
        # try str first, then int as fallback.
        rel_key = str(image_id)
        if rel_key not in self.rel_annotations and isinstance(image_id, int):
            rel_key = image_id
        rel_target = self.rel_annotations[rel_key]

        target = {'image_id': image_id, 'annotations': target, 'rel_annotations': rel_target}

        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        # TODO add relation gt in the target
        rel_annotations = target['rel_annotations']

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        # TODO add relation gt in the target
        target['rel_annotations'] = torch.tensor(rel_annotations)

        return image, target


def make_coco_transforms(image_set, test_min_size=800, test_max_size=1333):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    #T.RandomSizeCrop(384, 600), # TODO: cropping causes that some boxes are dropped then no tensor in the relation part! What should we do?
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([test_min_size], max_size=test_max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    
    ann_path = args.data_root
    img_folder = args.data_root + 'images/'

    if image_set == 'train':
        ann_file = ann_path + 'train.json'
    elif image_set == 'val':
        if args.eval:
            ann_file = ann_path + 'test.json'
        else:
            ann_file = ann_path + 'val.json'

    use_official_h5_eval = bool(
        image_set == 'val'
        and getattr(args, 'eval', False)
        and getattr(args, 'vg_use_official_h5_eval_annotations', False)
    )
    dataset = CocoDetection(img_folder, ann_file,
                            transforms=make_coco_transforms(
                                image_set,
                                test_min_size=getattr(args, 'test_min_size', 800),
                                test_max_size=getattr(args, 'test_max_size', 1333)),
                            return_masks=False,
                            use_official_vg_h5_eval=use_official_h5_eval,
                            vg_h5_file=os.path.join(
                                args.data_root, 'VG-SGG-with-attri.h5'),
                            vg_image_data_file=os.path.join(
                                args.data_root, 'image_data.json'))
    return dataset
