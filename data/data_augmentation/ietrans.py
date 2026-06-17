"""
IETrans: Instance Exchange based Data Augmentation for SGG.

Reference: "Fine-Grained Scene Graph Generation with Data Transfer"
(Zhou et al., ECCV 2022)

Core idea: For tail predicate classes, exchange subject/object instances
between images to create new training samples, increasing the diversity
of rare predicate occurrences.

Algorithm:
  1. Build a predicate-to-triplet index from the training set
  2. For each tail predicate (below frequency threshold):
     a. Sample two images containing that predicate
     b. Exchange the subject/object instances between the two images
     c. Adjust bounding boxes to fit the new image context
  3. Add augmented samples to the training set

This helps alleviate the long-tail distribution by synthesizing
rare predicate occurrences.
"""
import random
import copy
import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class IETransAugmentation:
    """Instance Exchange based data augmentation for Scene Graph Generation.

    Args:
        pred_frequencies: [num_predicates] tensor/list of predicate frequencies.
        tail_threshold: Frequency percentile below which to consider as tail.
        exchange_prob: Probability of applying IETrans to a qualifying sample.
        max_augmentations: Maximum number of augmented samples to generate per epoch.
        seed: Random seed for reproducibility.
    """

    def __init__(self, pred_frequencies: List[float],
                 tail_threshold: float = 0.2,  # bottom 20% percentile
                 exchange_prob: float = 0.5,
                 max_augmentations: int = 100,
                 seed: int = 42):
        self.pred_frequencies = np.array(pred_frequencies)
        self.num_predicates = len(pred_frequencies)

        # Determine tail predicates
        threshold_val = np.percentile(self.pred_frequencies[self.pred_frequencies > 0],
                                       tail_threshold * 100)
        self.tail_predicates = set(
            np.where(self.pred_frequencies <= threshold_val)[0].tolist())

        self.exchange_prob = exchange_prob
        self.max_augmentations = max_augmentations

        # Index: predicate → list of (image_idx, sub_idx, obj_idx, sub_box, obj_box)
        self.predicate_index: Dict[int, List[Tuple[int, int, int,
                                                     np.ndarray, np.ndarray]]] = defaultdict(list)

        self.rng = np.random.RandomState(seed)

    def build_index(self, dataset) -> None:
        """Build predicate-to-instance index from the training dataset.

        Args:
            dataset: CocoDetection dataset with rel_annotations in targets.
        """
        self.predicate_index.clear()

        for img_idx in range(len(dataset)):
            _, target = dataset[img_idx]
            rel_anns = target.get("rel_annotations")
            boxes = target.get("boxes")
            labels = target.get("labels")

            if rel_anns is None or boxes is None:
                continue

            for ann in rel_anns:
                sub_idx, obj_idx, pred = int(ann[0]), int(ann[1]), int(ann[2])
                if pred in self.tail_predicates:
                    self.predicate_index[pred].append((
                        img_idx, sub_idx, obj_idx,
                        boxes[sub_idx].numpy().copy(),
                        boxes[obj_idx].numpy().copy(),
                        labels[sub_idx].item() if labels is not None else -1,
                    ))

    def _exchange_instances(self, target_a: dict, target_b: dict,
                            sub_a: int, obj_a: int,
                            sub_b: int, obj_b: int,
                            pred: int) -> Tuple[dict, dict]:
        """Exchange instances between two targets.

        Swaps the subject of sample_a with the subject of sample_b.
        The new relation is: (sub_from_b, obj_from_a, pred) for sample_a,
        and (sub_from_a, obj_from_b, pred) for sample_b.

        Args:
            target_a, target_b: Target dicts from two different images.
            sub_a, obj_a: Subject/object indices in target_a.
            sub_b, obj_b: Subject/object indices in target_b.
            pred: Predicate label to assign.

        Returns:
            (augmented_target_a, augmented_target_b) with new rel_annotations.
        """
        aug_a = copy.deepcopy(target_a)
        aug_b = copy.deepcopy(target_b)

        # Add exchanged relation to aug_a
        new_rel_a = torch.tensor([[sub_b, obj_a, pred]], dtype=torch.long)
        if aug_a["rel_annotations"].numel() > 0:
            aug_a["rel_annotations"] = torch.cat(
                [aug_a["rel_annotations"], new_rel_a], dim=0)
        else:
            aug_a["rel_annotations"] = new_rel_a

        # Add exchanged relation to aug_b
        new_rel_b = torch.tensor([[sub_a, obj_b, pred]], dtype=torch.long)
        if aug_b["rel_annotations"].numel() > 0:
            aug_b["rel_annotations"] = torch.cat(
                [aug_b["rel_annotations"], new_rel_b], dim=0)
        else:
            aug_b["rel_annotations"] = new_rel_b

        return aug_a, aug_b

    def augment_sample(self, target: dict, image_idx: int) -> Optional[dict]:
        """Apply IETrans augmentation to a single sample.

        Args:
            target: Target dict for a single image.
            image_idx: Index of the image in the dataset.

        Returns:
            Augmented target dict, or None if augmentation is not applicable.
        """
        if self.rng.random() > self.exchange_prob:
            return None

        rel_anns = target.get("rel_annotations")
        if rel_anns is None or rel_anns.numel() == 0:
            return None

        # Find a tail predicate in this sample
        tail_anns = []
        for ann in rel_anns:
            pred = int(ann[2])
            if pred in self.tail_predicates:
                tail_anns.append(ann)

        if not tail_anns:
            return None

        # Pick a random tail predicate
        ann = tail_anns[self.rng.randint(0, len(tail_anns))]
        sub_idx, obj_idx, pred = int(ann[0]), int(ann[1]), int(ann[2])

        # Find another sample with the same predicate
        candidates = self.predicate_index.get(pred, [])
        candidates = [c for c in candidates if c[0] != image_idx]

        if not candidates:
            return None

        # Pick a random candidate
        donor = candidates[self.rng.randint(0, len(candidates))]
        donor_img_idx, donor_sub, donor_obj, donor_sub_box, donor_obj_box, donor_sub_class = donor

        # Find an object in the current image with the same class as the donor's subject
        target_labels = target.get("labels")
        if target_labels is not None and donor_sub_class >= 0:
            matching_obj = (target_labels == donor_sub_class).nonzero(as_tuple=True)[0]
            if matching_obj.numel() > 0:
                local_sub = matching_obj[self.rng.randint(0, len(matching_obj))].item()
            else:
                return None  # no matching class in this image
        else:
            return None

        # Create augmented target by adding the exchanged relation
        aug_target = copy.deepcopy(target)

        # Add a new relation annotation using a local object of the donor's class
        new_rel = torch.tensor([[local_sub, obj_idx, pred]], dtype=torch.long)
        if aug_target["rel_annotations"].numel() > 0:
            aug_target["rel_annotations"] = torch.cat(
                [aug_target["rel_annotations"], new_rel], dim=0)
        else:
            aug_target["rel_annotations"] = new_rel

        return aug_target

    def augment_batch(self, targets: List[dict],
                      image_indices: List[int]) -> List[dict]:
        """Apply IETrans augmentation to a batch of targets.

        Args:
            targets: List of target dicts for a batch.
            image_indices: List of image indices corresponding to each target.

        Returns:
            (Possibly augmented) list of target dicts.
        """
        augmented = []
        aug_count = 0

        for i, (target, img_idx) in enumerate(zip(targets, image_indices)):
            if aug_count >= self.max_augmentations:
                augmented.append(target)
                continue

            aug = self.augment_sample(target, img_idx)
            if aug is not None:
                augmented.append(aug)
                aug_count += 1
            else:
                augmented.append(target)

        return augmented


def apply_ietrans_to_dataset(dataset, pred_frequencies: List[float],
                              tail_threshold: float = 0.2,
                              exchange_prob: float = 0.5) -> IETransAugmentation:
    """Create and initialize an IETrans augmentation for a dataset.

    Args:
        dataset: CocoDetection dataset.
        pred_frequencies: Per-predicate frequency counts.
        tail_threshold: Percentile for tail class cutoff.
        exchange_prob: Per-sample augmentation probability.

    Returns:
        Initialized IETransAugmentation object.
    """
    aug = IETransAugmentation(
        pred_frequencies=pred_frequencies,
        tail_threshold=tail_threshold,
        exchange_prob=exchange_prob,
    )
    aug.build_index(dataset)
    return aug
