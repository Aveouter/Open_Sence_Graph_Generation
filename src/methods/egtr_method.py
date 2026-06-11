# src/methods/egtr_method.py
"""
EGTR method wrapper — integrates the EGTR model into the OpenSGG framework.

EGTR: Extracting Graph from Transformer for Scene Graph Generation (CVPR 2024)
Uses Deformable DETR backbone + lightweight relation extraction head.
Depends on HuggingFace transformers (DeformableDetrConfig, DeformableDetrFeatureExtractor).
"""

import torch
import pickle
import torch.nn as nn
from .base_method import Base_method

from utils.misc import NestedTensor, nested_tensor_from_tensor_list


def build_egtr(args):
    """
    Build EGTR model from config and pretrained checkpoint.

    Args:
        args: Namespace or SimpleNamespace with:
            - egtr_pretrained_path: path to pretrained DeformableDETR checkpoint
            - architecture: HuggingFace model name (default: 'SenseTime/deformable-detr')
            - entity_nums: number of object classes (including bg, e.g. 151)
            - rel_nums: number of relation classes (including bg, e.g. 51)
            - num_queries: number of object queries (default: 200)
            - use_freq_bias: whether to use frequency bias
            - logit_adjustment: whether to apply logit adjustment at inference
            - logit_adj_tau: tau for logit adjustment
            - freq_bias_eps: epsilon for frequency bias
            - use_log_softmax: whether to use log_softmax for freq bias
            - rel_loss_coef: relation loss coefficient
            - connectivity_loss_coef: connectivity loss coefficient
            - bbox_loss_coef: bbox loss coefficient
            - giou_loss_coef: giou loss coefficient
            - ce_loss_coef: cross-entropy loss coefficient
            - eos_coef: eos coefficient for no-object class
            - focal_alpha: focal loss alpha
            - rel_sample_negatives: number of negative samples per positive relation
            - rel_sample_nonmatching: number of non-matching samples
            - smoothing: label smoothing for matcher
            - aux_loss: whether to use auxiliary losses
            - dec_layers: number of decoder layers
            - device: device string
    """
    from src.modules.egtr.deformable_detr import DeformableDetrConfig
    from src.modules.egtr.deformable_detr import (
        DeformableDetrFeatureExtractor,
        DeformableDetrFeatureExtractorWithAugmentorNoCrop,
    )
    from src.modules.egtr.egtr import DetrForSceneGraphGeneration

    device = torch.device(getattr(args, 'device', 'cuda'))
    architecture = getattr(args, 'architecture', 'SenseTime/deformable-detr')

    # Try to load config from pretrained, fall back to creating locally
    pretrained_path = getattr(args, 'egtr_pretrained_path', None)
    try:
        if pretrained_path is not None:
            config = DeformableDetrConfig.from_pretrained(pretrained_path)
        else:
            config = DeformableDetrConfig.from_pretrained(architecture)
    except (OSError, ValueError, EnvironmentError, pickle.UnpicklingError):
        # Offline mode: create config from scratch with defaults
        config = DeformableDetrConfig(
            backbone='resnet50',
            d_model=256,
            encoder_layers=6,
            decoder_layers=6,
            encoder_attention_heads=8,
            decoder_attention_heads=8,
            encoder_ffn_dim=1024,
            decoder_ffn_dim=1024,
            num_feature_levels=4,
            encoder_n_points=4,
            decoder_n_points=4,
            num_queries=200,
        )

    # Override config with args
    config.num_labels = getattr(args, 'entity_nums', 151)       # num object classes
    config.num_rel_labels = getattr(args, 'rel_nums', 51)        # num relation classes
    config.num_queries = getattr(args, 'num_queries', 200)
    config.use_freq_bias = getattr(args, 'use_freq_bias', True)
    config.logit_adjustment = getattr(args, 'logit_adjustment', False)
    config.logit_adj_tau = getattr(args, 'logit_adj_tau', 0.3)
    config.freq_bias_eps = getattr(args, 'freq_bias_eps', 1e-12)
    config.use_log_softmax = getattr(args, 'use_log_softmax', False)
    config.rel_loss_coefficient = getattr(args, 'rel_loss_coef', 15.0)
    config.connectivity_loss_coefficient = getattr(args, 'connectivity_loss_coef', 30.0)
    config.bbox_loss_coefficient = getattr(args, 'bbox_loss_coef', 5.0)
    config.giou_loss_coefficient = getattr(args, 'giou_loss_coef', 2.0)
    config.ce_loss_coefficient = getattr(args, 'ce_loss_coef', 2.0)
    config.eos_coefficient = getattr(args, 'eos_coef', 0.1)
    config.focal_alpha = getattr(args, 'focal_alpha', 0.25)
    config.rel_sample_negatives = getattr(args, 'rel_sample_negatives', 80)
    config.rel_sample_nonmatching = getattr(args, 'rel_sample_nonmatching', 80)
    config.rel_sample_negatives_largest = getattr(args, 'rel_sample_negatives_largest', True)
    config.rel_sample_nonmatching_largest = getattr(args, 'rel_sample_nonmatching_largest', True)
    config.smoothing = getattr(args, 'smoothing', 1e-14)
    config.auxiliary_loss = getattr(args, 'aux_loss', True)
    config.decoder_layers = getattr(args, 'dec_layers', 6)

    # Build feature extractor (for image preprocessing during data loading)
    min_size = getattr(args, 'min_size', 800)
    max_size = getattr(args, 'max_size', 1333)
    use_augment = getattr(args, 'use_augment', False)
    if use_augment:
        feature_extractor = DeformableDetrFeatureExtractorWithAugmentorNoCrop(
            size=min_size, max_size=max_size
        )
    else:
        feature_extractor = DeformableDetrFeatureExtractor(
            size=min_size, max_size=max_size
        )

    # Frequency bias matrix (computed from training data)
    fg_matrix = getattr(args, 'fg_matrix', None)

    # Build model
    try:
        model = DetrForSceneGraphGeneration.from_pretrained(
            pretrained_path if pretrained_path else architecture,
            config=config,
            ignore_mismatched_sizes=True,
            fg_matrix=fg_matrix,
        )
    except (OSError, ValueError, EnvironmentError, pickle.UnpicklingError):
        # Offline: instantiate model directly without pretrained weights
        model = DetrForSceneGraphGeneration(config, fg_matrix=fg_matrix)

    model.to(device)

    # Load custom checkpoint if provided (handles size mismatches).
    # When using BaseExperiment, ckpt loading is also done in
    # BaseExperiment.test() as the authoritative step; this is a
    # convenience path for standalone / non-Lightning usage.
    ckpt_path = getattr(args, 'ckpt_path', None)
    if ckpt_path:
        from src.exp import BaseExperiment
        state_dict = BaseExperiment._load_checkpoint_state_dict(ckpt_path)
        BaseExperiment._adapt_state_dict(state_dict, model)

    return model, feature_extractor


class EGTR_Method(Base_method):
    """
    EGTR method wrapper for PyTorch Lightning.

    Training batch convention:
        batch = (images, targets)
          images: Tensor[B, 3, H, W] or list[Tensor(3, H, W)]
          targets: list[dict], each dict contains:
            - "labels": LongTensor[num_obj] — object class labels
            - "boxes": FloatTensor[num_obj, 4] — boxes in (cx, cy, w, h), normalized
            - "rel_annotations": LongTensor[num_rel, 3] — (sub_idx, obj_idx, rel_label)
    """

    def __init__(self, **args):
        super().__init__(**args)
        self.feature_extractor = None

    def _build_criterion(self, **args):
        """EGTR criterion is built inside _build_model. Skip base class construction."""
        return None

    def _build_model(self, **args):
        model, feature_extractor = build_egtr(self.hparams)
        self.feature_extractor = feature_extractor
        # EGTR's criterion is built internally during forward when labels are provided
        self.criterion = None
        return model

    # ---------- forward / predict ----------

    def forward(self, images, targets=None, **kwargs):
        """
        Forward pass for EGTR.

        Args:
            images: Tensor or NestedTensor of batched images
            targets: list[dict] or None

        Returns:
            dict with keys: outputs, loss_dict (if targets provided), loss (if targets provided)
        """
        if isinstance(images, NestedTensor):
            pixel_values = images.tensors
            pixel_mask = images.mask
        elif isinstance(images, torch.Tensor):
            pixel_values = images
            pixel_mask = None
        else:
            samples = nested_tensor_from_tensor_list(images)
            pixel_values = samples.tensors
            pixel_mask = samples.mask

        pixel_values = pixel_values.to(self.device)
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(self.device)

        result = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=targets,
            output_attention_states=True,
            output_hidden_states=True,
        )

        out = {
            'outputs': {
                'pred_logits': result.logits,
                'pred_boxes': result.pred_boxes,
                'pred_rel': result.pred_rel,
                'pred_connectivity': result.pred_connectivity,
            }
        }

        if targets is not None and result.loss_dict is not None:
            out['loss_dict'] = result.loss_dict
            out['loss'] = result.loss

        return out

    # ---------- training / validation / test ----------

    def training_step(self, batch, batch_idx):
        images, targets_orig = self._egtr_extract_batch(batch)
        targets_orig = self._move_targets_to_device(targets_orig)
        targets = self._adapt_targets(targets_orig)

        result = self.forward(images, targets)
        loss_dict = result.get('loss_dict', {})
        total_loss = result.get('loss', torch.tensor(0.0, device=self.device))

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'train_{k}', v, on_step=True, on_epoch=False)

        return total_loss

    def _egtr_to_compact(self, outputs_list, targets_list):
        """Convert EGTR dense per-query outputs to compact per-relation format.

        dense:  pred_rel [Q,Q,P]≈7.8MB, pred_boxes [Q,4], pred_logits [Q,C]
        compact: rel_scores [R,P]≈0.004MB (R=GT relations≈20)
        """
        import numpy as np
        from utils.box_ops import box_cxcywh_to_xyxy, box_iou, rescale_bboxes

        # Normalize targets_list to always be a list of dicts
        if isinstance(targets_list, (tuple, list)):
            targets_list = list(targets_list)
        else:
            targets_list = [targets_list]

        compact = []
        for out, tgt in zip(outputs_list, targets_list):
            if not isinstance(tgt, dict):
                continue  # skip non-dict entries (e.g. from list-format batch)

            pred_boxes = out['pred_boxes'].detach().float().cpu()
            pred_logits = out['pred_logits'].detach().float().cpu()
            pred_rel = out['pred_rel'].detach().float().cpu()
            # Remove all leading batch/singleton dims until exactly 2D/3D
            while pred_boxes.dim() > 2 and pred_boxes.shape[0] == 1:
                pred_boxes = pred_boxes.squeeze(0)
                pred_logits = pred_logits.squeeze(0)
                pred_rel = pred_rel.squeeze(0)
            # Safety: if still >2D after squeeze, flatten remaining batch dims
            if pred_boxes.dim() > 2:
                pred_boxes = pred_boxes.reshape(-1, pred_boxes.shape[-1])
                pred_logits = pred_logits.reshape(-1, pred_logits.shape[-1])
                pred_rel = pred_rel.reshape(-1, -1, pred_rel.shape[-1])

            # Normalize gt_boxes to 2D [N,4]
            gt_boxes = tgt['boxes'].detach().float().cpu()
            while gt_boxes.dim() > 2:
                gt_boxes = gt_boxes.squeeze(0)
            if gt_boxes.dim() == 1:
                gt_boxes = gt_boxes.unsqueeze(0)

            gt_rels = tgt['rel_annotations'].detach().cpu().numpy().astype(np.int64)
            if gt_rels.ndim == 1:
                gt_rels = gt_rels.reshape(-1, 3) if gt_rels.size > 0 else np.zeros((0, 3), dtype=np.int64)
            R = gt_rels.shape[0]
            num_preds = pred_rel.shape[-1] - 1  # exclude bg class at index 0
            if R == 0:
                compact.append({'rel_scores': np.zeros((0, num_preds), dtype=np.float32),
                                'sub_boxes': np.zeros((0, 4), dtype=np.float32),
                                'obj_boxes': np.zeros((0, 4), dtype=np.float32),
                                'sub_scores': np.zeros(0, dtype=np.float32),
                                'obj_scores': np.zeros(0, dtype=np.float32),
                                'sub_classes': np.zeros(0, dtype=np.int64),
                                'obj_classes': np.zeros(0, dtype=np.int64)})
                continue

            # Guard: if pred_boxes has fewer entries than matched GT objects
            if pred_boxes.shape[0] == 0:
                compact.append({'rel_scores': np.zeros((R, num_preds), dtype=np.float32),
                                'sub_boxes': np.zeros((R, 4), dtype=np.float32),
                                'obj_boxes': np.zeros((R, 4), dtype=np.float32),
                                'sub_scores': np.zeros(R, dtype=np.float32),
                                'obj_scores': np.zeros(R, dtype=np.float32),
                                'sub_classes': np.zeros(R, dtype=np.int64),
                                'obj_classes': np.zeros(R, dtype=np.int64)})
                continue

            # IoU matching: pred queries → GT objects
            # pred_boxes: cxcywh normalized [0,1] (model output, already sigmoid-ed)
            # gt_boxes: cxcywh normalized [0,1] (from CocoDetection → Normalize transform)
            # Both are in the SAME normalized coordinate space — just convert format.
            pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)       # [Q,4] xyxy, normalized [0,1]
            gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)            # [M,4] xyxy, normalized [0,1]

            iou, _ = box_iou(pred_xyxy, gt_xyxy)
            # Guard NaN/Inf (degenerate boxes, empty intersections)
            iou = torch.where(iou.isfinite(), iou, torch.zeros_like(iou))
            matched = torch.argmax(iou, dim=0)

            rel_scores = np.zeros((R, num_preds), dtype=np.float32)
            sub_boxes = np.zeros((R, 4), dtype=np.float32)
            obj_boxes = np.zeros((R, 4), dtype=np.float32)
            sub_labels = np.zeros(R, dtype=np.int64)
            obj_labels = np.zeros(R, dtype=np.int64)
            pr = pred_rel.cpu().numpy()
            orig_size = tgt['orig_size'].detach().cpu() if torch.is_tensor(tgt['orig_size']) else tgt['orig_size']
            orig_wh = torch.flip(torch.as_tensor(orig_size), dims=[0])

            for r in range(R):
                si, oi = int(gt_rels[r, 0]), int(gt_rels[r, 1])
                if si >= len(matched) or oi >= len(matched):
                    continue
                sq, oq = int(matched[si]), int(matched[oi])
                # pred_rel convention: index 0 = bg/no-relation, indices 1..P-1 = 50 predicates.
                # Skip bg so softmax is computed over the 50 predicate classes only.
                sv = pr[sq, oq, 1:].astype(np.float64)
                sv -= sv.max(); sv = np.exp(sv) / np.exp(sv).sum()
                rel_scores[r] = sv.astype(np.float32)
                sub_boxes[r] = rescale_bboxes(pred_boxes[sq].unsqueeze(0), orig_wh).squeeze(0).numpy()
                obj_boxes[r] = rescale_bboxes(pred_boxes[oq].unsqueeze(0), orig_wh).squeeze(0).numpy()
                sub_labels[r] = int(torch.softmax(pred_logits[sq, :-1], dim=-1).argmax()) + 1
                obj_labels[r] = int(torch.softmax(pred_logits[oq, :-1], dim=-1).argmax()) + 1

            compact.append({'rel_scores': rel_scores, 'sub_boxes': sub_boxes,
                            'obj_boxes': obj_boxes, 'sub_scores': np.ones(R, dtype=np.float32),
                            'obj_scores': np.ones(R, dtype=np.float32),
                            'sub_classes': sub_labels, 'obj_classes': obj_labels})
        return compact

    def _cache_step(self, store, outputs, targets_orig, loss_dict, total_loss):
        # Normalize targets_orig to a list of dicts
        if isinstance(targets_orig, (tuple, list)):
            targets_list = list(targets_orig)
        else:
            targets_list = [targets_orig]
        compact = self._egtr_to_compact([outputs], targets_list)
        if len(compact) == 0:
            return
        store.append({
            'outputs': {k: [v] for k, v in compact[0].items()},
            'targets': [{kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv)
                         for kk, vv in t.items() if kk not in {'rel', 'iscrowd', 'area'}}
                        for t in targets_list if isinstance(t, dict)],
            'loss_dict': {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in loss_dict.items()},
            'total_loss': total_loss.detach().cpu() if torch.is_tensor(total_loss) else total_loss,
        })

    def _egtr_extract_batch(self, batch):
        """Extract (images, targets) from batch, handling all known formats.

        Known formats:
          (NestedTensor, list[dict])      — standard collate_fn
          (Tensor, list[dict])            — raw collate
          [(Tensor, dict), ...]           — list of per-sample pairs
          [Tensor, dict, Tensor, dict...] — flat alternating list
        """
        # Format A: (images, targets) tuple
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            first, second = batch[0], batch[1]
            # (NestedTensor, list_of_dicts) or (Tensor, list_of_dicts)
            has_tensors_attr = hasattr(first, 'tensors')
            is_tensor = isinstance(first, torch.Tensor)
            if has_tensors_attr or is_tensor:
                if isinstance(second, (list, tuple)):
                    # Only treat as targets if entries look like dicts
                    if all(isinstance(x, dict) for x in second):
                        return first, list(second)
                # second might be a single dict (batch_size=1 unwrapped)
                if isinstance(second, dict):
                    return first, [second]

        # Format B: list of (img, target) pairs — flatten
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            if isinstance(batch[0], (list, tuple)) and len(batch[0]) == 2:
                images = []
                targets = []
                for b in batch:
                    if isinstance(b, (list, tuple)) and len(b) >= 2:
                        images.append(b[0])
                        if isinstance(b[1], dict):
                            targets.append(b[1])
                if len(images) == len(targets) and len(images) > 0:
                    return images, targets

        # Format C: flat list [img, target, img, target, ...]
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images = [batch[i] for i in range(0, len(batch), 2)
                      if i + 1 < len(batch) and isinstance(batch[i+1], dict)]
            targets = [batch[i+1] for i in range(0, len(batch), 2)
                       if i + 1 < len(batch) and isinstance(batch[i+1], dict)]
            if len(images) == len(targets) and len(images) > 0:
                return images, targets

        raise TypeError(f"Unsupported EGTR batch format: {type(batch)}. "
                        f"Expected (images, targets) tuple or list of (img, target) pairs.")

    def validation_step(self, batch, batch_idx):
        images, targets_orig = self._egtr_extract_batch(batch)
        targets_orig = self._move_targets_to_device(targets_orig)
        targets = self._adapt_targets(targets_orig)
        result = self.forward(images, targets)
        loss_dict = result.get('loss_dict', {})
        total_loss = result.get('loss', torch.tensor(0.0, device=self.device))
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'val_{k}', v, on_step=False, on_epoch=True, sync_dist=True)
        self._cache_step(self.val_outputs, result['outputs'], targets_orig, loss_dict, total_loss)
        return total_loss

    def test_step(self, batch, batch_idx):
        images, targets_orig = self._egtr_extract_batch(batch)
        targets_orig = self._move_targets_to_device(targets_orig)
        targets = self._adapt_targets(targets_orig)
        result = self.forward(images, targets)
        loss_dict = result.get('loss_dict', {})
        total_loss = result.get('loss', torch.tensor(0.0, device=self.device))
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'test_{k}', v, on_step=False, on_epoch=True, sync_dist=True)
        self._cache_step(self.test_outputs, result['outputs'], targets_orig, loss_dict, total_loss)
        return total_loss

    # ---------- helpers ----------

    def _adapt_targets(self, targets):
        """
        Adapt target format from OpenSGG convention to EGTR convention.

        OpenSGG: dict with 'labels', 'boxes', 'rel_annotations' (sparse [N,3])
        EGTR: dict with 'class_labels', 'boxes', 'rel' (dense [num_queries, num_queries, num_rel])
        """
        adapted = []
        for t in targets:
            at = {}
            # Map keys
            if 'labels' in t:
                at['class_labels'] = t['labels']
            elif 'class_labels' in t:
                at['class_labels'] = t['class_labels']

            if 'boxes' in t:
                at['boxes'] = t['boxes']

            if 'image_id' in t:
                at['image_id'] = t['image_id']

            if 'orig_size' in t:
                at['orig_size'] = t['orig_size']
            if 'size' in t:
                at['size'] = t['size']

            if 'area' in t:
                at['area'] = t['area']
            if 'iscrowd' in t:
                at['iscrowd'] = t['iscrowd']

            # Convert sparse rel_annotations to dense rel matrix.
            # MUST be padded to num_queries: EGTR loss_relations indexes
            # target["rel"] with full_target_index (length num_object_queries),
            # which includes dummy slots beyond the GT objects.
            model_config = getattr(self.model, 'config', None)
            num_queries = getattr(model_config, 'num_queries', 200) if model_config is not None else 200
            num_rel_labels = getattr(model_config, 'num_rel_labels', 51) if model_config is not None else 51
            pad_size = max(num_queries, len(at.get('class_labels', at.get('labels', []))))

            if 'rel_annotations' in t and 'class_labels' in at:
                rel_matrix = torch.zeros(pad_size, pad_size, num_rel_labels,
                                         device=t.get('labels', t.get('class_labels')).device)
                for ann in t['rel_annotations']:
                    s, o, p = int(ann[0]), int(ann[1]), int(ann[2])
                    if 0 <= s < pad_size and 0 <= o < pad_size and 0 <= p < num_rel_labels:
                        rel_matrix[s, o, p] = 1.0
                at['rel'] = rel_matrix
            elif 'rel' in t:
                at['rel'] = t['rel']
            else:
                # Create empty rel matrix padded to num_queries
                at['rel'] = torch.zeros(pad_size, pad_size, num_rel_labels,
                                        device=self.device)

            adapted.append(at)
        return adapted
