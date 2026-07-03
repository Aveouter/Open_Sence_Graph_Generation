"""
RA-SGG method wrapper for PyTorch Lightning.

Strictly aligned with the official RA-SGG (ReTAGPENet) training pipeline:
  - Extracts per-object ROI features + union ROI features
  - Passes rel_labels (with 0 for background pairs) to the model
  - Handles add_data (mixup labels) for loss computation
  - Inference: standard PE-Net cosine similarity (no retrieval)

Reference: https://github.com/KanghoonYoon/torch-rasgg
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .motifs_method import Motifs_Method
from src.models.ra_sgg import build_ra_sgg


class RA_SGG_Method(Motifs_Method):
    """RA-SGG Lightning method — extends Motifs_Method for two-stage data flow.

    Key differences from Motifs_Method:
      1. Extracts union ROI features for each subject-object pair.
      2. Builds rel_labels tensor from rel_annotations (matching official format).
      3. Passes add_data['rel_labels_onehot'/'mixup_labels'] for loss computation.
      4. Training: model runs retrieval + augmentation pipeline.
      5. Inference: model runs standard PE-Net prototype-based classification.
    """

    def __init__(self, **args):
        super().__init__(**args)
        self._cur_iter = 0
        # Replace shared VisualFeatureExtractor with FPN extractor (4096-dim output)
        # — does NOT modify backbone.py, other methods unaffected
        from src.models.ra_sgg import build_fpn_extractor
        self._visual_extractor = build_fpn_extractor(self.hparams)

    def _build_model(self, **args):
        return build_ra_sgg(self.hparams)

    def _build_criterion(self, **args):
        return _RASGGCriterion(
            num_predicates=args.get('rel_nums', 51),
            use_mixup=args.get('ra_sgg_mixup', True),
        )

    # ---------- forward (overrides Motifs_Method.forward) ----------

    def forward(self, images, targets=None, **kwargs):
        """Forward pass — uses self.training to distinguish train vs eval.

        Training (self.training=True):
          PredCls/SGCls/SGDet: all use GT boxes/labels.
        Eval (self.training=False):
          PredCls/SGCls: GT boxes/labels.
          SGDet: RPN+box_head detector → proposal boxes/labels.
        """
        is_training = self.training
        eval_mode = getattr(self.hparams, 'eval_mode', 'predcls')

        # ---- SGDet eval: detect objects from scratch ----
        if eval_mode == 'sgdet' and not is_training:
            return self._forward_sgdet_eval(images, targets)

        # ---- PredCls / SGCls / SGDet-train: use GT boxes ----
        all_outputs = []
        if targets is None:
            targets = [{}] * (len(images) if isinstance(images, (list, tuple)) else 1)

        boxes_list = [t.get("boxes") for t in targets]
        labels_list = [t.get("labels") for t in targets]
        image_sizes = [t.get("size", t.get("orig_size")) for t in targets]

        # Build image list + device
        images_list = self._image_list_from_batch(images)
        if not images_list:
            raise NotImplementedError("Cannot infer image list from batch")

        device = next(self._visual_extractor.parameters()).device
        images_list = [img.to(device) for img in images_list]
        # Infer missing sizes from image shapes (needed for CI / synthetic data)
        for idx in range(len(targets)):
            if image_sizes[idx] is None:
                if idx < len(images_list):
                    image_sizes[idx] = torch.tensor(
                        images_list[idx].shape[-2:], dtype=torch.float32, device=device)
            if boxes_list[idx] is None or boxes_list[idx].numel() == 0:
                # Needs at least 2 objects — create dummy boxes for synthetic test
                boxes_list[idx] = torch.tensor(
                    [[0.3, 0.3, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]], device=device)
            if labels_list[idx] is None:
                labels_list[idx] = torch.tensor([1, 2], device=device)

        roi_feats_list, feature_maps = self._visual_extractor(
            images_list, boxes_list, image_sizes, return_feature_maps=True)

        pair_indices_list, rel_labels_list = self._build_rel_labels(
            boxes_list, targets)

        union_feats_list = self._visual_extractor.extract_union_features(
            feature_maps, boxes_list, image_sizes, pair_indices_list)

        need_obj_preds = eval_mode in ('sgcls', 'sgdet')
        for i, (roi_feat, union_feat, pair_idx, rel_lab, box, lab) in enumerate(
                zip(roi_feats_list, union_feats_list, pair_indices_list,
                    rel_labels_list, boxes_list, labels_list)):
            out = self.model(
                roi_features=roi_feat, union_features=union_feat,
                labels=lab, boxes=box, rel_labels=rel_lab,
                return_obj_preds=need_obj_preds, cur_iter=self._cur_iter)
            all_outputs.append(out)

        # ---- Batch outputs (same pattern as Motifs_Method) ----
        batched = {
            'rel_logits': [o['rel_logits'] for o in all_outputs],
            'entity_dists': [o['entity_dists'] for o in all_outputs],
            'pair_indices': [o['pair_indices'] for o in all_outputs],
            'sub_boxes': [o['sub_boxes'] for o in all_outputs],
            'obj_boxes': [o['obj_boxes'] for o in all_outputs],
            'obj_labels': [o['obj_labels'] for o in all_outputs],
            'predicate_bg_index': 'first',  # bg class at index 0
            'relation_softmax_scope': 'all',
        }

        # Collect add_losses
        add_losses = {}
        for o in all_outputs:
            for name, value in o.get('add_losses', {}).items():
                if torch.is_tensor(value):
                    add_losses.setdefault(name, []).append(value)
        if add_losses:
            batched['add_losses'] = {
                name: torch.stack(values).mean()
                for name, values in add_losses.items()
            }

        # Collect add_data (mixup labels, etc.)
        add_data = {}
        for o in all_outputs:
            for name, value in o.get('add_data', {}).items():
                if torch.is_tensor(value):
                    add_data.setdefault(name, []).append(value)
        if add_data:
            batched['add_data'] = add_data

        out = {'outputs': batched}

        if is_training and self.criterion is not None:
            loss_dict = self.criterion(batched, targets)
            out['loss_dict'] = loss_dict
            out['loss'] = loss_dict['loss_total']

        return out

    # ---------- SGDet eval path ----------

    def _forward_sgdet_eval(self, images, targets):
        """SGDet eval: RPN → box head → relation predictor."""
        images_list = self._image_list_from_batch(images)
        device = next(self._visual_extractor.parameters()).device
        images_list = [img.to(device) for img in images_list]
        image_sizes = [torch.tensor(img.shape[-2:], dtype=torch.float32, device=device) for img in images_list]

        # Detect objects
        det_boxes, det_labels, det_scores = self._visual_extractor.detect_objects(images_list, image_sizes)

        # Extract FPN features and union features
        all_outputs = []
        for img, boxes, labels, sz in zip(images_list, det_boxes, det_labels, image_sizes):
            if boxes.numel() < 2:
                all_outputs.append({'rel_logits': boxes.new_zeros(0, 51),
                    'entity_dists': boxes.new_zeros(0, 151),
                    'pair_indices': boxes.new_zeros(0, 2, dtype=torch.long),
                    'sub_boxes': boxes.new_zeros(0, 4), 'obj_boxes': boxes.new_zeros(0, 4),
                    'obj_labels': boxes.new_zeros(0, dtype=torch.long),
                    'add_losses': {}, 'add_data': {}})
                continue

            fpn_feats = self._visual_extractor.fpn(self._visual_extractor.backbone(img))
            roi_feats = self._visual_extractor.extract_object_features_from_boxes(
                [fpn_feats], [boxes], [sz])[0]

            from src.models.motifs import generate_object_pairs
            pairs = generate_object_pairs(boxes.size(0), boxes.device)
            rel_labels = torch.zeros(pairs.size(0), dtype=torch.long, device=boxes.device)

            union_feats = self._visual_extractor.extract_union_features(
                [fpn_feats], [boxes], [sz], [pairs])

            out = self.model(roi_features=roi_feats, union_features=union_feats[0] if union_feats else None,
                           labels=labels, boxes=boxes, rel_labels=rel_labels,
                           return_obj_preds=True, cur_iter=self._cur_iter)
            all_outputs.append(out)

        batched = {
            'rel_logits': [o['rel_logits'] for o in all_outputs],
            'entity_dists': [o['entity_dists'] for o in all_outputs],
            'pair_indices': [o['pair_indices'] for o in all_outputs],
            'sub_boxes': [o['sub_boxes'] for o in all_outputs],
            'obj_boxes': [o['obj_boxes'] for o in all_outputs],
            'obj_labels': [o['obj_labels'] for o in all_outputs],
            'obj_logits': [o.get('obj_logits') for o in all_outputs],
            'predicate_bg_index': 'first',
            'relation_softmax_scope': 'all',
        }
        return {'outputs': batched}

    # ---------- Helpers ----------

    def _build_rel_labels(self, boxes_list, targets):
        from src.models.motifs import generate_object_pairs
        pair_indices_list, rel_labels_list = [], []
        for box, target in zip(boxes_list, targets):
            N = box.size(0) if box is not None else 0
            if N < 2:
                pairs = box.new_zeros(0, 2, dtype=torch.long) if box is not None else torch.zeros(0, 2, dtype=torch.long)
                pair_indices_list.append(pairs)
                rel_labels_list.append(pairs.new_zeros(0, dtype=torch.long) if pairs.numel() == 0 else torch.zeros(0, dtype=torch.long))
                continue
            pairs = generate_object_pairs(N, box.device)
            rel_labels = torch.zeros(pairs.size(0), dtype=torch.long, device=box.device)
            rel_anns = target.get('rel_annotations')
            if rel_anns is not None and rel_anns.numel() > 0:
                for ann in rel_anns:
                    s, o, p = int(ann[0]), int(ann[1]), int(ann[2])
                    match = (pairs[:, 0] == s) & (pairs[:, 1] == o)
                    if match.any():
                        rel_labels[match] = p
            pair_indices_list.append(pairs)
            rel_labels_list.append(rel_labels)
        return pair_indices_list, rel_labels_list

    # ---------- training / eval steps ----------

    def training_step(self, batch, batch_idx):
        self._cur_iter = self.global_step
        images, targets = self._split_batch(batch)
        targets = self._move_targets_to_device(targets)

        result = self.forward(images, targets)
        loss_dict = result.get('loss_dict', {})
        total_loss = result.get('loss', torch.tensor(0.0, device=self.device))

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'train_{k}', v, on_step=True, on_epoch=False)

        return total_loss

    @torch.no_grad()
    def _eval_step(self, batch, prefix: str):
        """Evaluation step — uses standard PE-Net inference (no retrieval)."""
        images, targets = self._split_batch(batch)
        targets = self._move_targets_to_device(targets)

        result = self.forward(images, targets)
        outputs = result.get('outputs', {})
        loss_dict = result.get('loss_dict', {})
        total_loss = result.get('loss', torch.tensor(0.0, device=self.device))

        self.log(f'{prefix}_loss', total_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'{prefix}_{k}', v, on_step=False, on_epoch=True,
                         sync_dist=True)

        return outputs, targets, loss_dict, total_loss

    def validation_step(self, batch, batch_idx):
        outputs, targets, loss_dict, total_loss = self._eval_step(batch, 'val')
        self._cache_step_output(self.val_outputs, outputs, targets, loss_dict, total_loss)
        return total_loss

    def test_step(self, batch, batch_idx):
        outputs, targets, loss_dict, total_loss = self._eval_step(batch, 'test')
        self._cache_step_output(self.test_outputs, outputs, targets, loss_dict, total_loss)
        return total_loss


# ============================================================================
# RA-SGG Loss Criterion
# ============================================================================

class _RASGGCriterion(nn.Module):
    """RA-SGG loss evaluator — strictly aligned with official RelationLossComputation.

    Handles:
      - CE loss on relation logits (with soft mixup labels when MIXUP=True)
      - Prototype regularization losses from add_losses

    Reference: official roi_heads/relation_head/loss.py lines 17-113
    """

    def __init__(self, num_predicates: int, use_mixup: bool = True):
        super().__init__()
        self.num_predicates = num_predicates
        # Pure predicate count (excluding background)
        self.num_rel = num_predicates - 1 if num_predicates > 50 else num_predicates
        self.use_mixup = use_mixup

    def _target_predicate_index(self, pred_id: int, logits_dim: int, bg_index) -> int:
        """Map VG predicate IDs (1..50) to model's predicate-logit layout."""
        if logits_dim == self.num_rel:
            return pred_id - 1
        if logits_dim == self.num_rel + 1:
            return pred_id if bg_index == 'first' else pred_id - 1
        return -1

    def forward(self, outputs: dict, targets: list) -> dict:
        total_pred_loss = None
        total_pairs = 0

        rel_logits_list = outputs.get('rel_logits')
        pair_indices_list = outputs.get('pair_indices')
        add_data = outputs.get('add_data', {})
        add_losses = outputs.get('add_losses', {})
        bg_index = outputs.get('predicate_bg_index', 'first')

        if not isinstance(rel_logits_list, (list, tuple)):
            rel_logits_list = [rel_logits_list]
            pair_indices_list = [pair_indices_list] if pair_indices_list is not None else None

        # Mixup labels (from add_data) or standard labels (from targets)
        mixup_labels_list = None
        if self.use_mixup and 'rel_labels_onehot' in add_data:
            mixup_labels_list = add_data['rel_labels_onehot']
            if not isinstance(mixup_labels_list, (list, tuple)):
                mixup_labels_list = [mixup_labels_list]
        elif not self.use_mixup and 'mixup_labels' in add_data:
            mixup_labels_hard = add_data['mixup_labels']
            if not isinstance(mixup_labels_hard, (list, tuple)):
                mixup_labels_hard = [mixup_labels_hard]

        for i, t in enumerate(targets):
            if i >= len(rel_logits_list):
                break

            rel_logits = rel_logits_list[i]  # cosine similarity logits [P, C]
            pair_indices = pair_indices_list[i] if pair_indices_list is not None else None

            if rel_logits is None or rel_logits.numel() == 0:
                continue

            if rel_logits.dim() == 3:
                rel_logits = rel_logits.squeeze(0)
            if isinstance(pair_indices, torch.Tensor) and pair_indices.dim() == 3:
                pair_indices = pair_indices.squeeze(0)

            P = rel_logits.size(0)
            C = rel_logits.size(-1)

            # ---- Build GT labels from rel_annotations ----
            gt_preds = torch.full((P,), -1, dtype=torch.long, device=rel_logits.device)
            rel_anns = t.get('rel_annotations')
            if rel_anns is not None and rel_anns.numel() > 0:
                pair_to_pred = {}
                for ann in rel_anns:
                    s, o, p = int(ann[0]), int(ann[1]), int(ann[2])
                    p_idx = self._target_predicate_index(p, C, bg_index)
                    if 0 <= p_idx < C:
                        pair_to_pred[(s, o)] = p_idx

                if pair_indices is not None:
                    for p_idx in range(P):
                        s = int(pair_indices[p_idx, 0])
                        o = int(pair_indices[p_idx, 1])
                        if (s, o) in pair_to_pred:
                            gt_preds[p_idx] = pair_to_pred[(s, o)]

            # ---- Compute loss ----
            valid_mask = gt_preds >= 0
            if valid_mask.any():
                if self.use_mixup and mixup_labels_list is not None and i < len(mixup_labels_list):
                    # CE with soft mixup labels (one-hot vectors with Beta mixing)
                    soft_labels = mixup_labels_list[i].to(rel_logits.device)
                    if soft_labels.dim() == 3:
                        soft_labels = soft_labels.squeeze(0)
                    # Only compute on annotated pairs
                    soft_labels_valid = soft_labels[valid_mask]
                    if soft_labels_valid.size(-1) != C:
                        # Adjust to match logits dimension
                        if soft_labels_valid.size(-1) < C:
                            padded = torch.zeros(soft_labels_valid.size(0), C,
                                                 device=soft_labels_valid.device)
                            padded[:, :soft_labels_valid.size(-1)] = soft_labels_valid
                            soft_labels_valid = padded
                        elif soft_labels_valid.size(-1) > C:
                            soft_labels_valid = soft_labels_valid[:, :C]
                    # CE with soft labels: -sum(target * log_softmax(input))
                    log_probs = F.log_softmax(rel_logits[valid_mask], dim=-1)
                    pred_loss = -(soft_labels_valid * log_probs).sum(dim=-1).mean()
                else:
                    pred_loss = F.cross_entropy(
                        rel_logits[valid_mask], gt_preds[valid_mask],
                        reduction='mean')

                total_pred_loss = pred_loss if total_pred_loss is None else total_pred_loss + pred_loss
                total_pairs += 1

        if total_pred_loss is None:
            device_t = rel_logits_list[0].device if (
                isinstance(rel_logits_list, (list, tuple)) and len(rel_logits_list) > 0
                and torch.is_tensor(rel_logits_list[0])) else None
            total_pred_loss = torch.tensor(0.0, device=device_t)
            total_pairs = 1

        loss_dict = {
            'loss_predicate': total_pred_loss / max(total_pairs, 1),
        }

        # ---- Add prototype regularization losses from model ----
        if isinstance(add_losses, dict):
            for k, v in add_losses.items():
                if torch.is_tensor(v):
                    loss_dict[f'loss_{k}'] = v

        total_loss = loss_dict['loss_predicate']
        for k, v in loss_dict.items():
            if k != 'loss_predicate' and k != 'loss_total' and torch.is_tensor(v):
                total_loss = total_loss + v

        loss_dict['loss_total'] = total_loss
        return loss_dict
