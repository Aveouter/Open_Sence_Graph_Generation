# src/models/usg.py
"""
USG-Par: Universal Scene Graph Parser (CVPR 2025) — VG-only implementation.

Wu, Fei, Chua — "Universal Scene Graph Generation"

Architecture (5 modules adapted for VG-only):
  1. Modality-Specific Encoder: ResNet backbone + feature projection
  2. Shared Mask Decoder: Object query transformer decoder (DETR-style)
  3. Object Associator: Skipped for single-modality (identity)
  4. Relation Proposal Constructor (RPC): Bidirectional cross-attn + top-K
  5. Relation Decoder: Transformer decoder → predicate classification

Training: Hungarian matching (same as RelTR) for object detection loss +
          cross-entropy for predicate classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List, Tuple

from utils.misc import NestedTensor, nested_tensor_from_tensor_list
from utils import box_ops
from src.modules.layers.backbone import build_backbone
from src.modules.layers.matcher import build_matcher
from src.modules.usg.mask_decoder import USGMaskDecoder, USGObjectHead
from src.modules.usg.rpc import RelationProposalConstructor
from src.modules.usg.relation_decoder import USGRelationDecoder


# ==============================================================================
# USG Model
# ==============================================================================

class USG(nn.Module):
    """USG-Par model for VG-only scene graph generation.

    Args:
        backbone: ResNet backbone from build_backbone()
        mask_decoder: USGMaskDecoder for object query refinement
        object_head: USGObjectHead for box + class prediction
        rpc: RelationProposalConstructor for pair selection
        relation_decoder: USGRelationDecoder for predicate prediction
        num_classes: number of object classes (151)
        num_predicates: number of predicate classes (51)
        num_queries: number of object queries (200)
        aux_loss: use auxiliary losses from intermediate decoder layers
    """

    def __init__(
        self,
        backbone: nn.Module,
        mask_decoder: USGMaskDecoder,
        object_head: USGObjectHead,
        rpc: RelationProposalConstructor,
        relation_decoder: USGRelationDecoder,
        num_classes: int = 151,
        num_predicates: int = 51,
        num_queries: int = 200,
        aux_loss: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.mask_decoder = mask_decoder
        self.object_head = object_head
        self.rpc = rpc
        self.relation_decoder = relation_decoder
        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.num_queries = num_queries
        self.aux_loss = aux_loss
        self.hidden_dim = mask_decoder.d_model

        # Input projection: backbone channels → d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, self.hidden_dim, kernel_size=1)

    def forward(
        self,
        samples: NestedTensor,
        targets: Optional[List[Dict]] = None,
    ) -> Dict[str, Tensor]:
        """
        Args:
            samples: NestedTensor with .tensors [B, 3, H, W] and .mask [B, H, W]
            targets: optional list of target dicts (only needed for training)

        Returns:
            dict with keys for evaluation:
                pred_logits, pred_boxes, sub_logits, obj_logits,
                sub_boxes, obj_boxes, rel_logits
        """
        B = samples.tensors.shape[0]
        device = samples.tensors.device

        # ---- Encoder: backbone + feature projection ----
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()  # src: [B, C, H, W], mask: [B, H, W]

        # Flatten and project
        src_proj = self.input_proj(src)  # [B, d_model, H, W]
        C_src, H_src, W_src = src_proj.shape[1], src_proj.shape[2], src_proj.shape[3]
        src_flat = src_proj.flatten(2).transpose(1, 2)  # [B, HW, d_model]

        if mask is not None:
            mask_flat = F.interpolate(
                mask[None].float(), size=(H_src, W_src)
            ).to(torch.bool)[0]  # [B, H, W]
            mask_flat = mask_flat.flatten(1)  # [B, HW]
        else:
            mask_flat = None

        pos_flat = pos[-1].flatten(2).transpose(1, 2) if pos[-1] is not None else None
        if pos_flat is None:
            pos_flat = torch.zeros(B, H_src * W_src, self.hidden_dim, device=device)

        # ---- Mask Decoder: refine object queries ----
        hs = self.mask_decoder(src_flat, mask_flat, pos_flat)  # [B, Nq, d_model] or list if aux

        # Handle auxiliary outputs: hs is a list if decoder returns intermediate
        if isinstance(hs, (list, tuple)):
            final_hs = hs[-1] if isinstance(hs[-1], Tensor) else hs[-1]
        else:
            final_hs = hs

        # ---- Object Head: predict boxes and classes ----
        obj_logits, obj_boxes = self.object_head(final_hs)  # [B, Nq, C], [B, Nq, 4]

        # ---- RPC: relation proposal construction ----
        pair_indices, sub_feat, obj_feat, pair_scores = self.rpc(final_hs)

        # ---- Relation Decoder: predicate classification ----
        pred_logits = self.relation_decoder(
            sub_feat, obj_feat, src_flat, mask_flat, pos_flat
        )  # [B, K, num_predicates]

        # ---- Build eval-compatible outputs ----
        K = pair_indices.shape[1]
        sub_idx = pair_indices[:, :, 0]  # [B, K]
        obj_idx = pair_indices[:, :, 1]  # [B, K]

        # Gather sub/obj logits and boxes for selected pairs
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)
        sub_logits = obj_logits[batch_idx, sub_idx]  # [B, K, C]
        obj_logits_k = obj_logits[batch_idx, obj_idx]  # [B, K, C]
        sub_boxes = obj_boxes[batch_idx, sub_idx]       # [B, K, 4]
        obj_boxes_k = obj_boxes[batch_idx, obj_idx]     # [B, K, 4]

        out = {
            'pred_logits': obj_logits,       # [B, Nq, C]
            'pred_boxes': obj_boxes,          # [B, Nq, 4]
            'sub_logits': sub_logits,         # [B, K, C]
            'obj_logits': obj_logits_k,       # [B, K, C]
            'sub_boxes': sub_boxes,           # [B, K, 4]
            'obj_boxes': obj_boxes_k,         # [B, K, 4]
            'rel_logits': pred_logits,        # [B, K, P]
        }

        # Auxiliary outputs for deeper supervision
        if self.aux_loss and isinstance(hs, (list, tuple)) and len(hs) > 1:
            aux_outputs = []
            for layer_hs in hs[:-1]:  # all but last layer
                aux_obj_logits, aux_obj_boxes = self.object_head(layer_hs)
                aux_pair_idx, aux_sub_feat, aux_obj_feat, _ = self.rpc(layer_hs)
                aux_pred_logits = self.relation_decoder(
                    aux_sub_feat, aux_obj_feat, src_flat, mask_flat, pos_flat
                )
                aux_K = aux_pair_idx.shape[1]
                aux_si = aux_pair_idx[:, :, 0]
                aux_oi = aux_pair_idx[:, :, 1]
                aux_batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, aux_K)
                aux_outputs.append({
                    'pred_logits': aux_obj_logits,
                    'pred_boxes': aux_obj_boxes,
                    'sub_logits': aux_obj_logits[aux_batch_idx, aux_si],
                    'obj_logits': aux_obj_logits[aux_batch_idx, aux_oi],
                    'sub_boxes': aux_obj_boxes[aux_batch_idx, aux_si],
                    'obj_boxes': aux_obj_boxes[aux_batch_idx, aux_oi],
                    'rel_logits': aux_pred_logits,
                })
            out['aux_outputs'] = aux_outputs

        return out


# ==============================================================================
# USG Criterion: Hungarian matching + losses (similar to RelTR)
# ==============================================================================

class USGSetCriterion(nn.Module):
    """Loss for USG: Hungarian-matched object detection + predicate classification.

    Architecture follows RelTR's SetCriterion with adaptations for USG:
      - Object detection: CE + L1 + GIoU (same as DETR/RelTR)
      - Predicate classification: CE on matched triplets
    """

    def __init__(
        self,
        num_classes: int,
        num_predicates: int,
        matcher,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1,
    ):
        super().__init__()
        # num_classes and num_predicates from config include bg (e.g. 151, 51)
        # Model outputs num_classes+1 and num_predicates+1 (like RelTR, 152 and 52)
        self.num_classes = num_classes  # 151
        self.num_predicates = num_predicates  # 51
        self.num_classes_out = num_classes + 1  # 152
        self.num_predicates_out = num_predicates + 1  # 52
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.indices = None

        empty_weight = torch.ones(self.num_classes_out)
        empty_weight[self.num_classes] = eos_coef  # bg at index num_classes (151)
        self.register_buffer('empty_weight', empty_weight)

        empty_weight_rel = torch.ones(self.num_predicates_out)
        empty_weight_rel[self.num_predicates] = eos_coef  # bg at index num_predicates (51)
        self.register_buffer('empty_weight_rel', empty_weight_rel)

    def forward(self, outputs: Dict[str, Tensor], targets: List[Dict]) -> Dict[str, Tensor]:
        outputs_no_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_no_aux, targets)
        self.indices = indices

        losses = {}
        losses.update(self._loss_labels(outputs_no_aux, targets, indices))
        losses.update(self._loss_boxes(outputs_no_aux, targets, indices))
        losses.update(self._loss_relations(outputs_no_aux, targets, indices))

        if 'aux_outputs' in outputs:
            for i, aux in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(aux, targets)
                l_dict = self._loss_labels(aux, targets, aux_indices)
                l_dict.update(self._loss_boxes(aux, targets, aux_indices))
                l_dict.update(self._loss_relations(aux, targets, aux_indices))
                losses.update({f'{k}_{i}': v for k, v in l_dict.items()})

        return losses

    def _loss_labels(self, outputs, targets, indices):
        """Object classification loss (entity + sub + obj).

        Background class is at index self.num_classes (151 for VG).
        Output dim is self.num_classes_out (152).
        """
        pred_logits = outputs['pred_logits']    # [B, Nq, 152]
        sub_logits = outputs['sub_logits']      # [B, K, 152]
        obj_logits = outputs['obj_logits']      # [B, K, 152]
        bg_idx = self.num_classes  # 151

        # Entity indices
        entity_idx = self._get_src_permutation_idx(indices[0])
        target_classes_o = torch.cat([
            t['labels'][J] for t, (_, J) in zip(targets, indices[0])
        ])
        target_classes = torch.full(
            pred_logits.shape[:2], bg_idx,
            dtype=torch.int64, device=pred_logits.device
        )
        target_classes[entity_idx] = target_classes_o

        # Relation (triplet) indices: sub/obj labels
        triplet_idx = self._get_src_permutation_idx(indices[1])
        target_sub_o = torch.cat([
            t['labels'][t['rel_annotations'][J, 0]] for t, (_, J) in zip(targets, indices[1])
        ])
        target_obj_o = torch.cat([
            t['labels'][t['rel_annotations'][J, 1]] for t, (_, J) in zip(targets, indices[1])
        ])
        target_sub_classes = torch.full(
            sub_logits.shape[:2], bg_idx,
            dtype=torch.int64, device=sub_logits.device
        )
        target_obj_classes = torch.full(
            obj_logits.shape[:2], bg_idx,
            dtype=torch.int64, device=obj_logits.device
        )
        target_sub_classes[triplet_idx] = target_sub_o
        target_obj_classes[triplet_idx] = target_obj_o

        src_logits = torch.cat([pred_logits, sub_logits, obj_logits], dim=1)
        target_all = torch.cat([target_classes, target_sub_classes, target_obj_classes], dim=1)

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_all,
            self.empty_weight, reduction='none'
        )
        triplet_weight = torch.cat([
            torch.ones(pred_logits.shape[:2], device=pred_logits.device),
            indices[2] * 0.5,
            indices[3] * 0.5,
        ], dim=-1)
        loss_ce = (loss_ce * triplet_weight).sum() / max(
            self.empty_weight[target_all].sum(), 1.0
        )
        return {'loss_ce': loss_ce}

    def _loss_boxes(self, outputs, targets, indices):
        """Bounding box regression loss (L1 + GIoU)."""
        # Entity boxes
        entity_idx = self._get_src_permutation_idx(indices[0])
        pred_boxes_e = outputs['pred_boxes'][entity_idx]
        target_boxes_e = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices[0])
        ], dim=0)

        # Sub/Obj boxes from triplet matching
        triplet_idx = self._get_src_permutation_idx(indices[1])
        pred_boxes_s = outputs['sub_boxes'][triplet_idx]
        pred_boxes_o = outputs['obj_boxes'][triplet_idx]
        target_boxes_s = torch.cat([
            t['boxes'][t['rel_annotations'][i, 0]] for t, (_, i) in zip(targets, indices[1])
        ], dim=0)
        target_boxes_o = torch.cat([
            t['boxes'][t['rel_annotations'][i, 1]] for t, (_, i) in zip(targets, indices[1])
        ], dim=0)

        src_boxes = torch.cat([pred_boxes_e, pred_boxes_s, pred_boxes_o], dim=0)
        target_boxes = torch.cat([target_boxes_e, target_boxes_s, target_boxes_o], dim=0)

        num_boxes = max(src_boxes.shape[0], 1)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none').sum() / num_boxes

        loss_giou = (1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
        ))).sum() / num_boxes

        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def _loss_relations(self, outputs, targets, indices):
        """Predicate classification loss (CE on matched triplets).

        Background class is at index self.num_predicates (51 for VG).
        Output dim is self.num_predicates_out (52).
        Target predicates from dataset are 1-indexed (1-50), 0-indexed in loss.
        """
        rel_logits = outputs['rel_logits']     # [B, K, 52]
        bg_idx = self.num_predicates  # 51
        triplet_idx = self._get_src_permutation_idx(indices[1])
        target_rel_o = torch.cat([
            t['rel_annotations'][J, 2] for t, (_, J) in zip(targets, indices[1])
        ])
        # Convert 1-indexed → 0-indexed predicates
        target_rel = target_rel_o - 1
        target_rel = target_rel.clamp(0, self.num_predicates - 1)

        target_rel_dense = torch.full(
            rel_logits.shape[:2], bg_idx,
            dtype=torch.int64, device=rel_logits.device
        )
        target_rel_dense[triplet_idx] = target_rel

        loss_rel = F.cross_entropy(
            rel_logits.transpose(1, 2), target_rel_dense,
            self.empty_weight_rel
        )
        return {'loss_rel': loss_rel}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


# ==============================================================================
# Build function
# ==============================================================================

def build_usg(args):
    """Build USG model + criterion from config args."""
    num_classes = getattr(args, 'entity_nums', 151)
    num_predicates = getattr(args, 'rel_nums', 51)
    hidden_dim = getattr(args, 'hidden_dim', 256)
    nheads = getattr(args, 'nheads', 8)
    dec_layers = getattr(args, 'dec_layers', 6)
    dim_feedforward = getattr(args, 'dim_feedforward', 2048)
    dropout = getattr(args, 'dropout', 0.1)
    num_queries = getattr(args, 'num_queries', 200)
    aux_loss = getattr(args, 'aux_loss', True)
    pre_norm = getattr(args, 'pre_norm', True)
    rpc_layers = getattr(args, 'rpc_layers', 3)
    top_k_pairs = getattr(args, 'top_k_pairs', 64)
    rel_dec_layers = getattr(args, 'rel_dec_layers', 6)

    backbone = build_backbone(args)

    mask_decoder = USGMaskDecoder(
        d_model=hidden_dim,
        nhead=nheads,
        num_decoder_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_queries=num_queries,
        normalize_before=pre_norm,
    )

    object_head = USGObjectHead(
        d_model=hidden_dim,
        num_classes=num_classes,
    )

    rpc = RelationProposalConstructor(
        d_model=hidden_dim,
        nhead=nheads,
        num_rpc_layers=rpc_layers,
        top_k_pairs=top_k_pairs,
        dropout=dropout,
    )

    relation_decoder = USGRelationDecoder(
        d_model=hidden_dim,
        nhead=nheads,
        num_decoder_layers=rel_dec_layers,
        dim_feedforward=dim_feedforward,
        num_predicates=num_predicates,
        dropout=dropout,
        normalize_before=pre_norm,
    )

    model = USG(
        backbone=backbone,
        mask_decoder=mask_decoder,
        object_head=object_head,
        rpc=rpc,
        relation_decoder=relation_decoder,
        num_classes=num_classes,
        num_predicates=num_predicates,
        num_queries=num_queries,
        aux_loss=aux_loss,
    )

    matcher = build_matcher(args)

    weight_dict = {
        'loss_ce': getattr(args, 'ce_loss_coef', 1.0),
        'loss_bbox': getattr(args, 'bbox_loss_coef', 5.0),
        'loss_giou': getattr(args, 'giou_loss_coef', 2.0),
        'loss_rel': getattr(args, 'rel_loss_coef', 1.0),
    }

    if aux_loss:
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({f'{k}_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = USGSetCriterion(
        num_classes=num_classes,
        num_predicates=num_predicates,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=getattr(args, 'eos_coef', 0.1),
    )

    return model, criterion
