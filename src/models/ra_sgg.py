"""
RA-SGG: Retrieval-Augmented Scene Graph Generation.

Strictly aligned with the official repository:
  https://github.com/KanghoonYoon/torch-rasgg

Contains:
  - MLP: multi-layer perceptron matching official implementation
  - fusion_func: F.relu(x + y) - (x - y)^2
  - PENetBase: faithful port of PrototypeEmbeddingNetwork
  - RASGGModel: faithful port of ReTAGPENet (extends PENetBase)
  - build_ra_sgg: model factory
  - RASGGModel.build_memory_bank: static method for feature bank construction
"""
from __future__ import annotations

import math
import os
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torchvision.ops import roi_align
from torchvision.ops import nms as torchvision_nms
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights

from .motifs import FrequencyBias, generate_object_pairs


# ============================================================================
# FPN Feature Extractor — standalone, for RA-SGG only (does NOT touch backbone.py)
# ============================================================================
# Pipeline matches official maskrcnn_benchmark exactly:
#   Image → ResNeXt-101-32x8d (C2-C5) → FPN (P2-P5, 256 ch) → FPN Pooler (7×7)
#   → fc6(256×7×7→4096) + fc7(4096→4096) → [N, 4096] roi / union features

class _ResNetFPNBackbone(nn.Module):
    """ResNeXt-101-32x8d → C2-C5 feature maps for FPN.

    Matches official CONV_BODY=R-101-FPN with NUM_GROUPS=32, WIDTH_PER_GROUP=8.
    Uses torchvision resnext101_32x8d which has identical layer structure.
    """

    def __init__(self, pretrained: bool = True, frozen: bool = True):
        super().__init__()
        weights = ResNeXt101_32X8D_Weights.IMAGENET1K_V2 if pretrained else None
        rn = resnext101_32x8d(weights=weights)
        self.conv1 = rn.conv1
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.maxpool = rn.maxpool
        self.layer1 = rn.layer1  # C2: 3 blocks, 256 ch, s4
        self.layer2 = rn.layer2  # C3: 4 blocks, 512 ch, s8
        self.layer3 = rn.layer3  # C4: 23 blocks, 1024 ch, s16
        self.layer4 = rn.layer4  # C5: 3 blocks, 2048 ch, s32
        self.out_channels = [256, 512, 1024, 2048]
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> OrderedDict:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return OrderedDict([('c2', c2), ('c3', c3), ('c4', c4), ('c5', c5)])


class _FPN(nn.Module):
    """FPN top-down pathway: C2-C5 → P2-P5 all at out_channels=256."""

    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(ic, out_channels, 1) for ic in in_channels_list])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)

    def forward(self, features: OrderedDict) -> List[torch.Tensor]:
        names = list(features.keys())
        lats = [conv(features[n]) for conv, n in zip(self.lateral_convs, names)]
        for i in range(len(lats) - 1, 0, -1):
            h, w = lats[i - 1].shape[-2], lats[i - 1].shape[-1]
            lats[i - 1] = lats[i - 1] + F.interpolate(lats[i], size=(h, w), mode='nearest')
        return [conv(lat) for conv, lat in zip(self.output_convs, lats)]


class _FPNPooler(nn.Module):
    """Level-aware ROIAlign 7×7 using FPN heuristic."""

    def __init__(self, output_size: int = 7):
        super().__init__()
        self.output_size = output_size
        self.canonical_scale = 224.0
        self.canonical_level = 4
        self.k_min, self.k_max = 2, 5

    def _assign_levels(self, boxes: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
        w = boxes[:, 2] * image_size[1].float()
        h = boxes[:, 3] * image_size[0].float()
        area = torch.sqrt(w * h + 1e-6)
        lv = torch.floor(self.canonical_level + torch.log2(area / self.canonical_scale + 1e-6))
        return lv.clamp(self.k_min, self.k_max).long() - self.k_min

    def forward(self, fpn_feats: List[torch.Tensor],
                boxes: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
        N = boxes.size(0)
        if N == 0:
            return boxes.new_zeros(0, fpn_feats[0].size(1) * self.output_size ** 2)
        device, (img_h, img_w) = boxes.device, (image_size[0].float(), image_size[1].float())
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        lv = self._assign_levels(boxes, image_size)
        out = [None] * N
        for li in range(len(fpn_feats)):
            m = lv == li
            if not m.any():
                continue
            lb = torch.stack([x1[m], y1[m], x2[m], y2[m]], dim=1)
            r = torch.cat([torch.zeros(m.sum(), 1, device=device), lb], dim=1)
            fm = fpn_feats[li]
            if fm.dim() == 3:
                fm = fm.unsqueeze(0)
            ro = roi_align(fm, r, output_size=(self.output_size, self.output_size),
                           spatial_scale=1.0 / (2 ** (li + 2)), aligned=True)
            for j, idx in enumerate(torch.where(m)[0]):
                out[idx] = ro[j]
        return torch.stack([o for o in out if o is not None]).flatten(1)


class FPNFeatureExtractor(nn.Module):
    """Self-contained FPN feature extractor + detector for RA-SGG.

    PredCls / SGCls:
      Image → Backbone → FPN → Pooler → fc6+fc7 → [N, 4096] roi
      Union: rect_conv + union Pooler → ufc6+ufc7 → [P, 4096]

    SGDet (adds detector):
      Image → Backbone → FPN → RPN → proposals + NMS
      → Box Pooler → box_fc6+box_fc7 → cls_score + bbox_pred
      → NMS per class → detections → relation predictor
    """

    def __init__(self, pretrained: bool = True, frozen: bool = True, roi_output_size: int = 7):
        super().__init__()
        self.backbone = _ResNetFPNBackbone(pretrained, frozen)
        self.fpn = _FPN(self.backbone.out_channels, out_channels=256)
        self.pooler = _FPNPooler(output_size=roi_output_size)
        self.roi_output_size = roi_output_size
        fpn_ch = 256
        self.fc6 = nn.Linear(fpn_ch * roi_output_size ** 2, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.ufc6 = nn.Linear(fpn_ch * roi_output_size ** 2, 4096)
        self.ufc7 = nn.Linear(4096, 4096)
        for m in [self.fc6, self.fc7, self.ufc6, self.ufc7]:
            nn.init.kaiming_uniform_(m.weight, a=1)
            nn.init.constant_(m.bias, 0)
        self.rect_size = roi_output_size * 4 - 1
        self.rect_conv = nn.Sequential(
            nn.Conv2d(2, fpn_ch // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(fpn_ch // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(fpn_ch // 2, fpn_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(fpn_ch, momentum=0.01),
        )
        # ---- RPN head (official: shared 3×3 conv → cls_logits + bbox_pred) ----
        self.rpn_conv = nn.Conv2d(fpn_ch, 256, kernel_size=3, padding=1)
        self.rpn_cls_logits = nn.Conv2d(256, 4, kernel_size=1)
        self.rpn_bbox_pred = nn.Conv2d(256, 16, kernel_size=1)
        for m in [self.rpn_conv, self.rpn_cls_logits, self.rpn_bbox_pred]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
        # ---- Box detector head ----
        self.box_fc6 = nn.Linear(fpn_ch * roi_output_size ** 2, 4096)
        self.box_fc7 = nn.Linear(4096, 4096)
        self.box_cls_score = nn.Linear(4096, 151)
        self.box_bbox_pred = nn.Linear(4096, 151 * 4)
        for m in [self.box_fc6, self.box_fc7, self.box_cls_score, self.box_bbox_pred]:
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # ---- RPN anchors (official: ANCHOR_SIZES + ASPECT_RATIOS) ----
        anchor_sizes = [32, 64, 128, 256, 512]
        ar = torch.tensor([0.232, 0.633, 1.285, 3.151])
        self.rpn_cell_anchors = []
        for s in anchor_sizes:
            w = s * ar.sqrt()
            h = s / ar.sqrt()
            self.rpn_cell_anchors.append(
                torch.stack([-w/2, -h/2, w/2, h/2], dim=-1))  # [4, 4]
        self.rpn_pre_nms_top_n = dict(training=2000, testing=1000)
        self.rpn_post_nms_top_n = dict(training=2000, testing=1000)
        self.rpn_nms_thresh = 0.7
        self.rpn_strides = [4, 8, 16, 32, 64]  # P2-P6
        self.output_dim = 4096

    def _forward_mlp(self, x, union=False):
        fc6 = self.ufc6 if union else self.fc6
        fc7 = self.ufc7 if union else self.fc7
        x = x.flatten(1)
        x = F.relu(fc6(x))
        x = F.relu(fc7(x))
        return x

    def forward(self, images, boxes_list, image_sizes, return_feature_maps=False):
        dev = next(self.parameters()).device
        results, fpn_all = [], []
        for img, boxes, sz in zip(images, boxes_list, image_sizes):
            img = img.to(dev)
            boxes = boxes.to(dev)
            sz = sz.to(dev)
            with torch.set_grad_enabled(not all(p.requires_grad is False for p in self.backbone.parameters())):
                fpn_feats = self.fpn(self.backbone(img))
            if return_feature_maps:
                fpn_all.append(fpn_feats)
            results.append(self._forward_mlp(self.pooler(fpn_feats, boxes, sz)))
        return (results, fpn_all) if return_feature_maps else results

    def extract_union_features(self, fpn_feats_list, boxes_list, image_sizes, pair_indices_list):
        """Union features matching official RelationFeatureExtractor EXACTLY.

        official: union_vis (FPN Pooler) + rect_conv(head/tail masks) → fc6+fc7
        """
        results = []
        fpn_ch = 256
        for fpn_feats, boxes, sz, pi in zip(fpn_feats_list, boxes_list, image_sizes, pair_indices_list):
            P = pi.size(0)
            if P == 0:
                results.append(boxes.new_zeros(0, self.output_dim))
                continue
            dev = boxes.device
            img_h, img_w = sz[0].float(), sz[1].float()

            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = (cx - w/2)*img_w
            y1 = (cy - h/2)*img_h
            x2 = (cx + w/2)*img_w
            y2 = (cy + h/2)*img_h
            si, oi = pi[:, 0], pi[:, 1]

            # Union visual via FPN Pooler → [P, 256, 7, 7]
            ux1 = torch.min(x1[si], x1[oi])
            uy1 = torch.min(y1[si], y1[oi])
            ux2 = torch.max(x2[si], x2[oi])
            uy2 = torch.max(y2[si], y2[oi])
            ub = torch.stack([(ux1+ux2)/2/img_w, (uy1+uy2)/2/img_h, (ux2-ux1)/img_w, (uy2-uy1)/img_h], dim=1)
            union_vis = self.pooler(fpn_feats, ub, sz).view(P, fpn_ch, self.roi_output_size, self.roi_output_size)

            # Rectangle masks (head + tail) → rect_conv → [P, 256, 7, 7]
            x1_n = x1/img_w*self.rect_size
            y1_n = y1/img_h*self.rect_size
            x2_n = x2/img_w*self.rect_size
            y2_n = y2/img_h*self.rect_size
            masks = torch.zeros(P, 2, self.rect_size, self.rect_size, device=dev)
            for p in range(P):
                masks[p,0, y1_n[si[p]].floor().long().clamp(0,self.rect_size-1):y2_n[si[p]].ceil().long().clamp(0,self.rect_size-1)+1,
                       x1_n[si[p]].floor().long().clamp(0,self.rect_size-1):x2_n[si[p]].ceil().long().clamp(0,self.rect_size-1)+1] = 1.0
                masks[p,1, y1_n[oi[p]].floor().long().clamp(0,self.rect_size-1):y2_n[oi[p]].ceil().long().clamp(0,self.rect_size-1)+1,
                       x1_n[oi[p]].floor().long().clamp(0,self.rect_size-1):x2_n[oi[p]].ceil().long().clamp(0,self.rect_size-1)+1] = 1.0

            combined = union_vis + self.rect_conv(masks)
            results.append(self._forward_mlp(combined, union=True))
        return results

    # ---------- RPN Proposal Generation (SGDet) ----------

    def _rpn_forward(self, fpn_feats: List[torch.Tensor], training: bool):
        """RPN forward across all FPN levels. Returns objectness + box deltas."""
        logits, bbox_deltas, anchors = [], [], []
        for li, feat in enumerate(fpn_feats):
            t = self.rpn_conv(feat)
            logits.append(self.rpn_cls_logits(t))  # [1, 4, H, W]
            bbox_deltas.append(self.rpn_bbox_pred(t))  # [1, 16, H, W]
            # Generate anchors for this level and spatial size
            ca = self.rpn_cell_anchors[li].to(feat.device)  # [4, 4]
            stride = self.rpn_strides[li]
            H, W = feat.shape[-2], feat.shape[-1]
            sy = torch.arange(0, stride * H, stride, device=feat.device).float()
            sx = torch.arange(0, stride * W, stride, device=feat.device).float()
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            shifts = torch.stack([sx, sy, sx, sy], dim=-1).reshape(-1, 4)  # [H*W, 4]
            level_anchors = (ca.view(1, 4, 4) + shifts.view(-1, 1, 4)).reshape(-1, 4)  # [4*H*W, 4]
            anchors.append(level_anchors)
        return logits, bbox_deltas, anchors

    def _generate_proposals(self, fpn_feats, image_size, training):
        """Generate region proposals from RPN."""
        dev = fpn_feats[0].device
        logits, bbox_deltas, anchors = self._rpn_forward(fpn_feats, training)

        all_proposals, all_scores = [], []
        pre_nms = self.rpn_pre_nms_top_n['training' if training else 'testing']
        post_nms = self.rpn_post_nms_top_n['training' if training else 'testing']

        for logit, delta, anc in zip(logits, bbox_deltas, anchors):
            logit = logit.permute(0, 2, 3, 1).reshape(-1)      # [4*H*W]
            delta = delta.permute(0, 2, 3, 1).reshape(-1, 4)    # [4*H*W, 4]
            scores = torch.sigmoid(logit)

            # Top-k scores
            if scores.numel() > pre_nms:
                topk = torch.topk(scores, pre_nms)
                scores, idx = topk.values, topk.indices
                delta = delta[idx]
                anc = anc[idx]
            else:
                idx = torch.arange(scores.numel(), device=dev)

            # Decode proposals: apply deltas to anchors
            anc_w = anc[:, 2] - anc[:, 0]
            anc_h = anc[:, 3] - anc[:, 1]
            anc_cx = (anc[:, 0] + anc[:, 2]) / 2
            anc_cy = (anc[:, 1] + anc[:, 3]) / 2
            dx, dy, dw, dh = delta[:, 0], delta[:, 1], delta[:, 2], delta[:, 3]
            cx = anc_cx + dx * anc_w
            cy = anc_cy + dy * anc_h
            w = anc_w * torch.exp(dw)
            h = anc_h * torch.exp(dh)
            proposals = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

            # Clip to image
            ih, iw = image_size[0].float(), image_size[1].float()
            proposals[:, 0].clamp_(min=0)
            proposals[:, 1].clamp_(min=0)
            proposals[:, 2].clamp_(max=iw)
            proposals[:, 3].clamp_(max=ih)

            all_proposals.append(proposals)
            all_scores.append(scores)

        proposals = torch.cat(all_proposals, dim=0)
        scores = torch.cat(all_scores, dim=0)

        # NMS
        keep = torchvision_nms(proposals, scores, self.rpn_nms_thresh)
        if keep.numel() > post_nms:
            keep = keep[:post_nms]
        return proposals[keep], scores[keep]

    def extract_object_features_from_boxes(self, fpn_feats, boxes_list, image_sizes):
        """Extract per-box ROI features — used for SGDet where boxes come from detector."""
        results = []
        for fpn, boxes, sz in zip(fpn_feats, boxes_list, image_sizes):
            if boxes.numel() == 0:
                results.append(boxes.new_zeros(0, self.output_dim))
                continue
            pooled = self.pooler(fpn, boxes, sz)
            results.append(self._forward_mlp(pooled))
        return results

    def detect_objects(self, images, image_sizes, training=False):
        """Full SGDet detection pipeline.

        Returns: list of [N_i, 4] boxes (cxcywh norm), list of [N_i] labels,
                 list of [N_i] scores
        """
        dev = next(self.parameters()).device
        all_boxes_norm, all_labels, all_scores = [], [], []

        for img, sz in zip(images, image_sizes):
            img = img.to(dev)
            ih, iw = sz[0].float(), sz[1].float()
            fpn_feats = self.fpn(self.backbone(img))

            # RPN → proposals (xyxy abs)
            proposals, rpn_scores = self._generate_proposals(fpn_feats, sz, training)

            if proposals.numel() < 2:
                all_boxes_norm.append(proposals.new_zeros(0, 4))
                all_labels.append(proposals.new_zeros(0, dtype=torch.long))
                all_scores.append(proposals.new_zeros(0))
                continue

            # Convert proposals to cxcywh norm for pooler
            p_boxes = torch.stack([
                (proposals[:, 0] + proposals[:, 2]) / 2 / iw,
                (proposals[:, 1] + proposals[:, 3]) / 2 / ih,
                (proposals[:, 2] - proposals[:, 0]) / iw,
                (proposals[:, 3] - proposals[:, 1]) / ih,
            ], dim=-1)

            # Box head
            pooled = self.pooler(fpn_feats, p_boxes, sz)
            feats = F.relu(self.box_fc7(F.relu(self.box_fc6(pooled.flatten(1)))))
            cls = self.box_cls_score(feats)
            bbox = self.box_bbox_pred(feats).view(-1, 151, 4)

            # Predicted class and score
            scores, labels = F.softmax(cls, dim=-1)[:, 1:].max(dim=-1)
            labels = labels + 1

            # Decode boxes: apply class-specific deltas
            anc_w = p_boxes[:, 2] * iw
            anc_h = p_boxes[:, 3] * ih
            anc_cx = (proposals[:, 0] + proposals[:, 2]) / 2
            anc_cy = (proposals[:, 1] + proposals[:, 3]) / 2
            d = bbox[torch.arange(bbox.size(0)), labels]
            cx = anc_cx + d[:, 0] * anc_w
            cy = anc_cy + d[:, 1] * anc_h
            w = anc_w * torch.exp(d[:, 2])
            h = anc_h * torch.exp(d[:, 3])
            det_xyxy = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)
            det_xyxy[:, 0].clamp_(min=0)
            det_xyxy[:, 1].clamp_(min=0)
            det_xyxy[:, 2].clamp_(max=iw)
            det_xyxy[:, 3].clamp_(max=ih)

            # NMS → top-K
            keep = torchvision_nms(det_xyxy, scores, 0.5)
            if keep.numel() > 80:
                keep = keep[:80]

            # Convert to cxcywh norm for downstream
            det_cx = (det_xyxy[keep, 0] + det_xyxy[keep, 2]) / 2 / iw
            det_cy = (det_xyxy[keep, 1] + det_xyxy[keep, 3]) / 2 / ih
            det_w = (det_xyxy[keep, 2] - det_xyxy[keep, 0]) / iw
            det_h = (det_xyxy[keep, 3] - det_xyxy[keep, 1]) / ih
            all_boxes_norm.append(torch.stack([det_cx, det_cy, det_w, det_h], dim=-1))
            all_labels.append(labels[keep])
            all_scores.append(scores[keep])

        return all_boxes_norm, all_labels, all_scores

    def to(self, device):
        super().to(device)
        return self


def build_fpn_extractor(args) -> FPNFeatureExtractor:
    return FPNFeatureExtractor(
        pretrained=getattr(args, 'backbone_pretrained', True),
        frozen=getattr(args, 'backbone_frozen', True),
        roi_output_size=getattr(args, 'roi_output_size', 7),
    )


# ============================================================================
# Helper: MLP (exact match with official roi_relation_predictors.py lines 1027-1038)
# ============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron matching official implementation exactly."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# ============================================================================
# Fusion function (exact match with official roi_relation_predictors.py line 1042)
# ============================================================================

def fusion_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Official fusion: F.relu(x + y) - (x - y) ** 2."""
    return F.relu(x + y) - (x - y) ** 2


# ============================================================================
# Helper: one-hot encoding
# ============================================================================

def to_onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert class indices to one-hot vectors."""
    onehot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    onehot.scatter_(1, labels.unsqueeze(1).clamp(min=0, max=num_classes - 1), 1)
    return onehot


def encode_box_info(boxes: torch.Tensor) -> torch.Tensor:
    """9-dim box geometry feature (official encode_box_info from utils_motifs.py).

    Args:
        boxes: [N, 4] in (cx, cy, w, h) normalized format.

    Returns:
        [N, 9] tensor with (cx, cy, w, h, x1, y1, x2, y2, area).
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    area = w * h
    return torch.stack([cx, cy, w, h, x1, y1, x2, y2, area], dim=-1)


# ============================================================================
# Helper: load GloVe embeddings
# ============================================================================

def _load_glove_vectors(class_names: List[str], wv_dim: int = 300,
                         glove_dir: str = 'data/glove') -> torch.Tensor:
    """Load GloVe word vectors for given class names.

    Matches the logic of `obj_edge_vectors` and `rel_vectors` in the official
    repository (utils_motifs.py). Falls back to random initialization if GloVe
    file is not found.
    """
    glove_path = os.path.join(glove_dir, f'glove.6B.{wv_dim}d.txt')
    if not os.path.exists(glove_path):
        warnings.warn(
            f"GloVe file not found at {glove_path}. "
            f"Using random initialization for word embeddings. "
            f"Download from https://nlp.stanford.edu/data/glove.6B.zip for "
            f"reproduction-quality results.",
            RuntimeWarning, stacklevel=2,
        )
        # Fall back to random initialization
        return torch.randn(len(class_names), wv_dim) * 0.1

    with open(glove_path, 'r', encoding='utf-8') as f:
        glove = {}
        for line in f:
            parts = line.strip().split()
            if len(parts) == wv_dim + 1:
                glove[parts[0]] = torch.tensor(
                    [float(x) for x in parts[1:]], dtype=torch.float32)

    vecs = []
    for name in class_names:
        tokens = name.lower().replace('-', ' ').replace('_', ' ').split()
        token_vecs = []
        for token in tokens:
            if token in glove:
                token_vecs.append(glove[token])
        if token_vecs:
            vecs.append(torch.stack(token_vecs).mean(0))
        else:
            vecs.append(torch.randn(wv_dim) * 0.1)

    return torch.stack(vecs)


# The VG predicate class names (0-indexed, 0 = __background__, 1-50 = predicates)
VG_PREDICATE_NAMES = [
    '__background__',
    'above', 'against', 'at', 'attached to', 'behind',
    'belonging to', 'between', 'carrying', 'covered in', 'covering',
    'eating', 'flying in', 'for', 'from', 'growing on',
    'hanging from', 'has', 'holding', 'in', 'in front of',
    'laying on', 'looking at', 'lying on', 'made of', 'mounted on',
    'near', 'of', 'on', 'on back of', 'over',
    'painted on', 'parked on', 'part of', 'playing', 'riding',
    'says', 'sitting on', 'skating on', 'skiing on', 'standing on',
    'surfing on', 'to', 'under', 'using', 'walking in',
    'walking on', 'watching', 'wearing', 'wears', 'with',
]

# VG-150 standard object classes (exactly 150 + __background__ = 151 total)
VG_OBJECT_NAMES = [
    '__background__',
    'airplane', 'animal', 'arm', 'bag', 'banana',
    'basket', 'beach', 'bear', 'bed', 'bench',
    'bike', 'bird', 'board', 'boat', 'book',
    'boot', 'bottle', 'bowl', 'box', 'boy',
    'branch', 'building', 'bus', 'cabinet', 'cap',
    'car', 'cat', 'chair', 'child', 'clock',
    'coat', 'counter', 'cow', 'cup', 'curtain',
    'desk', 'dog', 'door', 'drawer', 'ear',
    'elephant', 'engine', 'eye', 'face', 'fence',
    'finger', 'flag', 'flower', 'food', 'fork',
    'fruit', 'giraffe', 'girl', 'glass', 'glove',
    'guy', 'hair', 'hand', 'handle', 'hat',
    'head', 'helmet', 'hill', 'horse', 'house',
    'jacket', 'jeans', 'kid', 'kite', 'lady',
    'lamp', 'laptop', 'leaf', 'leg', 'letter',
    'light', 'logo', 'man', 'men', 'mirror',
    'motorcycle', 'mountain', 'mouth', 'neck', 'nose',
    'number', 'orange', 'pant', 'paper', 'paw',
    'people', 'person', 'phone', 'pillow', 'pizza',
    'plane', 'plant', 'plate', 'player', 'pole',
    'post', 'pot', 'racket', 'railing', 'rock',
    'roof', 'room', 'screen', 'seat', 'sheep',
    'shelf', 'shirt', 'shoe', 'short', 'sidewalk',
    'sign', 'sink', 'skateboard', 'ski', 'skier',
    'snow', 'sock', 'stand', 'street', 'surfboard',
    'table', 'tail', 'tie', 'tile', 'tire',
    'toilet', 'towel', 'tower', 'track', 'train',
    'tree', 'truck', 'trunk', 'umbrella', 'vase',
    'vegetable', 'vehicle', 'wave', 'wheel', 'window',
    'windshield', 'wing', 'wire', 'woman', 'zebra',
]


def _predicate_frequencies_from_dataset(data_root: str) -> torch.Tensor:
    """Estimate predicate frequencies from training data annotations."""
    import json
    train_json = os.path.join(data_root, 'VisualGenome', 'train.json')
    if not os.path.exists(train_json):
        return None

    with open(train_json, 'r') as f:
        anns = json.load(f)

    # Count predicate frequencies
    pred_counts = torch.zeros(51, dtype=torch.float32)
    for ann in anns:
        for rel in ann.get('relations', []):
            pred = int(rel[2])
            if 0 <= pred < 51:
                pred_counts[pred] += 1

    # Background count = total annotated pairs * some factor
    total_rels = pred_counts[1:].sum()
    pred_counts[0] = total_rels * 20  # official uses 20x for bg
    return pred_counts


# ============================================================================
# VG Head / Body / Tail splits (exact match with official defaults.py)
# ============================================================================

VG_HEAD_IDS = [8, 20, 22, 29, 30, 31, 48]  # 1-indexed predicate IDs
VG_BODY_IDS = [1, 5, 6, 7, 9, 11, 16, 19, 21, 23, 25, 33, 35, 38, 40, 41, 43, 46, 47, 49, 50]
VG_TAIL_IDS = [2, 3, 4, 10, 12, 13, 14, 15, 17, 18, 24, 26, 27, 28, 32, 34, 36, 37, 39, 42, 44, 45]


# ============================================================================
# PENetBase: Faithful port of PrototypeEmbeddingNetwork
# ============================================================================

class PENetBase(nn.Module):
    """PE-Net backbone — strictly aligned with official PrototypeEmbeddingNetwork.

    Key dimensions (matching official):
      - mlp_dim = 2048
      - embed_dim = 300  (GloVe)
      - pooling_dim / visual_dim: input ROI feature dimension (4096 in official,
        2048 when using OpenSGG's ResNet backbone)

    Architecture:
      1. Object embedding projection: post_emb splits roi_features into sub/obj
      2. Semantic prototypes: GloVe embeddings projected via W_sub/W_obj/W_pred
      3. Gated fusion: semantic prototype + gated visual feature
      4. Union debiasing: rel_rep = fusion_so - gate_pred * h(xu)
      5. Projection head: projects rel_rep and prototypes to shared metric space
      6. Cosine similarity classification with learnable logit_scale
      7. Prototype regularizations (l21, dist) + Euclidean distance triplet loss
    """

    def __init__(
        self,
        num_classes: int = 151,
        num_predicates: int = 51,
        visual_dim: int = 2048,
        mlp_dim: int = 2048,
        embed_dim: int = 300,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        use_freq_bias: bool = True,
        freq_bias_eps: float = 1e-12,
        use_union: bool = True,
        mode: str = 'predcls',
        glove_dir: str = 'data/glove',
        rel_loss_type: str = 'ce',
        reweight_beta: float = 0.99999,
        obj_class_names: Optional[List[str]] = None,
        pred_class_names: Optional[List[str]] = None,
        pred_count_dict: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.mlp_dim = mlp_dim
        self.embed_dim = embed_dim
        self.use_union = use_union
        self.rel_loss_type = rel_loss_type
        self.num_rel_cls = num_predicates  # for add_losses compatibility
        self.hidden_dim = hidden_dim       # official: CONTEXT_HIDDEN_DIM=512
        self.pooling_dim = 4096             # official: CONTEXT_POOLING_DIM=4096 (MLP_HEAD_DIM)
        self.mode = mode  # predcls / sgcls / sgdet

        # ---------- Object feature projection ----------
        # Official: FPN2MLPFeatureExtractor outputs 4096-dim
        # Our backbone: FPN + Pooler + fc6+fc7 → 4096-dim (see backbone.py)
        self.post_emb = nn.Linear(self.pooling_dim, mlp_dim * 2)  # [4096] → [4096]

        # ---------- Union feature downsampling ----------
        # official: MLP(4096, 2048, 2048, 2)
        self.down_samp = MLP(self.pooling_dim, mlp_dim, mlp_dim, 2)

        # ---------- Load GloVe embeddings ----------
        obj_names = obj_class_names if obj_class_names else (VG_OBJECT_NAMES[:num_classes] if len(VG_OBJECT_NAMES) >= num_classes else VG_OBJECT_NAMES)
        pred_names = pred_class_names if pred_class_names else (VG_PREDICATE_NAMES[:num_predicates] if len(VG_PREDICATE_NAMES) >= num_predicates else VG_PREDICATE_NAMES)

        obj_embed_vecs = _load_glove_vectors(obj_names, wv_dim=embed_dim,
                                              glove_dir=glove_dir)
        rel_embed_vecs = _load_glove_vectors(pred_names, wv_dim=embed_dim,
                                              glove_dir=glove_dir)

        self.obj_embed = nn.Embedding(num_classes, embed_dim)
        self.rel_embed = nn.Embedding(num_predicates, embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

        # ---------- Semantic prototype projections ----------
        self.W_sub = MLP(embed_dim, mlp_dim // 2, mlp_dim, 2)
        self.W_obj = MLP(embed_dim, mlp_dim // 2, mlp_dim, 2)
        self.W_pred = MLP(embed_dim, mlp_dim // 2, mlp_dim, 2)

        # ---------- Gating ----------
        self.gate_sub = nn.Linear(mlp_dim * 2, mlp_dim)
        self.gate_obj = nn.Linear(mlp_dim * 2, mlp_dim)
        self.gate_pred = nn.Linear(mlp_dim * 2, mlp_dim)

        # ---------- Visual-to-semantic projection ----------
        self.vis2sem = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim * 2, mlp_dim),
        )

        # ---------- Projection head (2-layer MLP, output = 2*mlp_dim) ----------
        self.project_head = MLP(mlp_dim, mlp_dim, mlp_dim * 2, 2)

        # ---------- Residual blocks + LayerNorm ----------
        self.linear_sub = nn.Linear(mlp_dim, mlp_dim)
        self.linear_obj = nn.Linear(mlp_dim, mlp_dim)
        self.linear_pred = nn.Linear(mlp_dim, mlp_dim)
        self.linear_rel_rep = nn.Linear(mlp_dim, mlp_dim)

        self.norm_sub = nn.LayerNorm(mlp_dim)
        self.norm_obj = nn.LayerNorm(mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(mlp_dim)

        self.dropout_sub = nn.Dropout(dropout)
        self.dropout_obj = nn.Dropout(dropout)
        self.dropout_rel_rep = nn.Dropout(dropout)
        self.dropout_rel = nn.Dropout(dropout)
        self.dropout_pred = nn.Dropout(dropout)

        # ---------- Learnable temperature ----------
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        # ---------- Object label refinement (official POS_EMBED + OUT_OBJ + LIN_OBJ_CYX) ----------
        self.obj_dim = self.pooling_dim  # official: in_channels = 4096

        self.pos_embed = nn.Sequential(
            nn.Linear(9, 32),
            nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
        )

        self.obj_embed1 = nn.Embedding(num_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.out_obj = nn.Linear(self.hidden_dim, num_classes)
        nn.init.kaiming_uniform_(self.out_obj.weight, a=1)
        nn.init.constant_(self.out_obj.bias, 0)

        self.lin_obj_cyx = nn.Linear(self.obj_dim + self.embed_dim + 128, self.hidden_dim)
        nn.init.kaiming_uniform_(self.lin_obj_cyx.weight, a=1)
        nn.init.constant_(self.lin_obj_cyx.bias, 0)

        self.nms_thresh = 0.5  # official: cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        # ---------- Frequency bias ----------
        self.freq_bias = None
        if use_freq_bias:
            self.freq_bias = FrequencyBias(num_predicates, freq_bias_eps)

        # ---------- Class-balanced loss weight ----------
        self._build_loss_weight(pred_count_dict, reweight_beta)

        self._init_weights()

    def _build_loss_weight(self, pred_count_dict, reweight_beta):
        """Build class-balanced reweighting for CE loss (ce_rwt mode)."""
        num_preds = self.num_predicates

        if pred_count_dict is not None:
            counts = torch.tensor(
                [pred_count_dict.get(name, 1.0) for name in VG_PREDICATE_NAMES],
                dtype=torch.float32)
        else:
            # Uniform fallback
            counts = torch.ones(num_preds)

        if self.rel_loss_type == 'ce':
            self.register_buffer('rel_loss_weight', torch.ones(num_preds))
        elif self.rel_loss_type == 'ce_rwt':
            weight = (1.0 - reweight_beta) / (1.0 - reweight_beta ** counts)
            median = torch.median(weight[1:])
            weight = weight / median
            weight[0] = torch.min(weight[1:])
            self.register_buffer('rel_loss_weight', weight)
        else:
            self.register_buffer('rel_loss_weight', torch.ones(num_preds))

    def _init_weights(self):
        """Initialize non-embedding layers matching official make_fc pattern.

        Official `make_fc` uses kaiming_uniform_ with a=1 (matching Caffe2 XavierFill).
        PyTorch's default Linear init is also kaiming_uniform_, so we only need to
        handle LayerNorm here since Embeddings are already GloVe-initialized.
        """
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    # ---------- Forward ----------

    def forward(
        self,
        roi_features: torch.Tensor,
        union_features: Optional[torch.Tensor],
        labels: torch.Tensor,
        boxes: torch.Tensor,
        rel_labels: torch.Tensor,
        return_obj_preds: bool = False,
    ) -> dict:
        """PE-Net forward pass — strictly aligned with official code.

        Args:
            roi_features: [N, visual_dim] per-object ROI features.
            union_features: [P, visual_dim] per-pair union ROI features (or None
                            to approximate).
            labels: [N] object class labels.
            boxes: [N, 4] bounding boxes (cxcywh format).
            rel_labels: [P] predicate labels for each pair (0 for background).
            return_obj_preds: if True, also return object predictions.

        Returns:
            dict with keys:
              - rel_dists: [P, num_predicates] cosine similarity logits
              - entity_dists: [N, num_classes] object class logits (or one-hot)
              - pair_indices: [P, 2] (sub_idx, obj_idx)
              - sub_boxes: [P, 4], obj_boxes: [P, 4]
              - obj_labels: [N] object predictions
              - add_losses: dict of regularization losses
              - rel_rep: [P, mlp_dim*2] relation embeddings (for memory bank)
              - predicate_proto: [num_predicates, mlp_dim*2] prototypes
              - predicate_proto_pre: [num_predicates, mlp_dim] pre-projection
        """
        N = roi_features.size(0)
        device = roi_features.device
        add_losses = {}

        # ---- Step 1: Object label refinement (official: refine_obj_labels) ----
        # roi_features already 4096-dim from backbone FPN+MLP pipeline
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, labels, boxes)
        # For SGCls/SGDet: return_obj_preds controls whether to output obj_logits
        obj_logits_out = None
        if return_obj_preds and self.mode != 'predcls' and not self.training:
            obj_logits_out = self.out_obj(
                self.lin_obj_cyx(torch.cat([
                    roi_features,
                    self.obj_embed1(entity_preds),
                    self.pos_embed(encode_box_info(boxes)),
                ], dim=-1)))

        # ---- Step 2: Split visual features into sub/obj ----
        entity_rep = self.post_emb(roi_features)  # [N, mlp_dim*2]
        entity_rep = entity_rep.view(N, 2, self.mlp_dim)
        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)  # [N, mlp_dim]
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)  # [N, mlp_dim]

        # ---- Step 3: Get word embeddings for predicted/GT entities ----
        entity_embeds = self.obj_embed(entity_preds)  # [N, embed_dim]

        # ---- Step 4: Generate all directed pairs ----
        pairs = generate_object_pairs(N, device)
        if pairs.numel() == 0:
            return {
                'rel_logits': roi_features.new_zeros(0, self.num_predicates),
                'entity_dists': entity_dists,
                'pair_indices': pairs,
                'sub_boxes': boxes.new_zeros(0, 4),
                'obj_boxes': boxes.new_zeros(0, 4),
                'obj_labels': entity_preds,
                'add_losses': {},
                'rel_rep': roi_features.new_zeros(0, self.mlp_dim * 2),
                'predicate_proto': roi_features.new_zeros(0, self.mlp_dim * 2),
                'predicate_proto_pre': roi_features.new_zeros(0, self.mlp_dim),
            }

        s_idx = pairs[:, 0]
        o_idx = pairs[:, 1]

        # ---- Step 5: Semantic prototypes for subject/object ----
        s_embed = self.W_sub(entity_embeds[s_idx])  # [P, mlp_dim]
        o_embed = self.W_obj(entity_embeds[o_idx])  # [P, mlp_dim]

        # ---- Step 6: Visual-to-semantic projection ----
        sem_sub = self.vis2sem(sub_rep[s_idx])  # [P, mlp_dim]
        sem_obj = self.vis2sem(obj_rep[o_idx])  # [P, mlp_dim]

        # ---- Step 7: Gated fusion (semantic prototype + gated visual) ----
        gate_sem_sub = torch.sigmoid(
            self.gate_sub(torch.cat([s_embed, sem_sub], dim=-1)))
        gate_sem_obj = torch.sigmoid(
            self.gate_obj(torch.cat([o_embed, sem_obj], dim=-1)))

        sub = s_embed + sem_sub * gate_sem_sub  # s = Ws·ts + gs·h(xs)
        obj = o_embed + sem_obj * gate_sem_obj  # o = Wo·to + go·h(xo)

        # ---- Step 8: Residual + LayerNorm for convergence ----
        sub = self.norm_sub(self.dropout_sub(F.relu(self.linear_sub(sub))) + sub)
        obj = self.norm_obj(self.dropout_obj(F.relu(self.linear_obj(obj))) + obj)

        # ---- Step 9: Fusion F(s, o) using official fusion_func ----
        fusion_so = fusion_func(sub, obj)  # [P, mlp_dim]

        # ---- Step 10: Union feature debiasing ----
        if self.use_union and union_features is not None:
            sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
            gate_sem_pred = torch.sigmoid(
                self.gate_pred(torch.cat([fusion_so, sem_pred], dim=-1)))
            rel_rep = fusion_so - sem_pred * gate_sem_pred  # r = F(s,o) - gp·h(xu)
        else:
            # Fallback: approximate with average of sub/obj features
            union_vis = (sub_rep[s_idx] + obj_rep[o_idx]) / 2.0
            sem_pred = self.vis2sem(union_vis)
            gate_sem_pred = torch.sigmoid(
                self.gate_pred(torch.cat([fusion_so, sem_pred], dim=-1)))
            rel_rep = fusion_so - sem_pred * gate_sem_pred

        # ---- Step 11: Relation representation residual + LayerNorm ----
        rel_rep = self.norm_rel_rep(
            self.dropout_rel_rep(F.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        # ---- Step 12: Predicate prototypes ----
        predicate_proto = self.W_pred(self.rel_embed.weight)  # [C, mlp_dim]

        # ---- Step 13: Project to metric space ----
        rel_rep = self.project_head(self.dropout_rel(F.relu(rel_rep)))  # [P, mlp_dim*2]
        predicate_proto = self.project_head(
            self.dropout_pred(F.relu(predicate_proto)))  # [C, mlp_dim*2]

        # ---- Step 14: Cosine similarity classification ----
        rel_rep_norm = rel_rep / (rel_rep.norm(dim=1, keepdim=True) + 1e-8)
        pred_proto_norm = predicate_proto / (
            predicate_proto.norm(dim=1, keepdim=True) + 1e-8)

        rel_dists = rel_rep_norm @ pred_proto_norm.t() * self.logit_scale.exp().clamp(max=100.0)

        # ---- Step 15: Prototype regularization (training only) ----
        if self.training:
            self._compute_regularization_losses(
                predicate_proto, pred_proto_norm, rel_rep, rel_labels, add_losses)

        # ---- Step 16: Return ----
        return {
            'rel_logits': rel_dists,
            'entity_dists': entity_dists,
            'pair_indices': pairs,
            'sub_boxes': boxes[pairs[:, 0]],
            'obj_boxes': boxes[pairs[:, 1]],
            'obj_labels': entity_preds,
            'obj_logits': obj_logits_out,
            'add_losses': add_losses,
            'rel_rep': rel_rep,
            'predicate_proto': predicate_proto,
            'predicate_proto_pre': self.W_pred(self.rel_embed.weight),
            'rel_rep_norm': rel_rep_norm,
            'pred_proto_norm': pred_proto_norm,
            'logit_scale': self.logit_scale,
        }

    # ---------- Object Label Refinement (official refine_obj_labels) ----------

    def refine_obj_labels(self, roi_features: torch.Tensor, labels: torch.Tensor,
                          boxes: torch.Tensor):
        """Refine object labels — matches official PrototypeEmbeddingNetwork.

        Uses self.mode to determine behaviour:
          - predcls: return GT labels as one-hot
          - sgcls/sgdet: predict labels via out_obj head
        """
        use_gt_label = self.training or (self.mode == 'predcls')

        if use_gt_label:
            obj_labels = labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            # For sgcls/sgdet eval: predict via lin_obj_cyx → out_obj
            pos_embed = self.pos_embed(encode_box_info(boxes))
            obj_pre_rep = self.lin_obj_cyx(
                torch.cat([roi_features, self.obj_embed1.weight.mean(0, keepdim=True).expand(roi_features.size(0), -1), pos_embed], dim=-1))
            obj_logits = self.out_obj(obj_pre_rep)
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        pos_embed = self.pos_embed(encode_box_info(boxes))
        obj_pre_rep_for_pred = self.lin_obj_cyx(
            torch.cat([roi_features, obj_embed, pos_embed], dim=-1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)
            obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()

        return obj_dists, obj_preds

    def _compute_regularization_losses(
        self,
        predicate_proto: torch.Tensor,
        pred_proto_norm: torch.Tensor,
        rel_rep: torch.Tensor,
        rel_labels: torch.Tensor,
        add_losses: dict,
    ):
        """Compute prototype regularization losses (exact match with official)."""
        C = self.num_predicates

        # ---- Prototype Similarity Regularization (L_reg1 / l21_loss) ----
        target_proto_norm = pred_proto_norm.clone().detach()
        simil_mat = pred_proto_norm @ target_proto_norm.t()  # S = C_norm @ C_norm.T
        l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (C * C)
        add_losses['l21_loss'] = l21

        # ---- Prototype Distance Regularization (L_reg2 / dist_loss2) ----
        gamma2 = 7.0
        proto_a = predicate_proto.unsqueeze(1).expand(-1, C, -1)
        proto_b = predicate_proto.detach().unsqueeze(0).expand(C, -1, -1)
        proto_dis_mat = (proto_a - proto_b).norm(dim=2) ** 2  # d_ij = ||c_i - c_j||^2
        sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
        topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1  # k2=1
        dist_loss = torch.clamp(-topK_proto_dis + gamma2, min=0).mean()
        add_losses['dist_loss2'] = dist_loss

        # ---- Prototype-based Learning: Euclidean distance triplet loss (loss_dis) ----
        gamma1 = 1.0
        rel_rep_expand = rel_rep.unsqueeze(1).expand(-1, C, -1)  # [P, C, D]
        proto_expand = predicate_proto.unsqueeze(0).expand(rel_labels.size(0), -1, -1)
        distance_set = (rel_rep_expand - proto_expand).norm(dim=2) ** 2  # g_i = ||r - c_i||^2

        mask_neg = torch.ones(rel_labels.size(0), C, device=rel_labels.device)
        mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0

        distance_set_neg = distance_set * mask_neg
        distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]

        sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
        topK_sorted_distance_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # k1=10

        loss_sum = torch.clamp(
            distance_set_pos - topK_sorted_distance_neg + gamma1, min=0).mean()
        add_losses['loss_dis'] = loss_sum


# ============================================================================
# RASGGModel: Faithful port of ReTAGPENet
# ============================================================================

class RASGGModel(PENetBase):
    """RA-SGG: Retrieval-Augmented SGG — strictly aligned with official ReTAGPENet.

    Extends PENetBase with:
      1. Memory bank loading and k-NN retrieval
      2. Reliable multi-labeled instance selection (tau threshold)
      3. Unbiased multi-label augmentation (IPS sampling)
      4. Mixup label augmentation (Beta distribution)
      5. Modified Euclidean distance triplet loss with mixup labels
    """

    def __init__(
        self,
        num_classes: int = 151,
        num_predicates: int = 51,
        visual_dim: int = 2048,
        mlp_dim: int = 2048,
        embed_dim: int = 300,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        use_freq_bias: bool = True,
        freq_bias_eps: float = 1e-12,
        use_union: bool = True,
        glove_dir: str = 'data/glove',
        mode: str = 'predcls',
        rel_loss_type: str = 'ce',
        reweight_beta: float = 0.99999,
        obj_class_names: Optional[List[str]] = None,
        pred_class_names: Optional[List[str]] = None,
        pred_count_dict: Optional[Dict[str, float]] = None,
        # RA-SGG specific parameters
        num_retrievals: int = 10,
        threshold: float = 0.3,
        mixup: bool = True,
        mixup_alpha: float = 20.0,
        mixup_beta: float = 5.0,
        num_correct_bg: int = 1,
        head_ids: Optional[List[int]] = None,
        body_ids: Optional[List[int]] = None,
        tail_ids: Optional[List[int]] = None,
        memory_bank_path: Optional[str] = None,
    ):
        super().__init__(
            mode=mode,
            num_classes=num_classes,
            num_predicates=num_predicates,
            visual_dim=visual_dim,
            mlp_dim=mlp_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_freq_bias=use_freq_bias,
            freq_bias_eps=freq_bias_eps,
            use_union=use_union,
            glove_dir=glove_dir,
            rel_loss_type=rel_loss_type,
            reweight_beta=reweight_beta,
            obj_class_names=obj_class_names,
            pred_class_names=pred_class_names,
            pred_count_dict=pred_count_dict,
        )

        # ---- RA-SGG parameters ----
        self.num_retrievals = num_retrievals       # K
        self.tau = threshold                        # τ
        self.do_mixup = mixup
        self.mixup_alpha = mixup_alpha              # Beta α
        self.mixup_beta = mixup_beta                # Beta β
        self.num_correct_bg = num_correct_bg

        # ---- Head/Body/Tail split IDs ----
        self.register_buffer(
            'tail_ids',
            torch.tensor(tail_ids if tail_ids else VG_TAIL_IDS, dtype=torch.long))
        self.register_buffer(
            'body_ids',
            torch.tensor(body_ids if body_ids else VG_BODY_IDS, dtype=torch.long))
        self.register_buffer(
            'head_ids',
            torch.tensor(head_ids if head_ids else VG_HEAD_IDS, dtype=torch.long))

        # ---- Memory bank buffers (pre-projection: mlp_dim=2048, matching official) ----
        self.register_buffer('fb_keys', torch.empty(0))     # [M, mlp_dim]
        self.register_buffer('fb_values', torch.empty(0, dtype=torch.long))  # [M, 3]

        # ---- Retrieval statistics (for logging) ----
        self.pos_masks = {
            'all_poscnt': 0, 'all_cnt': 0,
            'bg_poscnt': 0, 'bg_cnt': 0,
            'head_poscnt': 0, 'head_cnt': 0,
            'body_poscnt': 0, 'body_cnt': 0,
            'tail_poscnt': 0, 'tail_cnt': 0,
        }
        self.pos_retrieval_prop = {
            'all': [], 'bg': [], 'head': [], 'body': [], 'tail': []}

        # ---- Load memory bank if path provided ----
        if memory_bank_path and os.path.exists(memory_bank_path):
            self.load_memory_bank(memory_bank_path)

    # ---------- Memory Bank ----------

    def load_memory_bank(self, bank_path: str):
        """Load pre-processed feature bank.

        Expected format: .npy file containing a dict with:
          - 'key': (M, D) FloatTensor — normalized relation embeddings
          - 'value': (M, 3) LongTensor — [sub_class, obj_class, pred_class]
        """
        fb = np.load(bank_path, allow_pickle=True).tolist()
        self.fb_keys = torch.FloatTensor(fb['key'])
        self.fb_values = torch.LongTensor(fb['value'])
        print(f"[RA-SGG] Loaded memory bank: {self.fb_keys.size(0)} entries "
              f"(dim={self.fb_keys.size(1)})")

    def _move_memory_to_device(self, device):
        """Move memory bank to target device."""
        if self.fb_keys.numel() > 0 and self.fb_keys.device != device:
            self.fb_keys = self.fb_keys.to(device)
        if self.fb_values.numel() > 0 and self.fb_values.device != device:
            self.fb_values = self.fb_values.to(device)

    def remap_external_state_dict(self, state_dict: dict, visual_extractor=None) -> dict:
        """Remap official maskrcnn_benchmark checkpoint keys to OpenSGG keys.

        Handles FOUR key spaces:
          1. Backbone:   backbone.body.* → backbone.*  (ResNeXt-101-32x8d)
          2. FPN:        backbone.fpn.fpn_innerN → fpn.lateral_convs.N-2
                         backbone.fpn.fpn_layerN → fpn.output_convs.N-2
          3. Box head:   roi_heads.relation.box_feature_extractor.fc[67] → mlp_head.[02]
          4. Predictor:  roi_heads.relation.predictor.* → * (direct)

        If *visual_extractor* is provided, backbone/FPN/box_extractor weights are
        loaded into it immediately.  The returned dict contains only predictor keys
        for the model itself.
        """
        remapped = {}
        ve_weights = {} if visual_extractor is not None else None

        for k, v in state_dict.items():
            new_k = k

            # ---- Predictor prefix strip ----
            for prefix in ['roi_heads.relation.predictor.', 'module.roi_heads.relation.predictor.']:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
                    break

            # ---- Backbone: backbone.body. → backbone. ----
            if new_k.startswith('backbone.body.'):
                new_k = 'backbone.' + new_k[len('backbone.body.'):]
                # stem: torchvision has conv1/bn1 at root, not under stem/
                new_k = new_k.replace('backbone.stem.', 'backbone.')
            elif new_k.startswith('backbone.fpn.'):
                # FPN lateral convs: fpn_inner1→lateral_convs.0, fpn_inner2→1, ...
                import re
                m_inner = re.match(r'backbone\.fpn\.fpn_inner(\d+)\.(.+)', new_k)
                m_layer = re.match(r'backbone\.fpn\.fpn_layer(\d+)\.(.+)', new_k)
                if m_inner:
                    idx = int(m_inner.group(1)) - 1  # fpn_inner1→0
                    new_k = f'fpn.lateral_convs.{idx}.{m_inner.group(2)}'
                elif m_layer:
                    idx = int(m_layer.group(1)) - 1
                    new_k = f'fpn.output_convs.{idx}.{m_layer.group(2)}'

            # ---- RPN head → rpn_* ----
            elif new_k.startswith('rpn.head.conv.'):
                new_k = new_k.replace('rpn.head.conv.', 'rpn_conv.')
            elif new_k.startswith('rpn.head.cls_logits.'):
                new_k = new_k.replace('rpn.head.cls_logits.', 'rpn_cls_logits.')
            elif new_k.startswith('rpn.head.bbox_pred.'):
                new_k = new_k.replace('rpn.head.bbox_pred.', 'rpn_bbox_pred.')
            # ---- Box feature extractor → box_fc6/box_fc7 (detector box head) ----
            elif new_k.startswith('roi_heads.box.feature_extractor.fc6'):
                new_k = new_k.replace('roi_heads.box.feature_extractor.fc6', 'box_fc6')
            elif new_k.startswith('roi_heads.box.feature_extractor.fc7'):
                new_k = new_k.replace('roi_heads.box.feature_extractor.fc7', 'box_fc7')
            # ---- Box predictor → box_cls_score / box_bbox_pred ----
            elif new_k.startswith('roi_heads.box.predictor.cls_score.'):
                new_k = new_k.replace('roi_heads.box.predictor.cls_score.', 'box_cls_score.')
            elif new_k.startswith('roi_heads.box.predictor.bbox_pred.'):
                new_k = new_k.replace('roi_heads.box.predictor.bbox_pred.', 'box_bbox_pred.')
            # ---- Relation box feature extractor → fc6/fc7 ----
            elif new_k.startswith('roi_heads.relation.box_feature_extractor.fc6'):
                new_k = new_k.replace('roi_heads.relation.box_feature_extractor.fc6', 'fc6')
            elif new_k.startswith('roi_heads.relation.box_feature_extractor.fc7'):
                new_k = new_k.replace('roi_heads.relation.box_feature_extractor.fc7', 'fc7')
            # ---- Union feature extractor → ufc6/ufc7 + rect_conv ----
            elif new_k.startswith('roi_heads.relation.union_feature_extractor.feature_extractor.fc6'):
                new_k = new_k.replace('roi_heads.relation.union_feature_extractor.feature_extractor.fc6', 'ufc6')
            elif new_k.startswith('roi_heads.relation.union_feature_extractor.feature_extractor.fc7'):
                new_k = new_k.replace('roi_heads.relation.union_feature_extractor.feature_extractor.fc7', 'ufc7')
            elif new_k.startswith('roi_heads.relation.union_feature_extractor.rect_conv.'):
                suffix = new_k[len('roi_heads.relation.union_feature_extractor.rect_conv.'):]
                # Skip MaxPool (index 3, no params)
                if suffix.startswith('3.') or suffix == '3':
                    continue
                new_k = 'rect_conv.' + suffix

            # ---- Feature bank buffers ----
            if new_k == 'featurebank_key_array':
                new_k = 'fb_keys'
            elif new_k == 'featurebank_value_array':
                new_k = 'fb_values'

            # Skip keys that are NOT parameters (iteration counter, optimizer state)
            if new_k.startswith('roi_heads.') and 'predictor' not in new_k and 'box_feature' not in new_k and 'relation' not in k:
                continue
            if new_k.startswith('rpn.') or new_k.startswith('optimizer'):
                continue

            remapped[new_k] = v

            # ---- Classify into visual_extractor vs model keys ----
            if ve_weights is not None and (
                    new_k.startswith('backbone.') or
                    new_k.startswith('fpn.') or
                    new_k.startswith('fc6') or new_k.startswith('fc7') or
                    new_k.startswith('ufc6') or new_k.startswith('ufc7') or
                    new_k.startswith('box_fc6') or new_k.startswith('box_fc7') or
                    new_k.startswith('box_cls_score') or new_k.startswith('box_bbox_pred') or
                    new_k.startswith('rpn_') or
                    new_k.startswith('rect_conv.')):
                ve_weights[new_k] = v

        # ---- Load visual extractor weights immediately ----
        ve_loaded = 0
        if ve_weights is not None and visual_extractor is not None:
            ve_sd = visual_extractor.state_dict()
            loadable = {}
            for rk, rv in ve_weights.items():
                if rk in ve_sd and ve_sd[rk].shape == rv.shape:
                    loadable[rk] = rv
                elif rk in ve_sd and 'num_batches_tracked' in rk:
                    # torchvision BN adds num_batches_tracked; official ckpt doesn't have it
                    if ve_sd[rk].shape == ():
                        loadable[rk] = torch.tensor(0, dtype=torch.long)
                        if rk.endswith('num_batches_tracked'):
                            continue
            ve_loaded = len(loadable)
            visual_extractor.load_state_dict(loadable, strict=False)
            print(f"[RA-SGG] Loaded {ve_loaded} backbone+FPN+box_extractor weights into visual extractor")

        # ---- Stats ----
        model_keys = set(self.state_dict().keys())
        remapped_keys = set(remapped.keys())
        matched = model_keys & remapped_keys
        only_model = model_keys - remapped_keys

        if only_model:
            model_missing = sorted(only_model)
            print(f"[RA-SGG] {len(model_missing)} model keys not in checkpoint "
                  f"(will use random init): {model_missing[:6]}...")

        print(f"[RA-SGG] Remapped {len(remapped)} keys → {len(matched)} model matched "
              f"+ {ve_loaded} visual-extractor matched "
              f"({len(only_model)} randomly initialized)")

        return remapped

    # ---------- Forward ----------

    def forward(
        self,
        roi_features: torch.Tensor,
        union_features: Optional[torch.Tensor],
        labels: torch.Tensor,
        boxes: torch.Tensor,
        rel_labels: torch.Tensor,
        return_obj_preds: bool = False,
        logger=None,
        cur_iter: int = 0,
    ) -> dict:
        """RA-SGG forward.

        Training mode (self.training=True and memory bank available):
          Full RA-SGG pipeline with retrieval, selection, augmentation.

        Inference mode or no memory bank:
          Standard PE-Net forward (cosine similarity classification).
        """
        use_retrieval = self.training and self.fb_keys.numel() > 0

        # ---- Step 1: Get PE-Net intermediate outputs ----
        N = roi_features.size(0)
        device = roi_features.device

        add_losses = {}

        # ---- Object label refinement ----
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, labels, boxes)

        # ---- Steps 2-12: PE-Net forward (replicated for intermediate access) ----
        entity_rep = self.post_emb(roi_features)
        entity_rep = entity_rep.view(N, 2, self.mlp_dim)
        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)

        entity_embeds = self.obj_embed(entity_preds)

        pairs = generate_object_pairs(N, device)
        if pairs.numel() == 0:
            return {
                'rel_logits': roi_features.new_zeros(0, self.num_predicates),
                'entity_dists': entity_dists,
                'pair_indices': pairs,
                'sub_boxes': boxes.new_zeros(0, 4),
                'obj_boxes': boxes.new_zeros(0, 4),
                'obj_labels': entity_preds,
                'add_losses': {},
                'add_data': {},
            }

        s_idx = pairs[:, 0]
        o_idx = pairs[:, 1]

        s_embed = self.W_sub(entity_embeds[s_idx])
        o_embed = self.W_obj(entity_embeds[o_idx])

        sem_sub = self.vis2sem(sub_rep[s_idx])
        sem_obj = self.vis2sem(obj_rep[o_idx])

        gate_sem_sub = torch.sigmoid(self.gate_sub(torch.cat([s_embed, sem_sub], dim=-1)))
        gate_sem_obj = torch.sigmoid(self.gate_obj(torch.cat([o_embed, sem_obj], dim=-1)))

        sub = s_embed + sem_sub * gate_sem_sub
        obj = o_embed + sem_obj * gate_sem_obj

        sub = self.norm_sub(self.dropout_sub(F.relu(self.linear_sub(sub))) + sub)
        obj = self.norm_obj(self.dropout_obj(F.relu(self.linear_obj(obj))) + obj)

        fusion_so = fusion_func(sub, obj)

        # Union debiasing
        if self.use_union and union_features is not None:
            sem_pred = self.vis2sem(self.down_samp(union_features))
            gate_sem_pred = torch.sigmoid(self.gate_pred(torch.cat([fusion_so, sem_pred], dim=-1)))
            rel_rep = fusion_so - sem_pred * gate_sem_pred
        else:
            union_vis = (sub_rep[s_idx] + obj_rep[o_idx]) / 2.0
            sem_pred = self.vis2sem(union_vis)
            gate_sem_pred = torch.sigmoid(self.gate_pred(torch.cat([fusion_so, sem_pred], dim=-1)))
            rel_rep = fusion_so - sem_pred * gate_sem_pred

        rel_rep = self.norm_rel_rep(
            self.dropout_rel_rep(F.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        predicate_proto = self.W_pred(self.rel_embed.weight)

        # ---- Pre-projection embeddings (mlp_dim=2048) — used for retrieval ----
        rel_rep_preproj = rel_rep  # BEFORE project_head

        add_data = {}
        if not use_retrieval:
            # ---- Standard PE-Net: project now for classification ----
            rel_rep_proj = self.project_head(self.dropout_rel(F.relu(rel_rep)))
            predicate_proto_proj = self.project_head(self.dropout_pred(F.relu(predicate_proto)))
            rel_rep_norm = rel_rep_proj / (rel_rep_proj.norm(dim=1, keepdim=True) + 1e-8)
            pred_proto_norm = predicate_proto_proj / (predicate_proto_proj.norm(dim=1, keepdim=True) + 1e-8)
            rel_dists = rel_rep_norm @ pred_proto_norm.t() * self.logit_scale.exp().clamp(max=100.0)

            if self.training:
                self._compute_regularization_losses(
                    predicate_proto_proj, pred_proto_norm, rel_rep_proj,
                    rel_labels, add_losses)

            return {
                'rel_logits': rel_dists,
                'entity_dists': entity_dists,
                'pair_indices': pairs,
                'sub_boxes': boxes[pairs[:, 0]],
                'obj_boxes': boxes[pairs[:, 1]],
                'obj_labels': entity_preds,
                'add_losses': add_losses,
                'add_data': {},
            }

        # ================================================================
        # RA-SGG Retrieval + Augmentation Pipeline (official ReTAGPENet)
        # ================================================================

        self._move_memory_to_device(device)

        # ---- RETRIEVAL (official: query with pre-projection rel_rep, mlp_dim=2048) ----
        with torch.no_grad():
            query_rels = rel_rep_preproj  # [P, mlp_dim] — BEFORE project_head
            key_rels = self.fb_keys       # [M, mlp_dim] — stored BEFORE project_head
            sim = torch.matmul(F.normalize(query_rels), F.normalize(key_rels).t())  # [P, M]
            # Retrieve 20*K neighbors, then use K after dropping self-match
            topk_vals, retrieved_idx = torch.topk(
                sim, min(20 * self.num_retrievals, key_rels.size(0)))

        # Extract retrieved values
        retrieved_value_pred = []
        retrieved_dists = []
        retrieved_ipss = []

        for i in range(retrieved_idx.shape[0]):
            ret_idx = retrieved_idx[i]
            # Remove self-retrieval: drop index 0, keep K
            ret_idx = ret_idx[1:self.num_retrievals + 1]

            ret_value = self.fb_values[ret_idx]  # [K, 3]
            ret_preds = ret_value[:, 2]  # [K]

            retrieved_value_pred.append(ret_preds)
            # Count distribution of retrieved predicates
            ret_dist = torch.sum(
                F.one_hot(ret_preds.clamp(min=0, max=self.num_predicates - 1),
                          num_classes=self.num_predicates), dim=0).float() / self.num_retrievals
            retrieved_dists.append(ret_dist)
            # IPS weights for retrieved predicates
            retrieved_ipss.append(self.rel_loss_weight[ret_preds])

        retrieved_value_pred = torch.stack(retrieved_value_pred)  # [P, K]
        retrieved_dists = torch.stack(retrieved_dists)  # [P, num_predicates]
        retrieved_ipss = torch.stack(retrieved_ipss)  # [P, K]

        del key_rels  # official explicitly deletes for memory

        # ---- PROJECT after retrieval (official: project_head AFTER retrieval) ----
        rel_rep_proj = self.project_head(self.dropout_rel(F.relu(rel_rep_preproj)))  # [P, mlp_dim*2]
        predicate_proto_proj = self.project_head(self.dropout_pred(F.relu(predicate_proto)))
        rel_rep_norm = rel_rep_proj / (rel_rep_proj.norm(dim=1, keepdim=True) + 1e-8)
        pred_proto_norm = predicate_proto_proj / (predicate_proto_proj.norm(dim=1, keepdim=True) + 1e-8)

        # ---- Cosine similarity classification ----
        rel_dists = rel_rep_norm @ pred_proto_norm.t() * self.logit_scale.exp().clamp(max=100.0)

        # ---- RELIABLE MULTI-LABEL SELECTION ----
        # rel_label_confidences: fraction of retrieved preds matching GT
        pos_ret_mask = (rel_labels.reshape(-1, 1) == retrieved_value_pred)
        rel_label_confidences = pos_ret_mask.sum(-1).float() / self.num_retrievals

        # FG change: low confidence + tail predicate in retrieved set
        mask_fg_change = (
            (rel_label_confidences < self.tau) &
            torch.any(torch.isin(retrieved_value_pred, self.tail_ids), dim=-1) &
            (rel_labels != 0)
        )
        mask_fg_nochange = (~mask_fg_change) & (rel_labels != 0)

        # BG change: tail predicate in retrieved set
        mask_bg_change = (
            torch.any(torch.isin(retrieved_value_pred, self.tail_ids), dim=-1) &
            (rel_labels == 0)
        )

        # Limit BG changes: at most num_correct_bg per image (official: first bg per image)
        # In per-image mode (single image per forward call), limit globally
        if mask_bg_change.any() and self.num_correct_bg > 0:
            bg_indices = torch.where(mask_bg_change)[0]
            # Keep only the first num_correct_bg indices
            keep_bg = torch.zeros_like(mask_bg_change)
            keep_bg[bg_indices[:self.num_correct_bg]] = True
            mask_bg_change = mask_bg_change & keep_bg

        mask_nochange = mask_fg_nochange
        mask_change = mask_fg_change | mask_bg_change

        # ---- UNBIASED AUGMENTATION (IPS sampling) ----
        # sample_prob = retrieved_dists * freq_rwt_weight
        sample_prob = retrieved_dists[mask_change] * self.rel_loss_weight.unsqueeze(0)
        sample_prob = sample_prob / torch.sum(sample_prob, dim=1, keepdim=True)

        mixup_labels = rel_labels.clone()
        if mask_change.any():
            sampled = torch.multinomial(sample_prob, 1).squeeze(-1)
            mixup_labels[mask_change] = sampled
        mixup_labels[mask_nochange] = rel_labels[mask_nochange]

        # ---- MIXUP ----
        if self.do_mixup:
            rel_labels_onehot = F.one_hot(rel_labels, num_classes=self.num_predicates).float()
            mixup_labels_onehot = F.onehot(mixup_labels, num_classes=self.num_predicates).float()
            beta_dist = Beta(
                torch.ones(len(rel_labels), device=device) * self.mixup_alpha,
                torch.ones(len(rel_labels), device=device) * self.mixup_beta)
            lambda_coef = beta_dist.sample().unsqueeze(1)
            rel_labels_onehot = lambda_coef * rel_labels_onehot + (1 - lambda_coef) * mixup_labels_onehot
            add_data['rel_labels_onehot'] = rel_labels_onehot
            add_data['lambda_coef'] = lambda_coef
        else:
            add_data['mixup_labels'] = mixup_labels

        # ---- PROTOTYPE REGULARIZATION (same as PE-Net) ----
        if self.training:
            target_proto_norm = pred_proto_norm.clone().detach()
            simil_mat = pred_proto_norm @ target_proto_norm.t()
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_predicates * self.num_predicates)
            add_losses['l21_loss'] = l21

            gamma2 = 7.0
            C = self.num_predicates
            proto_a = predicate_proto_proj.unsqueeze(1).expand(-1, C, -1)
            proto_b = predicate_proto_proj.detach().unsqueeze(0).expand(C, -1, -1)
            proto_dis_mat = (proto_a - proto_b).norm(dim=2) ** 2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1
            dist_loss = torch.clamp(-topK_proto_dis + gamma2, min=0).mean()
            add_losses['dist_loss2'] = dist_loss

            # Euclidean distance triplet loss (with mixup labels)
            gamma1 = 1.0
            rel_rep_expand = rel_rep_proj.unsqueeze(1).expand(-1, C, -1)
            proto_expand = predicate_proto_proj.unsqueeze(0).expand(rel_labels.size(0), -1, -1)
            distance_set = (rel_rep_expand - proto_expand).norm(dim=2) ** 2

            mask_neg = torch.ones(rel_labels.size(0), C, device=device)
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            mask_neg_mix = torch.ones(rel_labels.size(0), C, device=device)
            mask_neg_mix[torch.arange(rel_labels.size(0)), mixup_labels] = 0

            distance_set_neg = distance_set * mask_neg
            distance_set_neg_mix = distance_set * mask_neg_mix
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]
            distance_set_pos_mix = distance_set[torch.arange(rel_labels.size(0)), mixup_labels]

            sorted_dis_neg, _ = torch.sort(distance_set_neg, dim=1)
            sorted_dis_neg_mix, _ = torch.sort(distance_set_neg_mix, dim=1)

            topK_sorted_dis_neg = sorted_dis_neg[:, :11].sum(dim=1) / 10
            topK_sorted_dis_neg_mix = sorted_dis_neg_mix[:, :11].sum(dim=1) / 10

            if self.do_mixup:
                proto_euc_loss = (
                    lambda_coef.squeeze(-1) * (distance_set_pos - topK_sorted_dis_neg) +
                    (1 - lambda_coef.squeeze(-1)) * (distance_set_pos_mix - topK_sorted_dis_neg_mix) +
                    gamma1
                )
            else:
                proto_euc_loss = distance_set_pos - topK_sorted_dis_neg + gamma1

            loss_sum = torch.clamp(proto_euc_loss, min=0).mean()
            add_losses['loss_dis'] = loss_sum

            # ---- Retrieval statistics logging ----
            self._log_retrieval_stats(rel_labels, retrieved_value_pred, pos_ret_mask, cur_iter)

        return {
            'rel_logits': rel_dists,
            'entity_dists': entity_dists,  # from refine_obj_labels
            'pair_indices': pairs,
            'sub_boxes': boxes[pairs[:, 0]],
            'obj_boxes': boxes[pairs[:, 1]],
            'obj_labels': entity_preds,
            'add_losses': add_losses,
            'add_data': add_data,
            'rel_rep': rel_rep_proj,
            'predicate_proto': predicate_proto_proj,
            'predicate_proto_pre': predicate_proto,
            'rel_rep_norm': rel_rep_norm,
            'pred_proto_norm': pred_proto_norm,
        }

    def _log_retrieval_stats(self, rel_labels, retrieved_value_pred, pos_ret_mask, cur_iter):
        """Log retrieval statistics (matches official logging)."""
        with torch.no_grad():
            head_mask = torch.isin(rel_labels, self.head_ids)
            num_head = head_mask.sum().item()
            if num_head > 0:
                ret_success = (head_mask.unsqueeze(1).expand(-1, self.num_retrievals)) & pos_ret_mask
                self.pos_masks['head_poscnt'] += ret_success.sum().item()
                self.pos_masks['head_cnt'] += num_head * self.num_retrievals

            body_mask = torch.isin(rel_labels, self.body_ids)
            num_body = body_mask.sum().item()
            if num_body > 0:
                ret_success = (body_mask.unsqueeze(1).expand(-1, self.num_retrievals)) & pos_ret_mask
                self.pos_masks['body_poscnt'] += ret_success.sum().item()
                self.pos_masks['body_cnt'] += num_body * self.num_retrievals

            tail_mask = torch.isin(rel_labels, self.tail_ids)
            num_tail = tail_mask.sum().item()
            if num_tail > 0:
                ret_success = (tail_mask.unsqueeze(1).expand(-1, self.num_retrievals)) & pos_ret_mask
                self.pos_masks['tail_poscnt'] += ret_success.sum().item()
                self.pos_masks['tail_cnt'] += num_tail * self.num_retrievals

            bg_mask = (rel_labels == 0)
            num_bg = bg_mask.sum().item()
            if num_bg > 0:
                ret_success = (bg_mask.unsqueeze(1).expand(-1, self.num_retrievals)) & pos_ret_mask
                self.pos_masks['bg_poscnt'] += ret_success.sum().item()
                self.pos_masks['bg_cnt'] += num_bg * self.num_retrievals

            all_mask = (rel_labels != 0)
            num_all = all_mask.sum().item()
            if num_all > 0:
                ret_success = (all_mask.unsqueeze(1).expand(-1, self.num_retrievals)) & pos_ret_mask
                self.pos_masks['all_poscnt'] += ret_success.sum().item()
                self.pos_masks['all_cnt'] += num_all * self.num_retrievals

            if cur_iter % 200 == 0:
                pos_ret_all = (self.pos_masks['all_poscnt'] / max(self.pos_masks['all_cnt'], 1))
                pos_ret_bg = (self.pos_masks['bg_poscnt'] / max(self.pos_masks['bg_cnt'], 1))
                pos_ret_head = (self.pos_masks['head_poscnt'] / max(self.pos_masks['head_cnt'], 1))
                pos_ret_body = (self.pos_masks['body_poscnt'] / max(self.pos_masks['body_cnt'], 1))
                pos_ret_tail = (self.pos_masks['tail_poscnt'] / max(self.pos_masks['tail_cnt'], 1))
                print(f"[RA-SGG] iter={cur_iter} | Retrieval positive: "
                      f"all={pos_ret_all:.2f} bg={pos_ret_bg:.2f} "
                      f"head={pos_ret_head:.2f} body={pos_ret_body:.2f} tail={pos_ret_tail:.2f}")

                self.pos_retrieval_prop['all'].append(pos_ret_all)
                self.pos_retrieval_prop['bg'].append(pos_ret_bg)
                self.pos_retrieval_prop['head'].append(pos_ret_head)
                self.pos_retrieval_prop['body'].append(pos_ret_body)
                self.pos_retrieval_prop['tail'].append(pos_ret_tail)

                self.pos_masks = {
                    'all_poscnt': 0, 'all_cnt': 0,
                    'bg_poscnt': 0, 'bg_cnt': 0,
                    'head_poscnt': 0, 'head_cnt': 0,
                    'body_poscnt': 0, 'body_cnt': 0,
                    'tail_poscnt': 0, 'tail_cnt': 0,
                }

    # ---------- Memory Bank Construction (static method) ----------

    @staticmethod
    def build_memory_bank(
        ckpt_path: str,
        dataname: str = 'VisualGenome',
        data_root: str = './data',
        output_dir: str = 'data/VisualGenome/ra_sgg/',
        max_per_triplet: int = 100,
        mode: str = 'predcls',
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        """Build RA-SGG memory bank from a pre-trained PENetBase checkpoint.

        Args:
            ckpt_path: Path to a trained PENetBase / RASGGModel .ckpt checkpoint.
            dataname: Dataset name ('VisualGenome').
            data_root: Root data directory.
            output_dir: Directory to save memory bank files.
            max_per_triplet: Max entries stored per unique (sub, obj) pair.
            mode: 'predcls' (default — uses GT boxes and labels).
            batch_size: DataLoader batch size.
            num_workers: DataLoader workers.
        """
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

        from src.models.backbone import build_visual_extractor
        from utils.main_utils import get_dataset
        from types import SimpleNamespace

        print(f"[MemoryBank] Building memory bank from: {ckpt_path}")
        print(f"[MemoryBank] Mode: {mode}, Max per triplet: {max_per_triplet}")

        # ---- Load pre-trained model ----
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create a minimal model for extraction
        model = PENetBase(
            num_classes=151,
            num_predicates=51,
            visual_dim=4096,        # matches FPN+MLP output
            mlp_dim=2048,
            embed_dim=300,
            use_freq_bias=False,
        )

        # ---- Load checkpoint ----
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        state_dict = ckpt.get('state_dict', ckpt)
        # Strip 'model.' prefix if present
        clean_state = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                clean_state[k[6:]] = v
            else:
                clean_state[k] = v
        model.load_state_dict(clean_state, strict=False)
        model.to(device)
        model.eval()

        # ---- Load data ----
        config = {
            'dataname': dataname,
            'data_root': data_root,
            'batch_size': batch_size,
            'val_batch_size': batch_size,
            'num_workers': num_workers,
            'dist': False,
            'use_augment': False,
            'drop_last': False,
            'seed': 42,
        }
        SimpleNamespace(**config)
        train_loader, _, _ = get_dataset(dataname, config)

        # ---- Build visual extractor (FPN+MLP, matches official) ----
        visual_config = SimpleNamespace(
            backbone_arch='resnet101',
            backbone_pretrained=True,
            backbone_frozen=True,
            roi_output_size=7,
        )
        visual_extractor = build_visual_extractor(visual_config).to(device)
        visual_extractor.eval()

        # ---- Extract features ----
        featurebank_dict = {}
        count_dict = {}
        total_pairs = 0

        from tqdm import tqdm

        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Extracting features")):
            images, targets = batch[0], batch[1]

            if not isinstance(images, (list, tuple)):
                if hasattr(images, 'tensors'):
                    images = [img for img in images.tensors]
                else:
                    continue

            for img_idx, (img, target) in enumerate(zip(images, targets)):
                if torch.is_tensor(target.get('rel_annotations', None)):
                    rel_anns = target['rel_annotations']
                else:
                    continue

                if rel_anns.numel() == 0:
                    continue

                boxes = target['boxes'].to(device)
                labels = target['labels'].to(device)
                img_size = target.get('size', target.get('orig_size'))
                if img_size is None:
                    img_size = torch.tensor(img.shape[-2:], dtype=torch.float32)
                img_size = img_size.to(device)
                img = img.to(device)

                # Extract visual features via FPN+Pooler+MLP (matches official pipeline)
                with torch.no_grad():
                    roi_feats_list, fpn_feats = visual_extractor(
                        [img], [boxes], [img_size], return_feature_maps=True)
                    roi_feats = roi_feats_list[0]  # [N, 4096]

                    # Run PE-Net forward to get relation embeddings
                    N = roi_feats.size(0)
                    pairs = generate_object_pairs(N, device)
                    P = pairs.size(0)

                    # Compute relation embeddings (simplified — full PE-Net forward)
                    entity_rep = model.post_emb(roi_feats)
                    entity_rep = entity_rep.view(N, 2, model.mlp_dim)
                    sub_rep = entity_rep[:, 1]
                    obj_rep = entity_rep[:, 0]
                    entity_embeds = model.obj_embed(labels)
                    s_idx = pairs[:, 0]
                    o_idx = pairs[:, 1]

                    s_embed = model.W_sub(entity_embeds[s_idx])
                    o_embed = model.W_obj(entity_embeds[o_idx])
                    sem_sub = model.vis2sem(sub_rep[s_idx])
                    sem_obj = model.vis2sem(obj_rep[o_idx])

                    gate_s = torch.sigmoid(model.gate_sub(torch.cat([s_embed, sem_sub], dim=-1)))
                    gate_o = torch.sigmoid(model.gate_obj(torch.cat([o_embed, sem_obj], dim=-1)))

                    sub = s_embed + sem_sub * gate_s
                    obj = o_embed + sem_obj * gate_o
                    sub = model.norm_sub(model.dropout_sub(F.relu(model.linear_sub(sub))) + sub)
                    obj = model.norm_obj(model.dropout_obj(F.relu(model.linear_obj(obj))) + obj)

                    fusion_so = fusion_func(sub, obj)

                    # Extract union features via FPN pooler (matching training pipeline)
                    union_feats = visual_extractor.extract_union_features(
                        fpn_feats, [boxes], [img_size], [pairs])
                    sem_pred = model.vis2sem(model.down_samp(union_feats[0]))
                    gate_pred = torch.sigmoid(model.gate_pred(torch.cat([fusion_so, sem_pred], dim=-1)))
                    rel_rep = fusion_so - sem_pred * gate_pred

                    rel_rep = model.norm_rel_rep(
                        model.dropout_rel_rep(F.relu(model.linear_rel_rep(rel_rep))) + rel_rep)

                    # Store PRE-projection embeddings (mlp_dim=2048, matching official)
                    # Official feature bank stores BEFORE project_head
                    rel_rep_store = rel_rep.detach()  # [mlp_dim] — before project_head

                # ---- Build triplet map and store ----
                # Map pair indices to rel_annotations
                pair_to_rel = {}
                for ann in rel_anns:
                    s, o, p = int(ann[0]), int(ann[1]), int(ann[2])
                    pair_to_rel[(s, o)] = p

                for p_idx in range(P):
                    s = int(s_idx[p_idx].item())
                    o = int(o_idx[p_idx].item())
                    if (s, o) in pair_to_rel:
                        pred = pair_to_rel[(s, o)]
                        if pred == 0:
                            continue
                        tri_key = f"{s}_{o}_{pred}"
                        if tri_key not in featurebank_dict:
                            featurebank_dict[tri_key] = []
                            count_dict[tri_key] = 0
                        if count_dict[tri_key] < max_per_triplet:
                            count_dict[tri_key] += 1
                            featurebank_dict[tri_key].append((
                                rel_rep_store[p_idx].cpu(),
                                torch.tensor([s, o, pred], dtype=torch.long),
                            ))
                            total_pairs += 1

        # ---- Save memory bank ----
        os.makedirs(output_dir, exist_ok=True)

        all_keys = []
        all_values = []
        for tri_key, entries in featurebank_dict.items():
            for key, value in entries:
                all_keys.append(key)
                all_values.append(value)

        if len(all_keys) == 0:
            print("[MemoryBank] WARNING: No entries collected!")
            return

        keys_tensor = torch.stack(all_keys)  # [M, D]
        values_tensor = torch.stack(all_values)  # [M, 3]

        print(f"[MemoryBank] Total entries: {len(all_keys)}")
        print(f"[MemoryBank] Unique triplets: {len(featurebank_dict)}")
        print(f"[MemoryBank] Keys shape: {keys_tensor.shape}")
        print(f"[MemoryBank] Values shape: {values_tensor.shape}")

        bank = {
            'key': keys_tensor.numpy(),
            'value': values_tensor.numpy(),
        }
        output_path = os.path.join(output_dir, f'{mode}_fb_train.npy')
        np.save(output_path, bank)
        print(f"[MemoryBank] Saved to: {output_path}")

        # Also save inverse propensity
        pred_counts = torch.zeros(51, dtype=torch.float32)
        for val in all_values:
            pred_counts[int(val[2].item())] += 1
        pred_counts[0] = pred_counts[1:].sum() * 20  # bg count
        inv_propensity = (1.0 - 0.99999) / (1.0 - 0.99999 ** pred_counts)
        median = torch.median(inv_propensity[1:])
        inv_propensity = inv_propensity / median
        inv_propensity[0] = torch.min(inv_propensity[1:])
        torch.save(inv_propensity, os.path.join(output_dir, 'inverse_propensity.pt'))
        print("[MemoryBank] Saved inverse propensity weights.")


# ============================================================================
# Builder
# ============================================================================

def build_ra_sgg(args) -> RASGGModel:
    """Build RA-SGG model from config args."""
    # Estimate predicate frequencies for class-balanced loss weight
    pred_count_dict = None
    data_root = getattr(args, 'data_root', './data')
    try:
        pred_freqs = _predicate_frequencies_from_dataset(data_root)
        if pred_freqs is not None:
            pred_count_dict = {
                name: float(pred_freqs[i]) for i, name in enumerate(VG_PREDICATE_NAMES)
            }
    except Exception:
        pass

    model = RASGGModel(
        num_classes=getattr(args, 'entity_nums', 151),
        num_predicates=getattr(args, 'rel_nums', 51),
        visual_dim=getattr(args, 'visual_dim', 2048),
        mlp_dim=getattr(args, 'ra_sgg_mlp_dim', 2048),
        embed_dim=getattr(args, 'ra_sgg_embed_dim', 300),
        hidden_dim=getattr(args, 'hidden_dim', 512),
        dropout=getattr(args, 'dropout', 0.2),
        use_freq_bias=getattr(args, 'ra_sgg_use_bias', False),
        freq_bias_eps=getattr(args, 'freq_bias_eps', 1e-12),
        use_union=getattr(args, 'ra_sgg_use_union', True),
        mode=getattr(args, 'eval_mode', 'predcls'),
        glove_dir=getattr(args, 'ra_sgg_glove_dir', 'data/glove'),
        rel_loss_type=getattr(args, 'rel_loss_type', 'ce'),
        reweight_beta=getattr(args, 'reweight_beta', 0.99999),
        pred_count_dict=pred_count_dict,
        # RA-SGG parameters
        num_retrievals=getattr(args, 'ra_sgg_num_retrievals', 10),
        threshold=getattr(args, 'ra_sgg_threshold', 0.3),
        mixup=getattr(args, 'ra_sgg_mixup', True),
        mixup_alpha=getattr(args, 'ra_sgg_mixup_alpha', 20.0),
        mixup_beta=getattr(args, 'ra_sgg_mixup_beta', 5.0),
        num_correct_bg=getattr(args, 'ra_sgg_num_correct_bg', 1),
        memory_bank_path=getattr(args, 'ra_sgg_memory_bank_path', None),
    )
    return model
