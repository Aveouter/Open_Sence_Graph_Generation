"""PE-NET SGDet detector proposal components.

Eval-only port of the official maskrcnn-benchmark detector path needed before
PE-NET relation prediction:

  FPN features -> RPN proposals -> detector box predictor -> relation proposals

The relation predictor must re-extract ROI features with the separate relation
box extractor; detector ROI features are intentionally not returned as the
relation features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms as torchvision_nms

from src.models.backbone import PENetBoxFeatureExtractor


@dataclass
class PENetDetectorOutput:
    """Detector fields consumed by the PE-NET relation predictor."""

    boxes: torch.Tensor
    labels: torch.Tensor
    obj_dists: torch.Tensor
    boxes_per_cls: torch.Tensor


class BoxCoder:
    """Official maskrcnn-benchmark box coder."""

    def __init__(self, weights: tuple[float, float, float, float], bbox_xform_clip=None):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip or math.log(1000.0 / 16)

    def decode(self, rel_codes: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes.to(rel_codes.dtype)
        to_remove = 1
        widths = boxes[:, 2] - boxes[:, 0] + to_remove
        heights = boxes[:, 3] - boxes[:, 1] + to_remove
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1
        return pred_boxes


class PENetRPNHead(nn.Module):
    """Official ``SingleConvRPNHead`` with 4 anchors per location."""

    def __init__(self, in_channels: int = 256, mid_channels: int = 256, num_anchors: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(mid_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1, stride=1)
        for layer in (self.conv, self.cls_logits, self.bbox_pred):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features: tuple[torch.Tensor, ...]):
        logits, bbox_reg = [], []
        for feature in features:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class PENetBoxPredictor(nn.Module):
    """Official ``FPNPredictor`` for the VG detector box head."""

    def __init__(self, representation_size: int = 4096, num_classes: int = 151):
        super().__init__()
        self.cls_score = nn.Linear(representation_size, num_classes)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cls_score(x), self.bbox_pred(x)


def _generate_anchors(base_size: int, scale: float, ratios: tuple[float, ...]) -> torch.Tensor:
    """Generate official-style anchors for one FPN level."""
    area = float(base_size * base_size)
    anchors = []
    for ratio in ratios:
        w = round(math.sqrt(area / ratio))
        h = round(w * ratio)
        ws = w * float(scale) / base_size
        hs = h * float(scale) / base_size
        x_ctr = 0.5 * (base_size - 1)
        y_ctr = 0.5 * (base_size - 1)
        anchors.append(
            [
                x_ctr - 0.5 * (ws - 1),
                y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1),
                y_ctr + 0.5 * (hs - 1),
            ]
        )
    return torch.tensor(anchors, dtype=torch.float32)


def _permute_and_flatten(layer: torch.Tensor, n: int, a: int, c: int, h: int, w: int):
    layer = layer.view(n, -1, c, h, w)
    layer = layer.permute(0, 3, 4, 1, 2)
    return layer.reshape(n, -1, c)


def _clip_boxes(boxes: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
    h, w = float(image_size[0].item()), float(image_size[1].item())
    boxes[:, 0].clamp_(min=0, max=w - 1)
    boxes[:, 1].clamp_(min=0, max=h - 1)
    boxes[:, 2].clamp_(min=0, max=w - 1)
    boxes[:, 3].clamp_(min=0, max=h - 1)
    return boxes


def _remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    return ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)


def _xyxy_to_cxcywh_norm(boxes: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
    h, w = image_size[0].float(), image_size[1].float()
    return torch.stack(
        [
            (boxes[:, 0] + boxes[:, 2]) * 0.5 / w,
            (boxes[:, 1] + boxes[:, 3]) * 0.5 / h,
            (boxes[:, 2] - boxes[:, 0] + 1) / w,
            (boxes[:, 3] - boxes[:, 1] + 1) / h,
        ],
        dim=-1,
    )


class PENetSGDetProposalGenerator(nn.Module):
    """Official-style PE-NET detector proposal generator for eval-only SGDet."""

    def __init__(
        self,
        num_classes: int = 151,
        rpn_pre_nms_top_n: int = 6000,
        rpn_post_nms_top_n: int = 1000,
        detections_per_img: int = 80,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.rpn_head = PENetRPNHead()
        self.box_predictor = PENetBoxPredictor(num_classes=num_classes)
        self.rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
        self.rpn_post_nms_top_n = rpn_post_nms_top_n
        self.rpn_nms_thresh = 0.7
        self.rpn_min_size = 0
        self.box_score_thresh = 0.01
        self.box_nms_thresh = 0.3
        self.post_nms_per_cls_topn = 300
        self.detections_per_img = detections_per_img
        self.strides = (4, 8, 16, 32, 64)
        self.anchor_sizes = (32, 64, 128, 256, 512)
        self.aspect_ratios = (0.23232838, 0.63365731, 1.28478321, 3.15089189)
        anchors = [
            _generate_anchors(stride, size, self.aspect_ratios)
            for stride, size in zip(self.strides, self.anchor_sizes)
        ]
        for index, anchor in enumerate(anchors):
            self.register_buffer(f"cell_anchor_{index}", anchor)

    def _cell_anchor(self, index: int) -> torch.Tensor:
        return getattr(self, f"cell_anchor_{index}")

    def _grid_anchors(self, features: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        all_anchors = []
        for i, feature in enumerate(features):
            _, _, h, w = feature.shape
            stride = self.strides[i]
            base = self._cell_anchor(i).to(feature.device)
            shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=feature.device)
            shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=feature.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shifts = torch.stack(
                (shift_x.reshape(-1), shift_y.reshape(-1), shift_x.reshape(-1), shift_y.reshape(-1)),
                dim=1,
            )
            all_anchors.append((shifts[:, None, :] + base[None, :, :]).reshape(-1, 4))
        return all_anchors

    def _rpn_proposals(self, fpn_features: tuple[torch.Tensor, ...], image_size: torch.Tensor) -> torch.Tensor:
        objectness, regression = self.rpn_head(fpn_features)
        anchors = self._grid_anchors(fpn_features)
        proposals_all, scores_all = [], []
        for obj, reg, anc in zip(objectness, regression, anchors):
            n, a, h, w = obj.shape
            if n != 1:
                raise ValueError("PENet SGDet detector expects per-image feature batches.")
            obj = _permute_and_flatten(obj, n, a, 1, h, w).view(n, -1).sigmoid()
            reg = _permute_and_flatten(reg, n, a, 4, h, w)
            num_anchors = a * h * w
            pre_nms = min(self.rpn_pre_nms_top_n, num_anchors)
            scores, topk_idx = obj.topk(pre_nms, dim=1, sorted=True)
            reg = reg[torch.arange(n, device=obj.device)[:, None], topk_idx]
            anc = anc.reshape(1, -1, 4)[:, topk_idx.squeeze(0), :]
            proposals = self.rpn_box_coder.decode(reg.reshape(-1, 4), anc.reshape(-1, 4))
            proposals = _clip_boxes(proposals.view(-1, 4), image_size)
            keep = _remove_small_boxes(proposals, self.rpn_min_size)
            proposals = proposals[keep]
            scores = scores.reshape(-1)[keep]
            keep = torchvision_nms(proposals, scores, self.rpn_nms_thresh)
            keep = keep[: self.rpn_post_nms_top_n]
            proposals_all.append(proposals[keep])
            scores_all.append(scores[keep])

        proposals = torch.cat(proposals_all, dim=0)
        scores = torch.cat(scores_all, dim=0)
        top_n = min(self.rpn_post_nms_top_n, scores.numel())
        _, inds_sorted = torch.topk(scores, top_n, dim=0, sorted=True)
        return proposals[inds_sorted]

    def _filter_box_results(
        self,
        boxes_per_cls: torch.Tensor,
        class_prob: torch.Tensor,
        image_size: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result_labels = []
        orig_inds = []
        inds_all = class_prob > self.box_score_thresh
        for cls_idx in range(1, self.num_classes):
            inds = inds_all[:, cls_idx].nonzero().squeeze(1)
            if inds.numel() == 0:
                continue
            scores_j = class_prob[inds, cls_idx]
            boxes_j = boxes_per_cls[inds, cls_idx]
            keep = torchvision_nms(boxes_j, scores_j, self.box_nms_thresh)
            keep = keep[: self.post_nms_per_cls_topn]
            inds = inds[keep]
            result_labels.append(torch.full((keep.numel(),), cls_idx, dtype=torch.long, device=boxes_j.device))
            orig_inds.append(inds)

        if not orig_inds:
            empty_long = boxes_per_cls.new_zeros(0).long()
            return empty_long, empty_long, boxes_per_cls.new_zeros(0, self.num_classes, 4)

        # Official duplicate filtering: one class per original proposal.
        inds_all[:, 0] = False
        keep_mask = torch.zeros_like(inds_all)
        for inds, labels in zip(orig_inds, result_labels):
            keep_mask[inds, labels] = True
        dist_scores = class_prob * keep_mask.float()
        scores_pre, labels_pre = dist_scores.max(1)
        final_inds = scores_pre.nonzero().squeeze(1)
        boxes = boxes_per_cls[final_inds, labels_pre[final_inds]]
        scores = scores_pre[final_inds]
        labels = labels_pre[final_inds]

        if boxes.numel() > 0:
            boxes = _clip_boxes(boxes, image_size)
        if boxes.size(0) > self.detections_per_img:
            threshold, _ = torch.kthvalue(
                scores.cpu(),
                boxes.size(0) - self.detections_per_img + 1,
            )
            keep = (scores >= threshold.item()).nonzero().squeeze(1)
            final_inds = final_inds[keep]
            labels = labels[keep]
        return final_inds, labels, boxes_per_cls[final_inds]

    def forward(
        self,
        fpn_features: tuple[torch.Tensor, ...],
        box_extractor: PENetBoxFeatureExtractor,
        image_size: torch.Tensor,
    ) -> PENetDetectorOutput:
        fpn_features_batched = tuple(
            f.unsqueeze(0) if f.dim() == 3 else f for f in fpn_features
        )
        fpn_features_single = tuple(
            f.squeeze(0) if f.dim() == 4 and f.size(0) == 1 else f
            for f in fpn_features
        )
        proposals_xyxy = self._rpn_proposals(fpn_features_batched, image_size)
        if proposals_xyxy.numel() == 0:
            device = fpn_features_batched[0].device
            return PENetDetectorOutput(
                boxes=torch.zeros(0, 4, device=device),
                labels=torch.zeros(0, dtype=torch.long, device=device),
                obj_dists=torch.zeros(0, self.num_classes, device=device),
                boxes_per_cls=torch.zeros(0, self.num_classes, 4, device=device),
            )

        proposal_boxes = _xyxy_to_cxcywh_norm(proposals_xyxy, image_size)
        roi_feats = box_extractor(fpn_features_single, proposal_boxes, image_size)
        class_logits, box_regression = self.box_predictor(roi_feats)
        class_prob = F.softmax(class_logits, dim=-1)
        pred_boxes = self.box_coder.decode(box_regression, proposals_xyxy)
        pred_boxes = _clip_boxes(pred_boxes, image_size)
        boxes_per_cls = pred_boxes.reshape(-1, self.num_classes, 4)
        orig_inds, labels, final_boxes_per_cls = self._filter_box_results(
            boxes_per_cls,
            class_prob,
            image_size,
        )
        if orig_inds.numel() == 0:
            device = fpn_features_batched[0].device
            return PENetDetectorOutput(
                boxes=torch.zeros(0, 4, device=device),
                labels=torch.zeros(0, dtype=torch.long, device=device),
                obj_dists=torch.zeros(0, self.num_classes, device=device),
                boxes_per_cls=torch.zeros(0, self.num_classes, 4, device=device),
            )
        selected_boxes = final_boxes_per_cls[
            torch.arange(labels.numel(), device=labels.device),
            labels,
        ]
        return PENetDetectorOutput(
            boxes=_xyxy_to_cxcywh_norm(selected_boxes, image_size),
            labels=labels.long(),
            obj_dists=class_logits[orig_inds],
            boxes_per_cls=final_boxes_per_cls,
        )
