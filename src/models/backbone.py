"""
Visual backbone for classical SGG methods.

Supports ResNet / ResNeXt backbones with optional FPN neck for PENet.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.models import (
    resnet50,
    resnet101,
    resnext101_32x8d,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNeXt101_32X8D_Weights,
)
from typing import List, Tuple


# =========================================================================
# ResNet / ResNeXt backbone
# =========================================================================


class ResNetBackbone(nn.Module):
    _ARCH_BUILDERS = {
        "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
        "resnet101": (resnet101, ResNet101_Weights.IMAGENET1K_V1),
        "resnext101_32x8d": (resnext101_32x8d, ResNeXt101_32X8D_Weights.IMAGENET1K_V2),
    }

    def __init__(self, arch="resnet50", pretrained=True, frozen=True):
        super().__init__()
        if arch not in self._ARCH_BUILDERS:
            raise ValueError(f"Unsupported arch: {arch}")
        builder, weight_cls = self._ARCH_BUILDERS[arch]
        model = builder(weights=weight_cls if pretrained else None)
        self.arch = arch
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.output_dim = 2048
        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, images, return_all_scales=False):
        sq = images.dim() == 3
        if sq:
            images = images.unsqueeze(0)
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # stride 4
        x1 = self.layer1(x)  # stride 4
        x2 = self.layer2(x1)  # stride 8
        x3 = self.layer3(x2)  # stride 16
        x4 = self.layer4(x3)  # stride 32
        if return_all_scales:
            r = {4: x1, 8: x2, 16: x3, 32: x4}
        else:
            r = x4
        if sq:
            if return_all_scales:
                r = {k: v.squeeze(0) for k, v in r.items()}
            else:
                r = r.squeeze(0)
        return r

    @property
    def stride(self):
        return 32


# =========================================================================
# ROI align extractor (used by Motifs, VCTree, etc.)
# =========================================================================


class ROIAlignExtractor(nn.Module):
    def __init__(self, output_size=7, pool="avg"):
        super().__init__()
        self.output_size = output_size
        self.pool = pool

    def forward(self, feature_map, boxes, image_size):
        C, H_f, W_f = feature_map.shape
        device = feature_map.device
        N = boxes.size(0)
        if N == 0:
            return feature_map.new_zeros(0, C)
        img_h, img_w = image_size[0].float(), image_size[1].float()
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        scale_x, scale_y = W_f / img_w, H_f / img_h
        batch_idx = torch.zeros(N, 1, device=device)
        rois = torch.cat(
            [
                batch_idx,
                (x1 * scale_x).unsqueeze(1),
                (y1 * scale_y).unsqueeze(1),
                (x2 * scale_x).unsqueeze(1),
                (y2 * scale_y).unsqueeze(1),
            ],
            dim=1,
        )
        fm_b = feature_map.unsqueeze(0)
        roi_feats = roi_align(
            fm_b,
            rois,
            output_size=(self.output_size, self.output_size),
            spatial_scale=1.0,
            aligned=True,
        )
        if self.pool == "avg":
            roi_feats = F.adaptive_avg_pool2d(roi_feats, (1, 1))
            return roi_feats.squeeze(-1).squeeze(-1)
        return roi_feats.flatten(1)


# =========================================================================
# Visual feature extractor (Motifs/VCTree-compatible)
# =========================================================================


class VisualFeatureExtractor(nn.Module):
    def __init__(
        self, arch="resnet50", pretrained=True, frozen=True, roi_output_size=7
    ):
        super().__init__()
        self.backbone = ResNetBackbone(arch, pretrained, frozen)
        self.roi_align = ROIAlignExtractor(output_size=roi_output_size, pool="avg")
        self.output_dim = self.backbone.output_dim

    def forward(self, images, boxes_list, image_sizes, return_feature_maps=False):
        results = []
        for img, boxes, sz in zip(images, boxes_list, image_sizes):
            img = img.to(next(self.backbone.parameters()).device)
            boxes, sz = boxes.to(img.device), sz.to(img.device)
            with torch.set_grad_enabled(
                not all(not p.requires_grad for p in self.backbone.parameters())
            ):
                fm = self.backbone(img)
            feats = self.roi_align(fm, boxes, sz)
            results.append((feats, fm) if return_feature_maps else feats)
        return results

    def to(self, device):
        super().to(device)
        return self


def build_visual_extractor(args):
    return VisualFeatureExtractor(
        arch=getattr(args, "backbone_arch", "resnet50"),
        pretrained=getattr(args, "backbone_pretrained", True),
        frozen=getattr(args, "backbone_frozen", True),
        roi_output_size=getattr(args, "roi_output_size", 7),
    )


# =========================================================================
# FPN Neck — exact match of official maskrcnn_benchmark FPN
# =========================================================================


class _ConvBlock(nn.Module):
    """Conv2d used as `conv_block` by official FPN."""

    def __init__(self, in_c, out_c, kernel, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, kernel // 2)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class LastLevelMaxPool(nn.Module):
    """P6 = maxpool(P5)."""

    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class FPNNeck(nn.Module):
    """Feature Pyramid Network — exact match of official ``FPN`` class.

    Pipeline:
      C_i → 1×1 lateral conv → 256-dim
      Top-down: P_{i+1} ↑2 (nearest) + lateral_i → 3×3 smoothing → P_i
      P6 = maxpool(P5)
    """

    def __init__(self, in_channels_list=(256, 512, 1024, 2048), out_channels=256):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_c in in_channels_list:
            self.inner_blocks.append(_ConvBlock(in_c, out_channels, 1))
            self.layer_blocks.append(_ConvBlock(out_channels, out_channels, 3))
        self.top_blocks = LastLevelMaxPool()

    def forward(self, x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """x: (C2, C3, C4, C5) → (P2, P3, P4, P5, P6)."""
        # Backbone returns [C,H,W]; add batch dim for Conv2d
        sq = x[0].dim() == 3
        if sq:
            x = tuple(v.unsqueeze(0) for v in x)

        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        for feat, ib, lb in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            last_inner = ib(feat) + inner_top_down
            results.insert(0, lb(last_inner))
        if self.top_blocks is not None:
            results.extend(self.top_blocks(results[-1]))

        if sq:
            results = [r.squeeze(0) for r in results]
        return tuple(results)


# =========================================================================
# LevelMapper + FPNPooler — exact match of official Pooler
# =========================================================================


class LevelMapper:
    """FPN level assignment matching official ``LevelMapper``.

    target_lvl = floor(lvl0 + log2(sqrt(area) / s0 + eps))
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        self.k_min, self.k_max = k_min, k_max
        self.s0, self.lvl0, self.eps = canonical_scale, canonical_level, eps

    def __call__(self, boxes_xyxy: List[torch.Tensor]) -> torch.Tensor:
        areas = torch.cat(
            [(b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1) for b in boxes_xyxy],
            dim=0,
        )
        s = torch.sqrt(areas)
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min


class FPNPooler(nn.Module):
    """FPN-aware ROI pooler matching official ``Pooler``.

    Two modes:
      - ``cat_all_levels=False`` (default): route each ROI to a single FPN
        level via LevelMapper then ROI Align.  Matches the official box-head
        ``FPN2MLPFeatureExtractor`` pooler (cat_all_levels=False).
      - ``cat_all_levels=True``: pool every ROI from ALL FPN levels, concat
        along the channel dimension, then reduce via a ``reduce_channel`` conv.
        Matches the official union feature extractor (POOLING_ALL_LEVELS=True).
    """

    def __init__(
        self,
        output_size=7,
        scales=(0.25, 0.125, 0.0625, 0.03125),
        sampling_ratio=2,
        cat_all_levels=False,
        in_channels=256,
    ):
        super().__init__()
        self.output_size = output_size
        self.scales = scales
        self.sampling_ratio = sampling_ratio
        self.cat_all_levels = cat_all_levels

        if cat_all_levels:
            num_scales = len(scales)
            self.reduce_channel = nn.Sequential(
                nn.Conv2d(in_channels * num_scales, in_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        else:
            lvl_min = int(-math.log2(scales[0]))
            lvl_max = int(-math.log2(scales[-1]))
            self.map_levels = LevelMapper(lvl_min, lvl_max)

    def forward(self, fpn_features, boxes_xyxy_list):
        """fpn_features: tuple of [C, H_i, W_i]  (P2, P3, P4, P5, ...).
        boxes_xyxy_list: list of [N_i, 4] absolute (x1,y1,x2,y2)."""
        if self.cat_all_levels:
            return self._forward_cat_all_levels(fpn_features, boxes_xyxy_list)
        return self._forward_single_level(fpn_features, boxes_xyxy_list)

    def _forward_single_level(self, fpn_features, boxes_xyxy_list):
        """Route each ROI to one FPN level based on area."""
        num_levels = len(fpn_features)
        C = fpn_features[0].size(0)
        out_s = self.output_size
        device, dtype = fpn_features[0].device, fpn_features[0].dtype

        rois_list = []
        for i, b in enumerate(boxes_xyxy_list):
            if b.numel() == 0:
                continue
            ids = torch.full((b.size(0), 1), i, dtype=dtype, device=device)
            rois_list.append(torch.cat([ids, b], dim=1))
        if not rois_list:
            return torch.zeros(0, C, out_s, out_s, device=device, dtype=dtype)
        rois = torch.cat(rois_list, dim=0)
        total_N = rois.size(0)

        levels = self.map_levels(boxes_xyxy_list).clamp(0, num_levels - 1)
        result = torch.zeros(total_N, C, out_s, out_s, device=device, dtype=dtype)
        for lvl in range(num_levels):
            mask = levels == lvl
            if not mask.any():
                continue
            rois_lvl = rois[mask]
            feat = fpn_features[lvl].unsqueeze(0)
            pooled = roi_align(
                feat,
                rois_lvl,
                output_size=(out_s, out_s),
                spatial_scale=self.scales[lvl],
                sampling_ratio=self.sampling_ratio,
                aligned=True,
            )
            result[mask] = pooled
        return result

    def _forward_cat_all_levels(self, fpn_features, boxes_xyxy_list):
        """Pool from ALL FPN levels, concat, then reduce_channel.

        Official RelationFeatureExtractor pooler with POOLING_ALL_LEVELS=True.
        Uses only the first ``len(scales)`` FPN levels (P2–P5).
        """
        num_levels = min(len(self.scales), len(fpn_features))
        C = fpn_features[0].size(0)
        out_s = self.output_size
        device, dtype = fpn_features[0].device, fpn_features[0].dtype

        rois_list = []
        for i, b in enumerate(boxes_xyxy_list):
            if b.numel() == 0:
                continue
            ids = torch.full((b.size(0), 1), i, dtype=dtype, device=device)
            rois_list.append(torch.cat([ids, b], dim=1))
        if not rois_list:
            return torch.zeros(0, C, out_s, out_s, device=device, dtype=dtype)
        rois = torch.cat(rois_list, dim=0)
        total_N = rois.size(0)

        level_results = []
        for lvl in range(num_levels):
            feat = fpn_features[lvl].unsqueeze(0)
            pooled = roi_align(
                feat,
                rois,
                output_size=(out_s, out_s),
                spatial_scale=self.scales[lvl],
                sampling_ratio=self.sampling_ratio,
                aligned=True,
            )
            level_results.append(pooled)  # each [total_N, C, out_s, out_s]

        concat = torch.cat(level_results, dim=1)  # [total_N, C*L, out_s, out_s]
        return self.reduce_channel(concat)  # [total_N, C, out_s, out_s]


# =========================================================================
# PENet-specific feature extractors (use FPN + FPNPooler)
# =========================================================================


class PENetBoxFeatureExtractor(nn.Module):
    """Box feature extractor — matches official FPN2MLPFeatureExtractor.

    FPNPooler (single-level) → flatten → fc6(12544→4096) + fc7(4096→4096).
    """

    def __init__(self, roi_output_size=7, representation_size=4096):
        super().__init__()
        self.pooler = FPNPooler(
            output_size=roi_output_size,
            scales=(0.25, 0.125, 0.0625, 0.03125),
            sampling_ratio=2,
        )
        input_size = 256 * roi_output_size * roi_output_size  # 12544
        self.fc6 = nn.Linear(input_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        for m in [self.fc6, self.fc7]:
            nn.init.kaiming_uniform_(m.weight, a=1)
            nn.init.constant_(m.bias, 0)
        self.out_channels = representation_size

    def forward(self, fpn_features, boxes, image_size):
        """boxes: [N,4] (cx,cy,w,h norm), image_size: [2] (H,W)."""
        img_h, img_w = image_size[0].float().item(), image_size[1].float().item()
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        xyxy = torch.stack(
            [
                (cx - w / 2) * img_w,
                (cy - h / 2) * img_h,
                (cx + w / 2) * img_w,
                (cy + h / 2) * img_h,
            ],
            dim=-1,
        )
        pooled = self.pooler(fpn_features, [xyxy])  # [N, 256, 7, 7]
        x = F.relu(self.fc6(pooled.flatten(1)))  # [N, 4096]
        return F.relu(self.fc7(x))


class PENetUnionFeatureExtractor(nn.Module):
    """Union feature extractor — matches official RelationFeatureExtractor.

    Multi-level FPNPooler (cat_all_levels=True) on union boxes
    + rect_conv → element-wise add → fc6+fc7 → 4096.
    """

    def __init__(self, roi_output_size=7, representation_size=4096):
        super().__init__()
        self.roi_output_size = roi_output_size
        self.rect_size = roi_output_size * 4 - 1  # 27
        self.pooler = FPNPooler(
            output_size=roi_output_size,
            scales=(0.25, 0.125, 0.0625, 0.03125),
            sampling_ratio=2,
            cat_all_levels=True,
            in_channels=256,
        )
        self.rect_conv = nn.Sequential(
            nn.Conv2d(2, 128, 7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        )
        input_size = 256 * roi_output_size * roi_output_size  # 12544
        self.fc6 = nn.Linear(input_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        for m in [self.fc6, self.fc7]:
            nn.init.kaiming_uniform_(m.weight, a=1)
            nn.init.constant_(m.bias, 0)
        self.out_channels = representation_size

    def forward(self, fpn_features, boxes, pairs, image_size):
        """boxes: [N,4] (cx,cy,w,h norm), pairs: [P,2], image_size: [2]."""
        device = boxes.device
        P = pairs.size(0)
        img_h, img_w = image_size[0].float().item(), image_size[1].float().item()
        rs = self.rect_size

        # Normalized cxcywh → absolute xyxy
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        xyxy = torch.stack(
            [
                (cx - w / 2) * img_w,
                (cy - h / 2) * img_h,
                (cx + w / 2) * img_w,
                (cy + h / 2) * img_h,
            ],
            dim=-1,
        )
        s_boxes = xyxy[pairs[:, 0]]
        o_boxes = xyxy[pairs[:, 1]]

        # Union boxes + FPN pooler
        union_xyxy = torch.stack(
            [
                torch.min(s_boxes[:, 0], o_boxes[:, 0]),
                torch.min(s_boxes[:, 1], o_boxes[:, 1]),
                torch.max(s_boxes[:, 2], o_boxes[:, 2]),
                torch.max(s_boxes[:, 3], o_boxes[:, 3]),
            ],
            dim=-1,
        )
        union_vis = self.pooler(fpn_features, [union_xyxy])  # [P, 256, 7, 7]

        # Binary rect masks + rect_conv
        xr = torch.arange(rs, device=device).view(1, 1, -1).expand(P, rs, rs)
        yr = torch.arange(rs, device=device).view(1, -1, 1).expand(P, rs, rs)

        def resize_box(b, sz):
            s = float(sz - 1)
            return torch.stack(
                [
                    b[:, 0] / img_w * s,
                    b[:, 1] / img_h * s,
                    b[:, 2] / img_w * s,
                    b[:, 3] / img_h * s,
                ],
                dim=-1,
            )

        s_r, o_r = resize_box(s_boxes, rs), resize_box(o_boxes, rs)

        def make_rect(b):
            return (
                (xr >= b[:, 0].floor().view(-1, 1, 1).long())
                & (xr <= b[:, 2].ceil().view(-1, 1, 1).long())
                & (yr >= b[:, 1].floor().view(-1, 1, 1).long())
                & (yr <= b[:, 3].ceil().view(-1, 1, 1).long())
            ).float()

        rect_input = torch.stack([make_rect(s_r), make_rect(o_r)], dim=1)
        rect_feats = self.rect_conv(rect_input)  # [P, 256, 7, 7]

        combined = union_vis + rect_feats  # element-wise add
        x = F.relu(self.fc6(combined.flatten(1)))  # [P, 4096]
        return F.relu(self.fc7(x))
