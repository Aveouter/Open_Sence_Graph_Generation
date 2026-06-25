"""
USG-Par: Universal Scene Graph Parser (Wu et al., CVPR 2025).

Image-only adaptation for Visual Genome SGG.
Aligned with official implementation at https://github.com/ChocoWu/USG.

Architecture (§3, matched to official code):
  Step 1: Frozen OpenCLIP ConvNeXt-L + Transformer Pixel Decoder
  Step 2: Shared Mask Decoder (Mask2Former-style, 9 layers, masked cross-attn)
  Step 3: Detection Head (class + box regression, adapted for VG bbox)
  Step 4: RPC (4 layers, two-way cross-attn + self-attn + FFN)
  Step 5: Relation Decoder (6 layers, cross-attn H + self-attn + FFN)

Output format: RelTR-compatible 5-key schema
"""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import linear_sum_assignment

from utils.misc import NestedTensor
from utils import box_ops


# ==============================================================================
# Defaults (matched to official USG configs/psg.yaml)
# ==============================================================================
DEFAULT_MODEL = "convnext_large_d_320"
DEFAULT_PRETRAINED = "laion2b_s29b_b131k_ft_soup"
IMAGE_SIZE = 320


# ==============================================================================
# Step 1a: Frozen OpenCLIP ConvNeXt-L Backbone
# ==============================================================================


class ConvNeXtBackbone(nn.Module):
    """Frozen CLIP-ConvNeXt-L trunk. Returns last 3 stage feature maps.

    Official code: usg_par/encoders/image_encoder.py:ConvNeXtBackbone
    ConvNeXt-L channels: [192, 384, 768, 1536] → keep stages 1,2,3: [384, 768, 1536]
    ConvNeXt-B channels (fallback): [128, 256, 512, 1024] → keep: [256, 512, 1024]
    """

    def __init__(self, clip_model: nn.Module, freeze: bool = True):
        super().__init__()
        self.trunk = clip_model.visual.trunk
        self.out_channels = [f["num_chs"] for f in self.trunk.feature_info][1:]
        if freeze:
            for p in self.trunk.parameters():
                p.requires_grad_(False)
        self._frozen = freeze

    def forward(self, x: Tensor) -> List[Tensor]:
        if self._frozen:
            with torch.no_grad():
                y = self.trunk.stem(x)
                feats = []
                for stage in self.trunk.stages:
                    y = stage(y)
                    feats.append(y)
                return feats[1:]  # drop stride-4
        y = self.trunk.stem(x)
        feats = []
        for stage in self.trunk.stages:
            y = stage(y)
            feats.append(y)
        return feats[1:]


# ==============================================================================
# Step 1b: Transformer Pixel Decoder (official: encoders/image_encoder.py)
# ==============================================================================


def _sine_pos_embed_2d(
    h: int, w: int, dim: int, device, temperature: float = 10000.0
) -> Tensor:
    assert dim % 4 == 0
    d = dim // 2
    y = torch.arange(h, device=device).float()
    x = torch.arange(w, device=device).float()
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    omega = torch.arange(d // 2, device=device).float()
    omega = 1.0 / (temperature ** (omega / (d // 2)))
    out_x = xx.flatten()[:, None] * omega[None, :]
    out_y = yy.flatten()[:, None] * omega[None, :]
    pe = torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=1)
    return pe  # (h*w, dim)


class PixelDecoder(nn.Module):
    """Transformer-based multi-scale pixel decoder.

    Official: usg_par/encoders/image_encoder.py:PixelDecoder
    Project backbone features → flatten + pos embed + level embed →
    TransformerEncoder over all tokens → split back to per-scale maps.
    """

    def __init__(
        self,
        in_channels: List[int],
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        mask_feature_size_scale: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.mask_feature_size_scale = mask_feature_size_scale
        self.input_proj = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in in_channels])
        self.level_embed = nn.Parameter(torch.randn(len(in_channels), dim))
        layer = nn.TransformerEncoderLayer(
            dim, num_heads, dim_feedforward=ffn_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.mask_proj = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, feats: List[Tensor]) -> Dict[str, Tensor]:
        """
        Returns dict:
          feats_per_scale: list of (B, L_i, d) coarse→fine flattened features
          feat_sizes: list of (H_i, W_i)
          mask_features: (B, d, H, W) high-res per-pixel embedding
          context: (B, L_coarsest, d) compact context for relation decoder
        """
        b = feats[0].size(0)
        device = feats[0].device

        projected = [proj(f) for proj, f in zip(self.input_proj, feats)]
        sizes = [(f.shape[-2], f.shape[-1]) for f in projected]

        tokens, splits = [], []
        for lvl, f in enumerate(projected):
            h, w = f.shape[-2:]
            t = f.flatten(2).transpose(1, 2)
            pos = _sine_pos_embed_2d(h, w, self.dim, device)[None]
            t = t + pos + self.level_embed[lvl][None, None]
            tokens.append(t)
            splits.append(h * w)

        fused = self.encoder(torch.cat(tokens, dim=1))

        per_scale_maps = []
        offset = 0
        for (h, w), n in zip(sizes, splits):
            chunk = (
                fused[:, offset : offset + n].transpose(1, 2).reshape(b, self.dim, h, w)
            )
            per_scale_maps.append(chunk)
            offset += n

        # H_3: high-res mask features from finest map
        finest = per_scale_maps[0]
        up = F.interpolate(
            finest,
            scale_factor=self.mask_feature_size_scale,
            mode="bilinear",
            align_corners=False,
        )
        mask_features = self.mask_proj(up)

        # Coarse → fine order for mask decoder round-robin
        order = list(reversed(range(len(per_scale_maps))))
        feats_per_scale = [per_scale_maps[i].flatten(2).transpose(1, 2) for i in order]
        feat_sizes = [sizes[i] for i in order]
        context = feats_per_scale[0]  # coarsest = compact H

        return {
            "feats_per_scale": feats_per_scale,
            "feat_sizes": feat_sizes,
            "mask_features": mask_features,
            "context": context,
        }


# ==============================================================================
# MLP (official: usg_par/layers.py:MLP)
# ==============================================================================


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = None,
        out_dim: int = None,
        num_layers: int = 2,
    ):
        super().__init__()
        h = hidden_dim or in_dim
        o = out_dim or in_dim
        dims = [in_dim] + [h] * (num_layers - 1) + [o]
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


# ==============================================================================
# Step 2: Shared Mask Decoder (official: mask_decoder.py)
# ==============================================================================


def _build_attn_mask(
    mask_logits: Tensor, size: Tuple[int, int], num_heads: int
) -> Tensor:
    """Binarize & resize mask into masked-attention mask. Official: mask_decoder.py."""
    b, n, _, _ = mask_logits.shape
    resized = F.interpolate(
        mask_logits, size=size, mode="bilinear", align_corners=False
    )
    attn = resized.sigmoid() < 0.5
    attn = attn.flatten(2)
    all_masked = attn.all(dim=-1, keepdim=True)
    attn = attn & ~all_masked
    attn = attn.unsqueeze(1).expand(-1, num_heads, -1, -1).reshape(b * num_heads, n, -1)
    return attn.detach()


class _MaskPredictor(nn.Module):
    """MLP(query) → mask_embed; mask_logits = <mask_embed, pixel_embed>."""

    def __init__(self, dim: int, num_layers: int = 3):
        super().__init__()
        self.mask_embed = MLP(dim, dim, dim, num_layers=num_layers)

    def forward(self, query: Tensor, pixel_embed: Tensor) -> Tensor:
        me = self.mask_embed(query)
        return torch.einsum("bnd,bdhw->bnhw", me, pixel_embed)


class _MaskDecoderLayer(nn.Module):
    """One Mask2Former layer: masked cross-attn → self-attn → FFN (post-norm)."""

    def __init__(
        self, dim: int, num_heads: int = 8, ffn_dim: int = 2048, dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.cross = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, query, feat, attn_mask=None, feat_key_padding_mask=None):
        ca, _ = self.cross(
            query,
            feat,
            feat,
            attn_mask=attn_mask,
            key_padding_mask=feat_key_padding_mask,
        )
        query = self.norm1(query + ca)
        sa, _ = self.self_attn(query, query, query)
        query = self.norm2(query + sa)
        query = self.norm3(query + self.ffn(query))
        return query


class SharedMaskDecoder(nn.Module):
    """Mask2Former-style cascaded decoder (shared across modalities).

    Official: mask_decoder.py:SharedMaskDecoder
    """

    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 9,
        num_scales: int = 3,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_scales = num_scales
        self.num_heads = num_heads
        self.layers = nn.ModuleList(
            [
                _MaskDecoderLayer(dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.mask_predictor = _MaskPredictor(dim)
        self.decoder_norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: Tensor,  # (B, N, d)
        feats_per_scale: List[Tensor],  # list of (B, L_i, d) coarse→fine
        feat_sizes: List[Tuple[int, int]],  # list of (H_i, W_i)
        mask_features: Tensor,  # (B, d, H, W)
        feat_key_padding_masks: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        b, n, d = query.shape
        x = query
        mask_logits = self.mask_predictor(x, mask_features)

        for layer_idx in range(self.num_layers):
            s = layer_idx % self.num_scales
            feat = feats_per_scale[s]
            kpm = (
                feat_key_padding_masks[s]
                if feat_key_padding_masks is not None
                else None
            )
            attn_mask = _build_attn_mask(mask_logits, feat_sizes[s], self.num_heads)
            x = self.layers[layer_idx](
                x, feat, attn_mask=attn_mask, feat_key_padding_mask=kpm
            )
            mask_logits = self.mask_predictor(x, mask_features)

        x = self.decoder_norm(x)
        return x, mask_logits


# ==============================================================================
# Step 3: Detection Head (adapted for VG bbox: class + box regression)
# ==============================================================================


class USGDetectionHead(nn.Module):
    """Predict object class logits + bounding boxes from refined queries.

    For VG (fixed 151 classes): linear classifier + MLP bbox regressor.
    Official code uses open-vocab cosine classifier; we adapt to VG's fixed vocab.
    """

    def __init__(self, dim: int = 256, num_classes: int = 151):
        super().__init__()
        self.class_head = nn.Linear(dim, num_classes)
        self.bbox_head = MLP(dim, dim, 4, num_layers=3)

    def forward(self, queries: Tensor) -> Tuple[Tensor, Tensor]:
        return self.class_head(queries), self.bbox_head(queries).sigmoid()


# ==============================================================================
# Step 4: RPC (official: rpc.py)
# ==============================================================================


class _TwoWayRACLayer(nn.Module):
    """Two-way relation-aware cross-attention layer.

    Subject stream: cross-attn(obj) → self-attn → FFN (post-norm)
    Object stream:  cross-attn(sub) → self-attn → FFN (post-norm)
    """

    def __init__(
        self, dim: int, num_heads: int = 8, ffn_dim: int = 2048, dropout: float = 0.0
    ):
        super().__init__()

        def mha():
            return nn.MultiheadAttention(
                dim, num_heads, dropout=dropout, batch_first=True
            )

        def _ffn():
            return nn.Sequential(
                nn.Linear(dim, ffn_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, dim),
            )

        self.sub_cross, self.sub_self, self.sub_ffn = mha(), mha(), _ffn()
        self.sub_n1, self.sub_n2, self.sub_n3 = (
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
        )
        self.obj_cross, self.obj_self, self.obj_ffn = mha(), mha(), _ffn()
        self.obj_n1, self.obj_n2, self.obj_n3 = (
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x_sub: Tensor, x_obj: Tensor) -> Tuple[Tensor, Tensor]:
        s_ca, _ = self.sub_cross(x_sub, x_obj, x_obj)
        x_sub = self.sub_n1(x_sub + s_ca)
        s_sa, _ = self.sub_self(x_sub, x_sub, x_sub)
        x_sub = self.sub_n2(x_sub + s_sa)
        x_sub = self.sub_n3(x_sub + self.sub_ffn(x_sub))

        o_ca, _ = self.obj_cross(x_obj, x_sub, x_sub)
        x_obj = self.obj_n1(x_obj + o_ca)
        o_sa, _ = self.obj_self(x_obj, x_obj, x_obj)
        x_obj = self.obj_n2(x_obj + o_sa)
        x_obj = self.obj_n3(x_obj + self.obj_ffn(x_obj))
        return x_sub, x_obj


def _gather_tokens(x: Tensor, idx: Tensor) -> Tensor:
    """Gather x (B, N, d) at indices idx (B, k) → (B, k, d)."""
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


def _pairwise_cosine(a: Tensor, b: Tensor) -> Tensor:
    """Pairwise cosine similarity: (B, N, d), (B, M, d) → (B, N, M)."""
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    return torch.bmm(a_n, b_n.transpose(1, 2))


class RelationProposalConstructor(nn.Module):
    """Subject/object projectors + two-way RAC + top-k pair selection.

    Official: rpc.py:RelationProposalConstructor
    """

    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        top_k: int = 100,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        self.sub_proj = MLP(dim, dim, dim, num_layers=2)
        self.obj_proj = MLP(dim, dim, dim, num_layers=2)
        self.layers = nn.ModuleList(
            [
                _TwoWayRACLayer(dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, obj_queries: Tensor, top_k: Optional[int] = None
    ) -> Dict[str, Tensor]:
        B, N, _ = obj_queries.shape
        k = top_k if top_k is not None else self.top_k

        e_sub = self.sub_proj(obj_queries)
        e_obj = self.obj_proj(obj_queries)
        x_sub, x_obj = e_sub, e_obj
        for layer in self.layers:
            x_sub, x_obj = layer(x_sub, x_obj)

        c = _pairwise_cosine(x_sub, x_obj)
        k = min(k, N * N)
        scores, flat_idx = c.reshape(B, N * N).topk(k, dim=-1)
        sub_idx = flat_idx // N
        obj_idx = flat_idx % N

        return {
            "pair_confidence": c,
            "q_sub": _gather_tokens(x_sub, sub_idx),
            "q_obj": _gather_tokens(x_obj, obj_idx),
            "e_sub": _gather_tokens(e_sub, sub_idx),
            "e_obj": _gather_tokens(e_obj, obj_idx),
            "sub_idx": sub_idx,
            "obj_idx": obj_idx,
            "scores": scores,
            # RelTR-compat: boxes + logits from detection head (attached later in USGModel)
        }


# ==============================================================================
# Step 5: Relation Decoder (official: relation_decoder.py)
# ==============================================================================


class _RelationDecoderLayer(nn.Module):
    """Cross-attn(H) → self-attn → FFN (post-norm)."""

    def __init__(
        self, dim: int, num_heads: int = 8, ffn_dim: int = 2048, dropout: float = 0.0
    ):
        super().__init__()
        self.cross = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self, x_rel: Tensor, h: Tensor, h_kpm: Optional[Tensor] = None
    ) -> Tensor:
        ca, _ = self.cross(x_rel, h, h, key_padding_mask=h_kpm)
        x_rel = self.norm1(x_rel + ca)
        sa, _ = self.self_attn(x_rel, x_rel, x_rel)
        x_rel = self.norm2(x_rel + sa)
        x_rel = self.norm3(x_rel + self.ffn(x_rel))
        return x_rel


class USGRelationDecoder(nn.Module):
    """Transformer relation decoder + predicate classifier.

    Official: relation_decoder.py:RelationDecoder
    """

    def __init__(
        self,
        dim: int = 256,
        num_predicates: int = 51,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.rel_input_proj = nn.Linear(2 * dim, dim)
        self.layers = nn.ModuleList(
            [
                _RelationDecoderLayer(dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(dim, num_predicates)

    def build_rel_queries(self, q_sub, e_sub, q_obj, e_obj) -> Tensor:
        sub = q_sub + e_sub
        obj = q_obj + e_obj
        return self.rel_input_proj(torch.cat([sub, obj], dim=-1))

    def forward(self, q_sub, e_sub, q_obj, e_obj, h, h_kpm=None) -> Tensor:
        q_rel = self.build_rel_queries(q_sub, e_sub, q_obj, e_obj)
        x = q_rel
        for layer in self.layers:
            x = layer(x, h, h_kpm)
        return self.classifier(x)


# ==============================================================================
# USGModel
# ==============================================================================


class USGModel(nn.Module):
    """USG-Par for image-only VG SGG, aligned with official implementation."""

    def __init__(
        self,
        backbone: nn.Module,
        pixel_decoder: nn.Module,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_classes: int = 151,
        num_predicates: int = 51,
        mask_decoder_layers: int = 9,
        rpc_layers: int = 4,
        relation_layers: int = 6,
        nheads: int = 8,
        ffn_dim: int = 2048,
        top_k: int = 100,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.top_k = top_k

        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))

        self.mask_decoder = SharedMaskDecoder(
            dim=hidden_dim,
            num_layers=mask_decoder_layers,
            num_scales=3,
            num_heads=nheads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.detection_head = USGDetectionHead(hidden_dim, num_classes)
        self.rpc = RelationProposalConstructor(
            dim=hidden_dim,
            num_layers=rpc_layers,
            num_heads=nheads,
            ffn_dim=ffn_dim,
            top_k=top_k,
            dropout=dropout,
        )
        self.relation_decoder = USGRelationDecoder(
            dim=hidden_dim,
            num_predicates=num_predicates,
            num_layers=relation_layers,
            num_heads=nheads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(
        self,
        samples: NestedTensor,
        targets: Optional[List[Dict]] = None,
    ) -> Dict[str, Tensor]:
        B = samples.tensors.shape[0]

        # ---- Step 1: Backbone + Pixel Decoder ----
        backbone_feats = self.backbone(samples.tensors)  # [s8, s16, s32]
        pd_out = self.pixel_decoder(backbone_feats)
        feats_per_scale = pd_out["feats_per_scale"]  # coarse→fine
        feat_sizes = pd_out["feat_sizes"]
        mask_features = pd_out["mask_features"]
        context = pd_out["context"]  # coarsest, for relation decoder

        # ---- Step 2: Shared Mask Decoder ----
        q0 = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        queries, _ = self.mask_decoder(q0, feats_per_scale, feat_sizes, mask_features)

        # ---- Step 3: Detection Head ----
        class_logits, pred_boxes = self.detection_head(queries)

        # ---- Step 4: RPC ----
        if targets is not None and all(
            t.get("rel_annotations") is not None and len(t["rel_annotations"]) > 0
            for t in targets
        ):
            k = max(
                int(max(len(t["rel_annotations"]) for t in targets) * 2), self.top_k
            )
        else:
            k = self.top_k
        rpc_out = self.rpc(queries, top_k=k)

        # Attach detection outputs to RPC pairs for RelTR-compatible eval
        rpc_out["sub_boxes"] = _gather_tokens(pred_boxes, rpc_out["sub_idx"])
        rpc_out["obj_boxes"] = _gather_tokens(pred_boxes, rpc_out["obj_idx"])
        rpc_out["sub_logits"] = _gather_tokens(class_logits, rpc_out["sub_idx"])
        rpc_out["obj_logits"] = _gather_tokens(class_logits, rpc_out["obj_idx"])

        # ---- Step 5: Relation Decoder ----
        rel_logits = self.relation_decoder(
            rpc_out["q_sub"],
            rpc_out["e_sub"],
            rpc_out["q_obj"],
            rpc_out["e_obj"],
            context,
        )

        return {
            "pred_logits": class_logits,
            "pred_boxes": pred_boxes,
            "sub_logits": rpc_out["sub_logits"],
            "obj_logits": rpc_out["obj_logits"],
            "sub_boxes": rpc_out["sub_boxes"],
            "obj_boxes": rpc_out["obj_boxes"],
            "rel_logits": rel_logits,
            "rpc_output": rpc_out,
            "class_logits": class_logits,
            "queries": queries,
        }


# ==============================================================================
# Hungarian Matcher (official: losses.py:HungarianMatcher, adapted for VG bbox)
# ==============================================================================


class HungarianMatcher(nn.Module):
    """Bipartite matching of predicted queries to GT entities, adapted for VG.

    Cost = w_class * (-prob[gt_class]) + w_bbox * L1 + w_giou * (1 - GIoU)
    """

    def __init__(self, w_class: float = 2.0, w_bbox: float = 5.0, w_giou: float = 2.0):
        super().__init__()
        self.w_class = w_class
        self.w_bbox = w_bbox
        self.w_giou = w_giou

    @torch.no_grad()
    def forward(
        self, cls_logits: Tensor, pred_boxes: Tensor, targets: List[Dict]
    ) -> List[Tuple[Tensor, Tensor]]:
        B, N = cls_logits.shape[:2]
        device = cls_logits.device
        indices = []

        for b in range(B):
            gt_labels = targets[b]["labels"].to(device)
            gt_boxes = targets[b]["boxes"].to(device)
            M = len(gt_labels)
            if M == 0:
                indices.append(
                    (
                        torch.empty(0, dtype=torch.long, device=device),
                        torch.empty(0, dtype=torch.long, device=device),
                    )
                )
                continue

            prob = cls_logits[b].softmax(-1)
            cost_class = -prob[:, gt_labels]  # (N, M)
            cost = self.w_class * cost_class

            pred_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[b])
            gt_xyxy = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            cost_l1 = torch.cdist(pred_boxes[b], gt_boxes, p=1)
            cost_giou = 1 - box_ops.generalized_box_iou(pred_xyxy, gt_xyxy)

            cost = cost + self.w_bbox * cost_l1 + self.w_giou * cost_giou

            pred_idx, gt_idx = linear_sum_assignment(cost.cpu().numpy())
            indices.append(
                (
                    torch.as_tensor(pred_idx, dtype=torch.long, device=device),
                    torch.as_tensor(gt_idx, dtype=torch.long, device=device),
                )
            )
        return indices


# ==============================================================================
# USGCriterion (aligned with official losses.py, adapted for VG bbox)
# ==============================================================================


class USGCriterion(nn.Module):
    """L = α·L_cls + β·L_l1 + γ·L_giou + δ·L_rel + ε·L_pair"""

    def __init__(
        self,
        num_classes: int = 151,
        num_predicates: int = 51,
        num_queries: int = 100,
        class_loss_coef: float = 2.0,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        rel_loss_coef: float = 0.8,
        pair_loss_coef: float = 0.5,
        eos_coef: float = 0.1,
        matcher_w_class: float = 2.0,
        matcher_w_bbox: float = 5.0,
        matcher_w_giou: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.c_coef = class_loss_coef
        self.b_coef = bbox_loss_coef
        self.g_coef = giou_loss_coef
        self.r_coef = rel_loss_coef
        self.p_coef = pair_loss_coef
        ew = torch.ones(num_classes)
        ew[0] = eos_coef
        self.register_buffer("empty_weight", ew)
        self.matcher = HungarianMatcher(matcher_w_class, matcher_w_bbox, matcher_w_giou)
        self.rel_ce = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

    @staticmethod
    def _perm_idx(indices):
        return (
            torch.cat([torch.full_like(s, i) for i, (s, _) in enumerate(indices)]),
            torch.cat([s for (s, _) in indices]),
        )

    def _loss_cls(self, cl, targets, indices):
        bi, si = self._perm_idx(indices)
        tc = torch.zeros(cl.shape[:2], dtype=torch.long, device=cl.device)
        tco = torch.cat(
            [t["labels"][J.cpu()].to(cl.device) for t, (_, J) in zip(targets, indices)]
        )
        tc[bi, si] = tco
        return F.cross_entropy(
            cl.flatten(0, 1), tc.flatten(0), weight=self.empty_weight, ignore_index=0
        )

    def _loss_box(self, pb, targets, indices):
        bi, si = self._perm_idx(indices)
        sb = pb[bi, si]
        tb = torch.cat(
            [t["boxes"][i.cpu()].to(pb.device) for t, (_, i) in zip(targets, indices)]
        )
        l1 = F.l1_loss(sb, tb, reduction="sum")
        giou = (
            1
            - box_ops.generalized_box_iou_1to1(
                box_ops.box_cxcywh_to_xyxy(sb), box_ops.box_cxcywh_to_xyxy(tb)
            )
        ).sum()
        return l1, giou

    def _loss_rel(self, rl, rpc, targets, indices):
        """Predicate classification loss.

        Uses Hungarian matching indices to translate RPC query-pair indices
        into GT object indices, then checks against GT relation annotations.
        """
        _, dev, K = len(targets), rl.device, rl.shape[1]
        total, n_pairs = torch.tensor(0.0, device=dev), 0
        sub_idx = rpc["sub_idx"]  # (B, K): query indices [0..N-1]
        obj_idx = rpc["obj_idx"]

        for b, t in enumerate(targets):
            rels = t.get("rel_annotations")
            if rels is None or len(rels) == 0:
                continue

            # Build query→GT mapping from Hungarian matching
            pred_idx, gt_idx = indices[b]
            # q2g must cover [0..num_queries-1] (100); Hungarian only matches
            # a subset, but RPC pair indices can reference ANY query
            q2g = torch.full((self.num_queries,), -1, dtype=torch.long, device=dev)
            q2g[pred_idx] = gt_idx.to(dev)

            # GT relation lookup: (gt_s, gt_o) → predicate
            gp = {(int(r[0]), int(r[1])): int(r[2]) for r in rels}

            gt = torch.full((K,), -1, dtype=torch.long, device=dev)
            for k_idx in range(K):
                qs, qo = int(sub_idx[b, k_idx].item()), int(obj_idx[b, k_idx].item())
                gs, go = q2g[qs].item(), q2g[qo].item()
                if gs >= 0 and go >= 0 and (gs, go) in gp:
                    gt[k_idx] = gp[(gs, go)]

            v = gt >= 0
            if v.any():
                total += self.rel_ce(rl[b][v], gt[v])
                n_pairs += v.sum().item()

        return total / max(n_pairs, 1)

    def _loss_pair(self, rpc, targets, indices):
        """Weighted BCE on pair confidence matrix.

        The pair confidence matrix C (B, N, N) is indexed by query positions.
        GT annotations use object indices (0..M-1).  We use the Hungarian
        matching to build a per-sample object-index → query-index mapping,
        then set C[b, q_s, q_o] = 1 for each (s, o) relation whose subject
        and object are both matched to a query.
        """
        B, dev = len(targets), rpc["pair_confidence"].device
        c = rpc["pair_confidence"]
        _, N, _ = c.shape
        tm = torch.zeros(B, N, N, device=dev)
        for b, t in enumerate(targets):
            rels = t.get("rel_annotations")
            if rels is None or len(rels) == 0:
                continue
            # Hungarian: pred_idx → gt_idx (query index → object index)
            pred_idx, gt_idx = indices[b]
            if len(pred_idx) == 0:
                continue
            # invert: object index → first matched query index
            # (a GT object may be matched to at most one query by Hungarian)
            o2q = torch.full((len(t["labels"]),), -1, dtype=torch.long, device=dev)
            o2q[gt_idx.to(dev)] = pred_idx.to(dev)
            for r in rels:
                s, o = int(r[0]), int(r[1])
                if s < len(o2q) and o < len(o2q):
                    qs, qo = o2q[s].item(), o2q[o].item()
                    if qs >= 0 and qo >= 0:
                        tm[b, qs, qo] = 1.0
        pos = tm.sum().clamp_min(1.0)
        neg = tm.numel() - pos
        pw = max(neg / pos, 1.0)
        return F.binary_cross_entropy_with_logits(
            c, tm, pos_weight=c.new_tensor([pw]).float(), reduction="mean"
        )

    def forward(self, outputs, targets):
        cl = outputs.get("class_logits", outputs["pred_logits"])
        pb, rl, rpc = (
            outputs["pred_boxes"],
            outputs["rel_logits"],
            outputs["rpc_output"],
        )
        indices = self.matcher(cl, pb, targets)
        nb = max(sum(len(t["labels"]) for t in targets), 1)
        l_cls = self._loss_cls(cl, targets, indices) / nb
        l_l1, l_giou = self._loss_box(pb, targets, indices)
        l_l1, l_giou = l_l1 / nb, l_giou / nb
        l_rel = self._loss_rel(rl, rpc, targets, indices)
        l_pair = self._loss_pair(rpc, targets, indices)
        return {
            "loss_total": self.c_coef * l_cls
            + self.b_coef * l_l1
            + self.g_coef * l_giou
            + self.r_coef * l_rel
            + self.p_coef * l_pair,
            "loss_obj_cls": l_cls,
            "loss_l1": l_l1,
            "loss_giou": l_giou,
            "loss_rel": l_rel,
            "loss_pair": l_pair,
        }


# ==============================================================================
# Build function
# ==============================================================================


def build_usg(args):
    num_classes = getattr(args, "entity_nums", 151)
    num_predicates = getattr(args, "rel_nums", 51)
    hidden_dim = getattr(args, "hidden_dim", 256)
    num_queries = getattr(args, "num_queries", 100)
    mask_decoder_layers = getattr(args, "mask_decoder_layers", 9)
    rpc_layers = getattr(args, "rpc_layers", 4)
    relation_layers = getattr(args, "relation_layers", 6)
    nheads = getattr(args, "nheads", 8)
    ffn_dim = getattr(args, "ffn_dim", 2048)
    pixel_ffn_dim = getattr(args, "pixel_ffn_dim", 1024)
    top_k = getattr(args, "top_k", 100)
    dropout = getattr(args, "dropout", 0.0)

    clip_model_name = getattr(args, "clip_model", None) or DEFAULT_MODEL
    clip_pretrained = getattr(args, "clip_pretrained", DEFAULT_PRETRAINED)
    freeze_backbone = getattr(args, "backbone_frozen", True)

    # Build OpenCLIP model
    try:
        import open_clip

        clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        print(f"[USG] Loaded OpenCLIP {clip_model_name} ({clip_pretrained})")
    except Exception:
        print(
            f"[USG] OpenCLIP {clip_model_name} not available, trying convnext_base fallback"
        )
        try:
            import open_clip

            clip_model, _, _ = open_clip.create_model_and_transforms(
                "convnext_base", pretrained="laion400m_s13b_b51k"
            )
            print("[USG] Loaded OpenCLIP convnext_base (fallback)")
        except Exception:
            print("[USG] OpenCLIP unavailable, using ResNet-50 fallback")
            clip_model = None

    # Build backbone
    if clip_model is not None:
        backbone = ConvNeXtBackbone(clip_model, freeze=freeze_backbone)
    else:
        backbone = _ResNetBackbone(freeze=freeze_backbone)

    # Build pixel decoder
    pixel_decoder = PixelDecoder(
        in_channels=backbone.out_channels,
        dim=hidden_dim,
        num_layers=4,
        num_heads=nheads,
        ffn_dim=pixel_ffn_dim,
    )

    model = USGModel(
        backbone=backbone,
        pixel_decoder=pixel_decoder,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_classes=num_classes,
        num_predicates=num_predicates,
        mask_decoder_layers=mask_decoder_layers,
        rpc_layers=rpc_layers,
        relation_layers=relation_layers,
        nheads=nheads,
        ffn_dim=ffn_dim,
        top_k=top_k,
        dropout=dropout,
    )

    criterion = USGCriterion(
        num_classes=num_classes,
        num_predicates=num_predicates,
        num_queries=num_queries,
        class_loss_coef=getattr(args, "class_loss_coef", 2.0),
        bbox_loss_coef=getattr(args, "bbox_loss_coef", 5.0),
        giou_loss_coef=getattr(args, "giou_loss_coef", 2.0),
        rel_loss_coef=getattr(args, "rel_loss_coef", 0.8),
        pair_loss_coef=getattr(args, "pair_loss_coef", 0.5),
        eos_coef=getattr(args, "eos_coef", 0.1),
    )

    return model, criterion


class _ResNetBackbone(nn.Module):
    """ResNet-50 fallback when OpenCLIP unavailable. Outputs 3 stage features."""

    def __init__(self, freeze: bool = True):
        super().__init__()
        import torchvision

        rn = torchvision.models.resnet50(weights="DEFAULT")
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4
        self.out_channels = [512, 1024, 2048]
        self._frozen = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x: Tensor) -> List[Tensor]:
        if self._frozen:
            with torch.no_grad():
                x = self.stem(x)
                x = self.layer1(x)
                f2 = self.layer2(x)
                f3 = self.layer3(f2)
                f4 = self.layer4(f3)
                return [f2, f3, f4]
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f2, f3, f4]
