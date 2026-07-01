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

import json
import math
import os
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
        with torch.set_grad_enabled(not self._frozen):
            y = self.trunk.stem(x)
            feats = []
            for stage in self.trunk.stages:
                y = stage(y)
                feats.append(y)
            return feats[1:]  # drop stride-4


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
# Step 3: Detection Head (official cosine classifier + VG bbox regression)
# ==============================================================================


class USGDetectionHead(nn.Module):
    """Predict object class logits + bounding boxes from refined queries.

    Official USG uses CLIP-style cosine classification against class-name text
    embeddings and appends a learnable "no object" embedding as the last class.
    Visual Genome does not provide masks in this OpenSGG path, so we keep a bbox
    regressor as the dataset adaptation replacing official mask supervision.
    """

    def __init__(
        self,
        dim: int = 256,
        num_classes: int = 151,
        mask_embed_layers: int = 3,
        logit_scale_init: float = math.log(1 / 0.07),
    ):
        super().__init__()
        self.dim = dim
        self.num_object_classes = max(num_classes - 1, 1)
        self.mask_embed = MLP(dim, dim, dim, num_layers=mask_embed_layers)
        self.query_proj = nn.Linear(dim, dim)
        self.no_object_embed = nn.Parameter(torch.randn(dim))
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init))
        self.fallback_class_embed = nn.Parameter(
            torch.randn(self.num_object_classes, dim) * 0.02
        )
        self.bbox_head = MLP(dim, dim, 4, num_layers=3)

    def classify(
        self,
        queries: Tensor,
        class_text_embeddings: Optional[Tensor] = None,
    ) -> Tensor:
        q = F.normalize(self.query_proj(queries), dim=-1)
        if class_text_embeddings is None or class_text_embeddings.numel() == 0:
            class_emb = self.fallback_class_embed
        else:
            class_emb = class_text_embeddings.to(device=q.device, dtype=q.dtype)
            if class_emb.shape[0] != self.num_object_classes:
                raise ValueError(
                    "USG class_text_embeddings must contain exactly "
                    f"{self.num_object_classes} object classes, got {class_emb.shape[0]}"
                )
        emb = torch.cat([class_emb, self.no_object_embed[None]], dim=0)
        emb = F.normalize(emb, dim=-1)
        return self.logit_scale.exp() * (q @ emb.t())

    def predict_masks(self, queries: Tensor, pixel_embed: Tensor) -> Tensor:
        mask_embed = self.mask_embed(queries)
        return torch.einsum("bnd,bdhw->bnhw", mask_embed, pixel_embed).sigmoid()

    def forward(
        self,
        queries: Tensor,
        class_text_embeddings: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        return self.classify(queries, class_text_embeddings), self.bbox_head(queries).sigmoid()


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
        o_ca, _ = self.obj_cross(x_obj, x_sub, x_sub)
        x_sub = self.sub_n1(x_sub + s_ca)
        x_obj = self.obj_n1(x_obj + o_ca)

        s_sa, _ = self.sub_self(x_sub, x_sub, x_sub)
        o_sa, _ = self.obj_self(x_obj, x_obj, x_obj)
        x_sub = self.sub_n2(x_sub + s_sa)
        x_obj = self.obj_n2(x_obj + o_sa)

        x_sub = self.sub_n3(x_sub + self.sub_ffn(x_sub))
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
        class_text_embeddings: Optional[Tensor] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.top_k = top_k
        if class_text_embeddings is None:
            self.register_buffer(
                "class_text_embeddings", torch.empty(0, hidden_dim), persistent=False
            )
            self.class_text_proj = nn.Identity()
        else:
            self.register_buffer(
                "class_text_embeddings", class_text_embeddings.float(), persistent=True
            )
            text_dim = int(class_text_embeddings.shape[-1])
            self.class_text_proj = (
                nn.Identity()
                if text_dim == hidden_dim
                else nn.Linear(text_dim, hidden_dim)
            )

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
        queries, mask_logits = self.mask_decoder(q0, feats_per_scale, feat_sizes, mask_features)
        # VG: mask_logits not used (no mask supervision needed).
        # PSG: retain mask_logits for panoptic mask loss.

        # ---- Step 3: Detection Head ----
        class_emb = (
            self.class_text_proj(self.class_text_embeddings)
            if self.class_text_embeddings.numel() > 0
            else None
        )
        class_logits, pred_boxes = self.detection_head(queries, class_emb)

        # ---- Step 4: RPC ----
        # During training, inflate K from GT so the matcher has headroom.
        # During eval, use fixed K so output shapes are consistent across batches
        # (required by _aggregate_step_outputs → torch.cat).
        if self.training and targets is not None and all(
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
            "model_family": "usg",
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


def _object_labels_to_usg_internal(labels: Tensor, num_classes: int) -> Tensor:
    """Map OpenSGG VG labels (1..150) to USG logits (0..149, no-object last)."""
    labels = labels.long()
    object_classes = num_classes - 1
    if labels.numel() == 0:
        return labels
    if labels.min().item() >= 1 and labels.max().item() <= object_classes:
        return labels - 1
    return labels.clamp(0, object_classes - 1)


def _predicate_label_to_internal(label: int, num_predicates: int) -> Optional[int]:
    """Map OpenSGG VG predicates (1..50) to official USG foreground ids (0..49)."""
    if 1 <= label <= num_predicates:
        return label - 1
    if 0 <= label < num_predicates:
        return label
    return None


def _auto_pos_weight(target: Tensor, fixed: Optional[float] = None) -> Tensor:
    if fixed is not None:
        return target.new_tensor(float(fixed))
    pos = target.sum().clamp_min(1.0)
    neg = target.numel() - target.sum()
    return (neg / pos).clamp_min(1.0)


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

            gt_labels = _object_labels_to_usg_internal(gt_labels, cls_logits.shape[-1])
            prob = cls_logits[b].sigmoid()
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
    """Official USG loss assembly with a VG bbox term.

    Official single-modality USG uses
      L = alpha * L_obj + gamma * (L_predicate_bce + L_pair_bce).
    The PSG mask CE/Dice terms are replaced here with VG L1/GIoU box losses,
    while object and relation classification keep the official sigmoid BCE
    semantics.
    """

    def __init__(
        self,
        num_classes: int = 151,
        num_predicates: int = 50,
        num_queries: int = 100,
        class_loss_coef: float = 1.0,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        rel_loss_coef: float = 0.8,
        pair_loss_coef: float = 1.0,
        eos_coef: float = 0.1,
        matcher_w_class: float = 2.0,
        matcher_w_bbox: float = 5.0,
        matcher_w_giou: float = 2.0,
        matched_class_weight: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.num_queries = num_queries
        self.c_coef = class_loss_coef
        self.b_coef = bbox_loss_coef
        self.g_coef = giou_loss_coef
        self.r_coef = rel_loss_coef
        self.p_coef = pair_loss_coef
        self.no_object_weight = eos_coef
        self.matched_class_weight = matched_class_weight
        self.matcher = HungarianMatcher(matcher_w_class, matcher_w_bbox, matcher_w_giou)

    def _loss_cls(self, cl: Tensor, targets: List[Dict], indices) -> Tensor:
        B, N, C1 = cl.shape
        no_obj = C1 - 1
        total = cl.new_zeros(())
        for b in range(B):
            pred_idx, gt_idx = indices[b]
            target = torch.full((N,), no_obj, dtype=torch.long, device=cl.device)
            q_weight = cl.new_full((N,), float(self.no_object_weight))
            if pred_idx.numel() > 0:
                gt_labels = targets[b]["labels"][gt_idx.cpu()].to(cl.device)
                gt_labels = _object_labels_to_usg_internal(gt_labels, C1)
                target[pred_idx] = gt_labels
                q_weight[pred_idx] = float(self.matched_class_weight)
            target_oh = F.one_hot(target, C1).float()
            per_query = F.binary_cross_entropy_with_logits(
                cl[b], target_oh, reduction="none"
            ).mean(-1)
            total = total + (per_query * q_weight).sum() / q_weight.sum().clamp_min(1.0)
        return total / max(B, 1)

    def _loss_box(self, pb: Tensor, targets: List[Dict], indices) -> Tuple[Tensor, Tensor]:
        src_boxes, target_boxes = [], []
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue
            src_boxes.append(pb[b, pred_idx])
            target_boxes.append(targets[b]["boxes"][gt_idx.cpu()].to(pb.device))
        if not src_boxes:
            zero = pb.new_zeros(())
            return zero, zero
        sb = torch.cat(src_boxes, dim=0)
        tb = torch.cat(target_boxes, dim=0)
        l1 = F.l1_loss(sb, tb, reduction="sum")
        giou = (
            1
            - box_ops.generalized_box_iou_1to1(
                box_ops.box_cxcywh_to_xyxy(sb), box_ops.box_cxcywh_to_xyxy(tb)
            )
        ).sum()
        return l1, giou

    def _build_relation_targets(self, rl: Tensor, rpc: Dict[str, Tensor], targets, indices):
        B, K, P = rl.shape
        _, N, _ = rpc["pair_confidence"].shape
        device = rl.device
        pair_gt = rl.new_zeros((B, N, N))
        predicate_target = rl.new_zeros((B, K, P))
        sub_idx = rpc["sub_idx"]
        obj_idx = rpc["obj_idx"]

        for b, target in enumerate(targets):
            rels = target.get("rel_annotations")
            if rels is None or len(rels) == 0:
                continue

            pred_idx, gt_idx = indices[b]
            q2g = torch.full((N,), -1, dtype=torch.long, device=device)
            q2g[pred_idx.to(device)] = gt_idx.to(device)
            gt2query = {
                int(g): int(q)
                for q, g in enumerate(q2g.detach().cpu().tolist())
                if g >= 0
            }

            pair_preds: Dict[Tuple[int, int], set] = {}
            rel_rows = rels.detach().cpu().tolist() if torch.is_tensor(rels) else rels
            for s, o, p in rel_rows:
                pred_label = _predicate_label_to_internal(int(p), P)
                if pred_label is None:
                    continue
                qs = gt2query.get(int(s))
                qo = gt2query.get(int(o))
                if qs is None or qo is None:
                    continue
                pair_gt[b, qs, qo] = 1.0
                pair_preds.setdefault((qs, qo), set()).add(pred_label)

            for pair_pos in range(K):
                preds = pair_preds.get(
                    (
                        int(sub_idx[b, pair_pos].item()),
                        int(obj_idx[b, pair_pos].item()),
                    )
                )
                if preds:
                    for pred_label in preds:
                        predicate_target[b, pair_pos, pred_label] = 1.0

        return pair_gt, predicate_target

    def _loss_relation_and_pair(self, rl: Tensor, rpc: Dict[str, Tensor], targets, indices):
        pair_gt, predicate_target = self._build_relation_targets(rl, rpc, targets, indices)
        l_rcls = F.binary_cross_entropy_with_logits(rl, predicate_target.float())
        l_pair = F.binary_cross_entropy_with_logits(
            rpc["pair_confidence"],
            pair_gt.float(),
            pos_weight=_auto_pos_weight(pair_gt),
        )
        return l_rcls, l_pair

    def forward(self, outputs, targets):
        cl = outputs.get("class_logits", outputs["pred_logits"])
        pb = outputs["pred_boxes"]
        rl = outputs["rel_logits"]
        rpc = outputs["rpc_output"]

        indices = self.matcher(cl, pb, targets)
        nb = max(sum(len(t["labels"]) for t in targets), 1)

        l_cls = self._loss_cls(cl, targets, indices)
        l_l1, l_giou = self._loss_box(pb, targets, indices)
        l_l1, l_giou = l_l1 / nb, l_giou / nb
        l_rcls, l_pair = self._loss_relation_and_pair(rl, rpc, targets, indices)
        l_rel = l_rcls + self.p_coef * l_pair

        return {
            "loss_total": self.c_coef * l_cls
            + self.b_coef * l_l1
            + self.g_coef * l_giou
            + self.r_coef * l_rel,
            "loss_obj_cls": l_cls,
            "loss_l1": l_l1,
            "loss_giou": l_giou,
            "loss_rel": l_rel,
            "loss_rel_cls": l_rcls,
            "loss_pair": l_pair,
        }


# ==============================================================================
# Fallback backbone (ResNet-50 when OpenCLIP unavailable)
# ==============================================================================


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
        with torch.set_grad_enabled(not self._frozen):
            x = self.stem(x)
            x = self.layer1(x)
            f2 = self.layer2(x)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            return [f2, f3, f4]


# ==============================================================================
# Build function
# ==============================================================================


def _candidate_vg_roots(args) -> List[str]:
    data_root = getattr(args, "data_root", "./data")
    dataname = getattr(args, "dataname", "VisualGenome")
    roots = [
        os.path.join(data_root, dataname),
        os.path.join(data_root, "VisualGenome"),
        data_root,
    ]
    seen, out = set(), []
    for root in roots:
        norm = os.path.normpath(root)
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _load_vg_object_class_names(args, num_object_classes: int) -> Optional[List[str]]:
    explicit = getattr(args, "usg_class_names_path", None)
    paths = [explicit] if explicit else []
    paths.extend(os.path.join(root, "train.json") for root in _candidate_vg_roots(args))
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        categories = ann.get("categories", [])
        by_id = {int(c["id"]): c["name"] for c in categories if "id" in c and "name" in c}
        names = [by_id.get(i) for i in range(1, num_object_classes + 1)]
        if all(names):
            print(f"[USG] Loaded {len(names)} VG object names from {path}")
            return names
    print("[USG] VG object names not found; detection head will learn class embeddings.")
    return None


@torch.no_grad()
def _build_class_text_embeddings(
    args,
    clip_model: Optional[nn.Module],
    clip_model_name: str,
    num_object_classes: int,
) -> Optional[Tensor]:
    """Build official-style frozen class-name embeddings with OpenCLIP."""
    if clip_model is None or not hasattr(clip_model, "encode_text"):
        return None
    names = _load_vg_object_class_names(args, num_object_classes)
    if not names:
        return None
    try:
        import open_clip

        tokenizer = open_clip.get_tokenizer(clip_model_name)
        device = next(clip_model.parameters()).device
        clip_model.eval()
        token_ids = tokenizer(names).to(device)
        emb = clip_model.encode_text(token_ids).float().detach().cpu()
        print(f"[USG] Built OpenCLIP class text embeddings: {tuple(emb.shape)}")
        return emb
    except Exception as exc:
        print(f"[USG] Failed to build class text embeddings: {exc}")
        return None


def build_usg(args):
    num_classes = getattr(args, "entity_nums", 151)
    rel_nums = getattr(args, "rel_nums", 51)
    num_predicates = getattr(
        args,
        "usg_num_predicates",
        rel_nums - 1 if rel_nums is not None and rel_nums > 50 else rel_nums,
    )
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

    class_text_embeddings = _build_class_text_embeddings(
        args,
        clip_model,
        clip_model_name,
        max(num_classes - 1, 1),
    )

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
        class_text_embeddings=class_text_embeddings,
    )

    criterion = USGCriterion(
        num_classes=num_classes,
        num_predicates=num_predicates,
        num_queries=num_queries,
        class_loss_coef=getattr(args, "class_loss_coef", 1.0),
        bbox_loss_coef=getattr(args, "bbox_loss_coef", 5.0),
        giou_loss_coef=getattr(args, "giou_loss_coef", 2.0),
        rel_loss_coef=getattr(args, "rel_loss_coef", 0.8),
        pair_loss_coef=getattr(args, "pair_loss_coef", 1.0),
        eos_coef=getattr(args, "eos_coef", 0.1),
    )

    return model, criterion
