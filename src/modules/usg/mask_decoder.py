# src/modules/usg/mask_decoder.py
"""
Shared Mask Decoder for USG-Par (Module 2).

Based on Mask2Former / DETR-style architecture:
  - N learnable object queries
  - L transformer decoder layers with self-attention + cross-attention
  - Outputs: refined object embeddings, bounding box predictions, class logits

Adapted for single-modality (VG-only): uses standard transformer decoder
with cross-attention to backbone feature maps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, List
import math

from utils.misc import NestedTensor


class USGMaskDecoder(nn.Module):
    """Transformer decoder that refines object queries via cross-attention to image features.

    Architecture (similar to DETR decoder):
      - Self-attention among object queries
      - Cross-attention from queries to backbone feature maps
      - FFN with residual connections

    Args:
        d_model: hidden dimension (256 by default, consistent with DETR-based SGG)
        nhead: number of attention heads (8)
        num_decoder_layers: number of decoder layers (6)
        dim_feedforward: FFN hidden dimension (2048)
        dropout: dropout rate (0.1)
        num_queries: number of object queries (200)
        normalize_before: whether to use pre-norm (True)
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_queries: int = 200,
        normalize_before: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=normalize_before,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model) if normalize_before else None,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,            # [B, HW, d_model] — flattened backbone features
        mask: Optional[Tensor], # [B, HW] — padding mask (True = ignore)
        pos_embed: Tensor,      # [B, HW, d_model] — positional embeddings
    ) -> Tensor:
        """
        Args:
            src: flattened feature map from backbone projection [B, HW, d_model]
            mask: padding mask [B, HW], True where padded
            pos_embed: positional encoding [B, HW, d_model]

        Returns:
            hs: [B, num_queries, d_model] — refined object query embeddings
        """
        B = src.shape[0]
        device = src.device

        # Repeat queries for batch
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Nq, d_model]

        # Add positional encoding to source
        src_with_pos = src + pos_embed

        # Decode: self-attn on queries, cross-attn to src
        # memory_key_padding_mask: True = ignore this position
        hs = self.decoder(
            tgt=query,
            memory=src_with_pos,
            memory_key_padding_mask=mask,
        )  # [B, Nq, d_model]

        return hs


class USGObjectHead(nn.Module):
    """Object detection heads: class prediction + bounding box regression.

    Following DETR/RelTR convention:
      - class_embed: Linear(d_model, num_classes + 1) — includes "no object" class
      - bbox_embed: MLP(d_model, d_model, 4, 3) — predicts normalized cxcywh boxes
    """

    def __init__(self, d_model: int = 256, num_classes: int = 151):
        super().__init__()
        # num_classes + 1: extra bg class (matching RelTR convention)
        # Hungarian matcher indexes with raw 1-indexed labels, so we need
        # enough output slots to index label 150 safely.
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

    def forward(self, hs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            hs: [B, Nq, d_model] object query features

        Returns:
            obj_logits: [B, Nq, num_classes] — class logits (NO bg class; sigmoid applied later)
            obj_boxes: [B, Nq, 4] — bounding boxes in normalized cxcywh [0,1]
        """
        obj_logits = self.class_embed(hs)
        obj_boxes = self.bbox_embed(hs).sigmoid()
        return obj_logits, obj_boxes


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_mask_decoder(args) -> USGMaskDecoder:
    """Build USG mask decoder from config args."""
    d_model = getattr(args, 'hidden_dim', 256)
    nhead = getattr(args, 'nheads', 8)
    num_decoder_layers = getattr(args, 'dec_layers', 6)
    dim_feedforward = getattr(args, 'dim_feedforward', 2048)
    dropout = getattr(args, 'dropout', 0.1)
    num_queries = getattr(args, 'num_queries', 200)
    normalize_before = getattr(args, 'pre_norm', True)

    return USGMaskDecoder(
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_queries=num_queries,
        normalize_before=normalize_before,
    )
