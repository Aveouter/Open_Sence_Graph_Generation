# src/modules/usg/relation_decoder.py
"""
Relation Decoder for USG-Par (Module 5).

Transformer decoder that takes relation queries (constructed from subject+object
features) and predicts predicate class logits via cross-attention to image features.

Architecture (from USG-Par §3.5):
  1. Construct relation queries: cat(sub_feat, obj_feat) → linear project
  2. L transformer decoder layers with:
     - Self-attention among relation queries
     - Cross-attention to image features
  3. Predicate classification head (MLP)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math


class USGRelationDecoder(nn.Module):
    """Transformer decoder for predicate classification.

    Args:
        d_model: hidden dimension (256)
        nhead: attention heads (8)
        num_decoder_layers: decoder layers (6 in paper)
        dim_feedforward: FFN dimension (2048)
        num_predicates: number of predicate classes (51)
        dropout: dropout rate (0.1)
        normalize_before: pre-norm (True)
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        num_predicates: int = 51,
        dropout: float = 0.1,
        normalize_before: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        # Project concatenated (sub_feat, obj_feat) to relation query space
        # sub_feat and obj_feat from RPC are d_model/2 each → d_model total
        self.query_proj = nn.Linear(d_model, d_model)

        # Transformer decoder
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

        # Predicate classification head
        # num_predicates + 1: extra bg class (matching RelTR convention)
        self.pred_classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_predicates + 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        sub_feat: Tensor,      # [B, K, D/2] — subject features from RPC
        obj_feat: Tensor,       # [B, K, D/2] — object features from RPC
        src: Tensor,            # [B, HW, D] — flattened backbone features
        mask: Optional[Tensor], # [B, HW] — padding mask (True = ignore)
        pos_embed: Tensor,      # [B, HW, D] — positional embeddings
    ) -> Tensor:
        """
        Args:
            sub_feat: subject features from RPC
            obj_feat: object features from RPC
            src: flattened feature map from backbone projection [B, HW, d_model]
            mask: padding mask [B, HW]
            pos_embed: positional encoding [B, HW, d_model]

        Returns:
            pred_logits: [B, K, num_predicates] — predicate class logits
        """
        B, K, D_half = sub_feat.shape

        # Construct relation queries: cat + project
        rel_query = torch.cat([sub_feat, obj_feat], dim=-1)  # [B, K, D]
        rel_query = self.query_proj(rel_query)                 # [B, K, d_model]

        # Self-attn on relation queries + cross-attn to image features
        src_with_pos = src + pos_embed
        rel_feat = self.decoder(
            tgt=rel_query,
            memory=src_with_pos,
            memory_key_padding_mask=mask,
        )  # [B, K, d_model]

        # Predicate classification
        pred_logits = self.pred_classifier(rel_feat)  # [B, K, num_predicates]

        return pred_logits


def build_relation_decoder(args):
    """Build USG relation decoder from config."""
    d_model = getattr(args, 'hidden_dim', 256)
    nhead = getattr(args, 'nheads', 8)
    num_decoder_layers = getattr(args, 'rel_dec_layers', 6)
    dim_feedforward = getattr(args, 'dim_feedforward', 2048)
    num_predicates = getattr(args, 'rel_nums', 51)
    dropout = getattr(args, 'dropout', 0.1)
    normalize_before = getattr(args, 'pre_norm', True)

    return USGRelationDecoder(
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        num_predicates=num_predicates,
        dropout=dropout,
        normalize_before=normalize_before,
    )
