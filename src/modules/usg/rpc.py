# src/modules/usg/rpc.py
"""
Relation Proposal Constructor (RPC) for USG-Par (Module 4).

Implements bidirectional relation-aware cross-attention between subject and
object embeddings, producing a confidence matrix over object pairs, followed
by top-K selection of the most promising (subject, object) pairs.

Architecture (from USG-Par §3.4):
  1. Project subject/object queries from object embeddings
  2. Bidirectional cross-attention: sub→obj and obj→sub
  3. Iteratively refine subject/object features over L_rpc layers (3 in paper)
  4. Compute pair confidence matrix via cosine similarity
  5. Top-K selection → relation pairs
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
import math


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional cross-attention between subject and object embeddings.

    For each layer:
      - Subjects attend to object features
      - Objects attend to subject features
      - Results are combined via residual connection + layer norm
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.sub_to_obj_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.obj_to_sub_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.sub_norm = nn.LayerNorm(d_model)
        self.obj_norm = nn.LayerNorm(d_model)

        self.sub_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.obj_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.sub_norm2 = nn.LayerNorm(d_model)
        self.obj_norm2 = nn.LayerNorm(d_model)

    def forward(
        self, sub_feat: Tensor, obj_feat: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            sub_feat: [B, Nq, d_model] subject features
            obj_feat: [B, Nq, d_model] object features

        Returns:
            refined_sub: [B, Nq, d_model]
            refined_obj: [B, Nq, d_model]
        """
        # Sub → Obj cross-attention: subjects query object context
        sub2obj, _ = self.sub_to_obj_attn(sub_feat, obj_feat, obj_feat)
        sub_feat = self.sub_norm(sub_feat + sub2obj)

        # Obj → Sub cross-attention: objects query subject context
        obj2sub, _ = self.obj_to_sub_attn(obj_feat, sub_feat, sub_feat)
        obj_feat = self.obj_norm(obj_feat + obj2sub)

        # FFN for each
        sub_feat = self.sub_norm2(sub_feat + self.sub_ffn(sub_feat))
        obj_feat = self.obj_norm2(obj_feat + self.obj_ffn(obj_feat))

        return sub_feat, obj_feat


class RelationProposalConstructor(nn.Module):
    """RPC: relation-aware cross-attention + top-K pair selection.

    Args:
        d_model: hidden dimension (256)
        nhead: attention heads (8)
        num_rpc_layers: number of RPC refinement layers (3 in paper)
        top_k_pairs: number of pairs to select for relation classification (64)
        dropout: dropout rate (0.1)
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_rpc_layers: int = 3,
        top_k_pairs: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.top_k_pairs = top_k_pairs

        # Sub/Obj projection from object queries
        self.sub_proj = nn.Linear(d_model, d_model // 2)
        self.obj_proj = nn.Linear(d_model, d_model // 2)

        # Bidirectional cross-attention blocks
        self.rpc_layers = nn.ModuleList([
            BidirectionalCrossAttention(d_model // 2, nhead, dropout)
            for _ in range(num_rpc_layers)
        ])

        # Confidence head: project sub+obj features to confidence score
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, obj_feat: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate relation proposals from object features.

        Args:
            obj_feat: [B, Nq, d_model] object query features

        Returns:
            pair_indices: [B, K, 2] — (sub_idx, obj_idx) pairs
            sub_feat: [B, K, d_model/2] — refined subject features for selected pairs
            obj_feat_k: [B, K, d_model/2] — refined object features for selected pairs
            confidence: [B, K] — confidence scores for selected pairs
        """
        B, Nq, D = obj_feat.shape
        device = obj_feat.device

        # Project to sub/obj spaces (half the dimension for efficiency)
        sub_feat = self.sub_proj(obj_feat)  # [B, Nq, D/2]
        obj_feat = self.obj_proj(obj_feat)  # [B, Nq, D/2]

        # Iterative bidirectional refinement
        for layer in self.rpc_layers:
            sub_feat, obj_feat = layer(sub_feat, obj_feat)

        # Compute pair confidence: cos_sim(sub, obj) for all pairs
        # For efficiency, we use a batched dot product
        sub_norm = F.normalize(sub_feat, p=2, dim=-1)  # [B, Nq, D/2]
        obj_norm = F.normalize(obj_feat, p=2, dim=-1)   # [B, Nq, D/2]
        confidence_matrix = torch.bmm(sub_norm, obj_norm.transpose(1, 2))  # [B, Nq, Nq]

        # Also compute a learnable confidence
        # concat sub and obj features for each pair: [B, Nq, Nq, D]
        sub_exp = sub_feat.unsqueeze(2).expand(-1, -1, Nq, -1)  # [B, Nq, Nq, D/2]
        obj_exp = obj_feat.unsqueeze(1).expand(-1, Nq, -1, -1)   # [B, Nq, Nq, D/2]
        pair_feat = torch.cat([sub_exp, obj_exp], dim=-1)          # [B, Nq, Nq, D]
        learned_conf = self.confidence_head(pair_feat).squeeze(-1)  # [B, Nq, Nq]

        # Combine cosine similarity with learned confidence
        pair_scores = confidence_matrix + learned_conf.tanh()  # [B, Nq, Nq]

        # Mask diagonal (no self-relations)
        diag_mask = torch.eye(Nq, device=device, dtype=torch.bool).unsqueeze(0)
        pair_scores = pair_scores.masked_fill(diag_mask, float('-inf'))

        # Top-K selection
        K = min(self.top_k_pairs, Nq * Nq)
        flat_scores = pair_scores.reshape(B, -1)
        topk_scores, topk_idx = torch.topk(flat_scores, k=K, dim=-1)

        sub_idx = (topk_idx // Nq).long()  # [B, K]
        obj_idx = (topk_idx % Nq).long()    # [B, K]

        # Gather selected features
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)
        sub_feat_k = sub_feat[batch_idx, sub_idx]  # [B, K, D/2]
        obj_feat_k = obj_feat[batch_idx, obj_idx]   # [B, K, D/2]

        pair_indices = torch.stack([sub_idx, obj_idx], dim=-1)  # [B, K, 2]

        return pair_indices, sub_feat_k, obj_feat_k, topk_scores


def build_rpc(args):
    """Build RelationProposalConstructor from config."""
    d_model = getattr(args, 'hidden_dim', 256)
    nhead = getattr(args, 'nheads', 8)
    num_rpc_layers = getattr(args, 'rpc_layers', 3)
    top_k_pairs = getattr(args, 'top_k_pairs', 64)
    dropout = getattr(args, 'dropout', 0.1)

    return RelationProposalConstructor(
        d_model=d_model,
        nhead=nhead,
        num_rpc_layers=num_rpc_layers,
        top_k_pairs=top_k_pairs,
        dropout=dropout,
    )
