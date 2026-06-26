# src/modules/flowsg/graph_transformer.py
"""
DiT-style Graph Transformer for FlowSG scene graph denoising.

Implements the denoiser architecture from FlowSG (CVPR 2026, §4.3):
  - AdaLN: Adaptive Layer Normalization modulated by time embedding
  - ReSA: Relation-modulated Self-Attention with FiLM gating
  - FMA: Flow-conditioned Message Aggregation with degree-aware moments
  - Global cross-attention to frozen CLIP image features

Paper: Hu, Qin, Yin, Li, Li, He — FlowSG (CVPR 2026)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict


# ==============================================================================
# Sinusoidal Time Embedding
# ==============================================================================


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding (same as diffusion/DiT)."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """t: [B] ∈ [0,1] → [B, dim]"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ==============================================================================
# Adaptive Layer Normalization (AdaLN) — DiT-style [44]
# ==============================================================================


class AdaLN(nn.Module):
    """Time-conditioned adaptive layer normalization.

    t_emb → Linear → (shift, scale, gate) for each modulated sub-layer.
    """

    def __init__(self, dim: int, time_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 3),
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        shift, scale, gate = self.modulation(t_emb).chunk(3, dim=-1)
        # t_emb is [B, 1, time_dim] or [B, N, time_dim]
        x = self.norm(x) * (1 + scale) + shift
        return x * gate.sigmoid()


# ==============================================================================
# Relation-modulated Self-Attention (ReSA) — §4.3 Eq.(16)
# ==============================================================================


class RelationModulatedSelfAttention(nn.Module):
    """Self-attention with FiLM-based predicate injection.

    α_{ij}(t) = softmax_j( q_i·k_j/√d + FiLM(e_{ij}) )

    The edge embedding e_{ij} modulates attention scores via a learned
    FiLM layer, selectively amplifying relation-consistent neighbors.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # FiLM: edge features → per-head attention bias
        self.film = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, num_heads),
        )

    def forward(
        self,
        node_feat: Tensor,  # [B, N, dim]
        edge_feat: Tensor,  # [B, N, N, dim] — dense edge features
        attn_mask: Optional[Tensor] = None,  # [B, N, N] bool
    ) -> Tensor:
        B, N, D = node_feat.shape
        H = self.num_heads
        Hd = self.head_dim

        q = (
            self.q_proj(node_feat).view(B, N, H, Hd).permute(0, 2, 1, 3)
        )  # [B, H, N, Hd]
        k = self.k_proj(node_feat).view(B, N, H, Hd).permute(0, 2, 1, 3)
        v = self.v_proj(node_feat).view(B, N, H, Hd).permute(0, 2, 1, 3)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # FiLM bias from edge features: [B, N, N, dim] → [B, N, N, H]
        film_bias = self.film(edge_feat)  # [B, N, N, H]
        film_bias = film_bias.permute(0, 3, 1, 2)  # [B, H, N, N]
        attn = attn + film_bias

        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, N, Hd]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        return self.out_proj(out)


# ==============================================================================
# Flow-conditioned Message Aggregation (FMA) — §4.3
# ==============================================================================


class FlowConditionedMessageAggregation(nn.Module):
    """Time/degree-aware neighborhood message aggregation.

    Computes neighborhood moments (mean, variance, skewness) and applies
    learned degree-aware scalers, conditioned on time and local context.

    M_i(t) = Σ_k softmax(W_β·ζ_i(t))_k · Ψ_k({v_j}_{j∈N(i)})

    where ζ_i(t) = [φ(t), log(1+deg(i,t)), r̄_i(t)]
    """

    def __init__(self, dim: int = 512, time_dim: int = 256, num_moments: int = 3):
        super().__init__()
        self.dim = dim
        self.num_moments = num_moments

        # Context embedding: [time + degree + edge_context] → dim
        self.context_proj = nn.Sequential(
            nn.Linear(time_dim + 2, dim),
            nn.SiLU(),
        )

        # Attention over moments
        self.moment_attn = nn.Linear(dim, num_moments)

        # Node update projection (takes weighted moments: [B,N,dim])
        self.update_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        node_feat: Tensor,  # [B, N, dim]
        edge_feat: Tensor,  # [B, N, N, dim] — dense edge features
        t_emb: Tensor,  # [B, time_dim]
    ) -> Tensor:
        B, N, D = node_feat.shape

        # Build degree information (from non-zero edges)
        # Edge "existence" is approximated by edge feature norm
        edge_norm = edge_feat.norm(dim=-1)  # [B, N, N]
        deg = (edge_norm > 0.1).float().sum(dim=-1)  # [B, N]

        # Average relation context per node
        r_bar = edge_feat.mean(dim=2)  # [B, N, dim]
        r_bar_norm = r_bar.norm(dim=-1, keepdim=True)  # [B, N, 1]

        # Context vector ζ_i(t) = [φ(t), log(1+deg), ||r̄||]
        log_deg = torch.log1p(deg).unsqueeze(-1)  # [B, N, 1]
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, time_dim]
        zeta = torch.cat(
            [t_emb_expanded, log_deg, r_bar_norm], dim=-1
        )  # [B, N, time_dim+2]

        context = self.context_proj(zeta)  # [B, N, dim]
        moment_weights = F.softmax(self.moment_attn(context), dim=-1)  # [B, N, M]

        # Compute neighborhood moments
        # Ψ₁ = mean (broadcast from edge features)
        mean_nbr = edge_feat.mean(dim=2)  # [B, N, dim] — mean over neighbors

        # Ψ₂ = variance
        diff = edge_feat - mean_nbr.unsqueeze(2)  # [B, N, N, dim]
        var_nbr = (diff**2).mean(dim=2)  # [B, N, dim]

        # Ψ₃ = skewness (normalized 3rd moment)
        var_eps = var_nbr + 1e-6
        skew_nbr = (diff**3).mean(dim=2) / (var_eps**1.5)  # [B, N, dim]

        # Stack moments and weight them
        moments = torch.stack([mean_nbr, var_nbr, skew_nbr], dim=-1)  # [B, N, dim, M]
        weighted = torch.einsum("bndm,bnm->bnd", moments, moment_weights)  # [B, N, dim]

        return self.update_proj(weighted)  # [B, N, dim]


# ==============================================================================
# DiT-style Transformer Block (AdaLN + ReSA + FMA + Cross-Attn)
# ==============================================================================


class FlowSGTransformerBlock(nn.Module):
    """Single DiT-style block for FlowSG denoiser (§4.3).

    Contains:
      1. AdaLN → ReSA (relation-modulated self-attention)
      2. AdaLN → Cross-Attention (to image features)
      3. AdaLN → FMA (flow-conditioned message aggregation)
      4. AdaLN → FFN

    Time code φ(t) modulates all blocks via AdaLN.
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        time_dim: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim

        # AdaLN layers (one per sub-block)
        self.adaln_resa = AdaLN(dim, time_dim)
        self.adaln_cross = AdaLN(dim, time_dim)
        self.adaln_fma = AdaLN(dim, time_dim)
        self.adaln_ffn = AdaLN(dim, time_dim)

        # Sub-layers
        self.resa = RelationModulatedSelfAttention(dim, num_heads, dropout)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.fma = FlowConditionedMessageAggregation(dim, time_dim)

        # Feed-forward network
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

        # Edge update MLP (refines edges from node pair features + time)
        # Paper §4.3 Eq.: z_ij = [h_i, h_j, h_i-h_j, M_i, M_j, φ(t)]
        self.edge_update = nn.Sequential(
            nn.Linear(dim * 5 + time_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Edge FiLM gate + projection (§4.3 Eq.(17))
        # e_ij^(ℓ+1) = FiLM(Emb(e_ij^(ℓ+1))) + W_e · e_ij^(ℓ+1)
        self.edge_norm = nn.LayerNorm(dim)
        self.edge_film = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
        )
        self.edge_fc = nn.Linear(dim, dim)

    def forward(
        self,
        node_feat: Tensor,  # [B, N, dim]
        edge_feat: Tensor,  # [B, N, N, dim]
        image_feat: Tensor,  # [B, L, dim] — CLIP image features
        image_mask: Optional[Tensor],  # [B, L] or None
        t_emb: Tensor,  # [B, time_dim]
    ) -> Tuple[Tensor, Tensor]:
        B, N, D = node_feat.shape

        # 1. ReSA (Relation-modulated Self-Attention)
        node_feat = node_feat + self.resa(
            self.adaln_resa(node_feat, t_emb.unsqueeze(1)),
            edge_feat,
        )

        # 2. Cross-Attention to image features
        q = self.adaln_cross(node_feat, t_emb.unsqueeze(1))
        if image_mask is not None and image_mask.any():
            cross_out, _ = self.cross_attn(
                q, image_feat, image_feat, key_padding_mask=image_mask
            )
        else:
            cross_out, _ = self.cross_attn(q, image_feat, image_feat)
        node_feat = node_feat + cross_out

        # 3. FMA (Flow-conditioned Message Aggregation)
        # Capture per-node messages M_i(t) for edge update coupling.
        fma_in = self.adaln_fma(node_feat, t_emb.unsqueeze(1))
        msg_i = self.fma(fma_in, edge_feat, t_emb)  # [B, N, dim] — Eq.(17)
        node_feat = node_feat + msg_i

        # 4. FFN
        node_feat = node_feat + self.ffn(self.adaln_ffn(node_feat, t_emb.unsqueeze(1)))

        # 5. Edge update (from refined node pairs + FMA messages + time context)
        # z_ij = [h_i, h_j, h_i-h_j, M_i, M_j, φ(t)]  — §4.3 Eq.(17)
        h_i = node_feat.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
        h_j = node_feat.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]
        h_diff = h_i - h_j
        m_i = msg_i.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
        m_j = msg_i.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]
        t_edge = t_emb.unsqueeze(1).unsqueeze(1).expand(-1, N, N, -1)

        z_ij = torch.cat(
            [h_i, h_j, h_diff, m_i, m_j, t_edge], dim=-1
        )  # [B, N, N, 5*D + time_dim]
        edge_feat_raw = edge_feat + self.edge_update(z_ij)

        # §4.3 Eq.(17): e_ij^(ℓ+1) = FiLM(Emb(e)) + W_e · e
        edge_norm = self.edge_norm(edge_feat_raw)
        film_params = self.edge_film(edge_norm)
        scale, bias = film_params.chunk(2, dim=-1)
        e_filmed = edge_feat_raw * (1.0 + scale.sigmoid()) + bias
        edge_feat = e_filmed + self.edge_fc(edge_feat_raw)

        return node_feat, edge_feat


# ==============================================================================
# Full Denoiser: Graph Transformer with stacked DiT blocks
# ==============================================================================


class FlowSGDenoiser(nn.Module):
    """FlowSG Denoiser: L DiT-style blocks with time conditioning.

    Args:
        dim: hidden dimension (512 in paper)
        num_heads: attention heads (8 in paper)
        num_blocks: number of DiT blocks (5 in paper)
        time_dim: time embedding dimension
        mlp_ratio: FFN expansion ratio
        dropout: dropout rate (0.1 in paper)
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        num_blocks: int = 5,
        time_dim: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Stacked DiT blocks
        self.blocks = nn.ModuleList(
            [
                FlowSGTransformerBlock(dim, num_heads, time_dim, mlp_ratio, dropout)
                for _ in range(num_blocks)
            ]
        )

        # Final AdaLN + output projections
        self.final_adaln = AdaLN(dim, time_dim)

    def forward(
        self,
        node_emb: Tensor,  # [B, N, dim] — initial node embeddings
        edge_emb: Tensor,  # [B, N, N, dim] — initial edge embeddings
        image_feat: Tensor,  # [B, L, dim] — CLIP image features
        image_mask: Optional[Tensor],  # [B, L]
        t: Tensor,  # [B] ∈ [0,1]
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            node_feat: [B, N, dim] refined node features
            edge_feat: [B, N, N, dim] refined edge features
        """
        # Time embedding
        t_emb = self.time_embed(t)  # [B, time_dim]

        node_feat = node_emb
        edge_feat = edge_emb

        for block in self.blocks:
            node_feat, edge_feat = block(
                node_feat,
                edge_feat,
                image_feat,
                image_mask,
                t_emb,
            )

        # Final normalization
        node_feat = self.final_adaln(node_feat, t_emb.unsqueeze(1))

        return node_feat, edge_feat


# ==============================================================================
# Output Heads (Geometry + Semantics)
# ==============================================================================


class GeometryHead(nn.Module):
    """Predicts velocity field for continuous box coordinates (§4.2, Eq.14).

    v_θ(g_t, t, C) — regresses conditional velocity field.
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 4),  # velocity for (cx, cy, w, h)
        )

    def forward(self, node_feat: Tensor) -> Tensor:
        """node_feat: [B, N, dim] → velocity: [B, N, 4]"""
        return self.net(node_feat)


class SemanticHead(nn.Module):
    """Predicts clean posteriors for discrete tokens (§4.2, Eq.19).

    For each slot, predicts time-conditioned p_{1|t}(clean | G_t, C).
    """

    def __init__(
        self,
        dim: int = 512,
        num_classes: int = 151,
        num_predicates: int = 51,
        codebook_size: int = 64,
        num_slots: int = 4,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.codebook_size = codebook_size
        self.obj_class_head = nn.Linear(dim, num_classes)  # object classes
        self.pred_head = nn.Linear(dim, num_predicates)  # predicate classes
        self.app_head = nn.Linear(
            dim, codebook_size * num_slots
        )  # per-slot appearance codes

    def forward(self, node_feat: Tensor, edge_feat: Tensor) -> Dict[str, Tensor]:
        """
        Returns:
            dict with keys:
              obj_logits: [B, N, num_classes]
              pred_logits: [B, N, N, num_predicates]
              app_logits: [B, N, M, codebook_size] — per-slot appearance codes
        """
        app_raw = self.app_head(node_feat)  # [B, N, M * codebook_size]
        B, N = node_feat.shape[:2]
        app_logits = app_raw.reshape(B, N, self.num_slots, self.codebook_size)
        return {
            "obj_logits": self.obj_class_head(node_feat),
            "pred_logits": self.pred_head(edge_feat),
            "app_logits": app_logits,
        }
