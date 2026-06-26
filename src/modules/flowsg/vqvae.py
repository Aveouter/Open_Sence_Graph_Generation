# src/modules/flowsg/vqvae.py
"""
VQ-VAE for scene graph tokenization — FlowSG §4.1.

Object appearance tokenizer:
  - CLIP cropped region features u_i ∈ R^d
  - Codebook H_obj = {e_k}^{K_a}_{k=1}, K_a=64, dim=512
  - M=4 ordered slots for factorization (InstructScene-style [36])
  - Quantize: a*_i = argmin_k ||u_i - e_k||^2

Relation predicate tokenizer:
  - CLIP text embeddings for relation words
  - Codebook H_rel, K_r=64, dim=512
  - M=4 ordered slots

Training: VQ-VAE objective (reconstruction + codebook + commitment loss)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict


class VectorQuantizer(nn.Module):
    """Vector Quantization with straight-through estimator.

    Args:
        num_embeddings: codebook size K (64 in paper)
        embedding_dim: code dimension d (512 in paper)
        commitment_cost: beta for commitment loss
    """

    def __init__(
        self,
        num_embeddings: int = 64,
        embedding_dim: int = 512,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            z: [..., D] continuous inputs
        Returns:
            z_q: [..., D] quantized
            indices: [...] codebook indices
            vq_loss: scalar
            perplexity: scalar
        """
        flat_z = z.reshape(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_z**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2.0 * torch.matmul(flat_z, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        z_q = torch.matmul(encodings, self.embedding.weight)
        z_q = z_q.reshape(z.shape)

        # VQ losses
        # codebook_loss: pulls codebook toward encoder output (sg[z_e] - e)^2
        codebook_loss = F.mse_loss(z.detach(), z_q)
        # commitment_loss: pulls encoder output toward codebook (z_e - sg[e])^2
        commitment_loss = F.mse_loss(z, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through
        z_q = z + (z_q - z).detach()

        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, encoding_indices.reshape(z.shape[:-1]), vq_loss, perplexity


class SlotwiseVQVAE(nn.Module):
    """Multi-slot VQ-VAE for appearance/relation tokenization (§4.1).

    Uses M=4 ordered slots to factorize the latent code, following [36].
    Each slot has its own codebook of size K=64 with dimension d=512.

    For appearance tokens:
      u_i = CLIP_img(crop(I, b_i))  →  quantize via M slots  →  a_i = [idx_1..idx_M]
    """

    def __init__(
        self,
        num_slots: int = 4,  # M=4 ordered slots
        codebook_size: int = 64,  # K=64 entries per slot
        embedding_dim: int = 512,  # d=512
        input_dim: int = 512,  # CLIP ViT-B/16 output dim
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim

        # Per-slot encoders: project input → code_dim
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, embedding_dim),
                    nn.LayerNorm(embedding_dim),
                )
                for _ in range(num_slots)
            ]
        )

        # Per-slot quantizers
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(codebook_size, embedding_dim, commitment_cost)
                for _ in range(num_slots)
            ]
        )

        # Decoder for reconstruction check
        self.decode_proj = nn.Linear(embedding_dim * num_slots, input_dim)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode continuous features through M slots.

        Args:
            x: [B, N, input_dim] input features (CLIP region features)

        Returns:
            z_q_stack: [B, N, M, embedding_dim] quantized per slot
            indices: [B, N, M] codebook indices per slot
            total_vq_loss: scalar
            avg_perplexity: scalar
        """
        B, N, _ = x.shape
        z_q_list = []
        idx_list = []
        total_vq_loss = 0.0
        total_perp = 0.0

        for m in range(self.num_slots):
            z = self.encoders[m](x)  # [B, N, embedding_dim]
            z_q, indices, vq_loss, perp = self.quantizers[m](z)
            z_q_list.append(z_q)
            idx_list.append(indices)
            total_vq_loss += vq_loss
            total_perp += perp

        z_q_stack = torch.stack(z_q_list, dim=2)  # [B, N, M, D]
        indices_stack = torch.stack(idx_list, dim=2)  # [B, N, M]
        return (
            z_q_stack,
            indices_stack,
            total_vq_loss / self.num_slots,
            total_perp / self.num_slots,
        )

    def decode(self, z_q_stack: Tensor) -> Tensor:
        """Decode quantized slot vectors back to feature space.

        Args:
            z_q_stack: [B, N, M, D]
        Returns:
            rec: [B, N, input_dim]
        """
        B, N, M, D = z_q_stack.shape
        flat = z_q_stack.reshape(B, N, M * D)
        return self.decode_proj(flat)

    def decode_indices(self, indices: Tensor) -> Tensor:
        """Decode directly from slot indices.

        Args:
            indices: [B, N, M] integer indices
        Returns:
            z_q_stack: [B, N, M, D]
        """
        z_q_list = []
        for m in range(self.num_slots):
            z_q = self.quantizers[m].embedding(indices[..., m])
            z_q_list.append(z_q)
        return torch.stack(z_q_list, dim=2)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Full VQ-VAE forward.

        Args:
            x: [B, N, D] continuous features
        Returns:
            dict with: z_q_stack, indices, vq_loss, perplexity, rec_features, rec_loss
        """
        z_q_stack, indices, vq_loss, perplexity = self.encode(x)
        rec = self.decode(z_q_stack)
        rec_loss = F.mse_loss(rec, x)
        return {
            "z_q_stack": z_q_stack,  # [B, N, M, D]
            "indices": indices,  # [B, N, M]
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "rec_features": rec,
            "rec_loss": rec_loss,
        }
