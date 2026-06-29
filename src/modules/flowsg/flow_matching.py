# src/modules/flowsg/flow_matching.py
"""
Hybrid Flow Matching for FlowSG (§3, §4.2).

Continuous Flow Matching (CFM) for bounding boxes:
  - Interpolant: g_t = (1-κ_t)·g_0 + κ_t·g_1  with κ_t = 1 - cos(πt/2)
  - Target velocity: u* = ∂_t ψ_t = κ̇_t · (g_1 - g_0)
  - Loss: ||v_θ(g_t, t, C) - u*||^2

Discrete Flow Matching (DFM) for categorical tokens:
  - Two-point path: p_t = (1-κ_t)·δ_{mask} + κ_t·δ_{clean}
  - Cross-entropy: CE(g_1, f_θ(g_t, t, C))

ODE Solver for inference.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable, Tuple


# ==============================================================================
# Cosine Scheduler — §4.2 "κ_t = 1 - cos(πt/2)"
# ==============================================================================


def cosine_schedule(t: Tensor) -> Tuple[Tensor, Tensor]:
    """Cosine noise schedule for flow matching.

    κ_t = 1 - cos(π·t / 2)
    κ̇_t = (π/2) · sin(π·t / 2)

    Args:
        t: [B] time ∈ [0, 1]

    Returns:
        kappa: [B, 1, 1] schedule value κ_t
        kappa_dot: [B, 1, 1] time derivative κ̇_t
    """
    t = t.float()
    kappa = 1.0 - torch.cos(math.pi * t / 2.0)
    kappa_dot = (math.pi / 2.0) * torch.sin(math.pi * t / 2.0)
    # Expand for broadcasting with [B, N, 4] tensors
    return kappa.view(-1, 1, 1), kappa_dot.view(-1, 1, 1)


# ==============================================================================
# Continuous Flow Matching (CFM) — §3 Eq.(2), §4.2 Eq.(13-14, 18)
# ==============================================================================


class ContinuousFlowMatching(nn.Module):
    """Conditional Flow Matching for bounding box geometry.

    g_0 ~ N(0, I)  (standard Gaussian prior)
    g_1 = boxes    (ground-truth, normalized [0,1])

    Interpolant:  g_t = (1-κ_t)·g_0 + κ_t·g_1
    Velocity:     u* = κ̇_t · (g_1 - g_0)

    Loss: L_CFM = ||v_θ(g_t, t, C) - κ̇_t·(g_1 - g_0)||^2
    """

    def __init__(self):
        super().__init__()

    def sample_prior(self, shape, device, dtype=None) -> Tensor:
        """g_0 ~ N(0, I_4)"""
        return torch.randn(shape, device=device, dtype=dtype)

    def interpolate(
        self, g_0: Tensor, g_1: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build interpolated state and target velocity.

        Args:
            g_0: [B, N, 4] Gaussian noise (prior)
            g_1: [B, N, 4] target boxes (cx, cy, w, h, normalized)
            t:   [B] time ∈ [0, 1]

        Returns:
            g_t: [B, N, 4] interpolated boxes
            u_star: [B, N, 4] target velocity = κ̇_t·(g_1 - g_0)
            kappa: [B, 1, 1] schedule value
            kappa_dot: [B, 1, 1] schedule derivative
        """
        kappa, kappa_dot = cosine_schedule(t)
        g_t = (1.0 - kappa) * g_0 + kappa * g_1
        u_star = kappa_dot * (g_1 - g_0)
        return g_t, u_star, kappa, kappa_dot

    def loss(
        self,
        pred_velocity: Tensor,
        u_star: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """CFM loss: MSE between predicted and target velocity.

        Args:
            pred_velocity: [B, N, 4] predicted v_θ(g_t, t, C)
            u_star: [B, N, 4] target velocity
            mask: [B, N] bool (True = valid node)

        Returns:
            scalar loss
        """
        loss = F.mse_loss(pred_velocity, u_star, reduction="none").mean(
            dim=-1
        )  # [B, N]
        if mask is not None and mask.any():
            loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
        else:
            loss = loss.mean()
        return loss


# ==============================================================================
# Discrete Flow Matching (DFM) — §3 Eq.(4), §4.2 Eq.(15, 19)
# ==============================================================================


class DiscreteFlowMatching(nn.Module):
    """Discrete flow for categorical tokens via masked prediction.

    - Source p_0 = δ_mask (all [MASK] tokens)
    - Target p_1 = δ_clean (ground-truth tokens)
    - Path: p_t = (1-κ_t)·δ_mask + κ_t·δ_clean

    Training: sample tokens from p_t, predict clean tokens via CE.
    Loss: L_DFM = CE(z_1, f_θ(z_t, t, C))

    For object categories: use detector predictions as priors (not masked).
    For appearance codes: fully masked.
    For predicate labels: fully masked.
    """

    def __init__(self, mask_token_id: int = -1):
        super().__init__()
        self.mask_token_id = mask_token_id

    def sample_tokens(
        self,
        clean_tokens: Tensor,
        t: Tensor,
        mask_id: int,
        vocab_size: int,
    ) -> Tensor:
        """Sample tokens from the two-point path p_t.

        With probability (1-κ_t): emit [MASK]
        With probability κ_t: emit clean token

        Args:
            clean_tokens: [B, N] ground-truth token indices
            t: [B] time
            mask_id: integer mask token ID
            vocab_size: total vocabulary size

        Returns:
            sampled: [B, N] sampled token indices
        """
        kappa, _ = cosine_schedule(t)
        kappa = kappa.squeeze(-1).squeeze(-1).unsqueeze(1)  # [B, 1]

        # Bernoulli sampling: mask with prob (1-κ_t)
        mask_prob = (1.0 - kappa).expand_as(clean_tokens.float())
        rand = torch.rand_like(mask_prob)
        is_masked = rand < mask_prob

        sampled = clean_tokens.clone()
        sampled[is_masked] = mask_id
        return sampled

    def loss(
        self,
        pred_logits: Tensor,
        clean_tokens: Tensor,
        t: Tensor,
        mask: Optional[Tensor] = None,
        masked_positions: Optional[Tensor] = None,
    ) -> Tensor:
        """DFM loss: time-conditioned cross-entropy.

        Computes CE only on positions where the token was actually masked
        (i.e., where the model needs to predict the clean value from [MASK]).

        Args:
            pred_logits: [..., vocab_size] predicted clean posteriors
            clean_tokens: [...] ground-truth token indices
            t: [B] time (not used here — classifier is time-conditioned externally)
            mask: [...] bool (True = valid position, e.g. padding mask)
            masked_positions: [...] bool (True = position was actually masked by DFM).
                When provided, loss is averaged only over masked positions.
                When None, falls back to all positions (or mask-filtered).

        Returns:
            scalar loss
        """
        vocab_size = pred_logits.shape[-1]
        # Clamp tokens to valid range
        clean_tokens = clean_tokens.clamp(0, vocab_size - 1)

        loss = F.cross_entropy(
            pred_logits.reshape(-1, vocab_size),
            clean_tokens.reshape(-1).long(),
            reduction="none",
        ).reshape(clean_tokens.shape)

        # Primary filter: only compute loss on actually-masked positions
        if masked_positions is not None and masked_positions.any():
            loss = loss[masked_positions].mean()
        elif mask is not None and mask.any():
            loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
        else:
            loss = loss.mean()
        return loss


# ==============================================================================
# CFM + DFM Combined Wrapper
# ==============================================================================


class HybridFlowMatching(nn.Module):
    """Combined Continuous + Discrete flow matching.

    CFM for boxes (g), DFM for semantics (s).
    """

    def __init__(self):
        super().__init__()
        self.cfm = ContinuousFlowMatching()
        self.dfm = DiscreteFlowMatching()

    @staticmethod
    def sample_time(batch_size: int, device) -> Tensor:
        """t ~ U[0, 1]"""
        return torch.rand(batch_size, device=device)


# ==============================================================================
# ODE Solver for inference
# ==============================================================================


class ODESolver:
    """Euler ODE solver for CFM inference.

    Integrates dg/dt = v_θ(g_t, t, C) from t=0 to t=1.
    """

    def __init__(self, num_steps: int = 10):
        self.num_steps = num_steps

    def solve(
        self,
        velocity_fn: Callable[[Tensor, Tensor], Tensor],
        g_0: Tensor,
    ) -> Tensor:
        """Solve ODE with Euler method.

        Args:
            velocity_fn: (g_t, t) → predicted velocity [B, N, 4]
            g_0: [B, N, 4] initial noise

        Returns:
            g_1: [B, N, 4] boxes at t=1
        """
        B = g_0.shape[0]
        device = g_0.device
        dt = 1.0 / self.num_steps
        g_t = g_0

        for step in range(self.num_steps):
            t = step * dt
            t_tensor = torch.full((B,), t, device=device, dtype=torch.float32)
            v = velocity_fn(g_t, t_tensor)
            g_t = g_t + v * dt

        return g_t
