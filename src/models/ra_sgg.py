"""Minimal RA-SGG adapter for OpenSGG.

RA-SGG/ReTAG augments a PENet-style predicate predictor with retrieval from a
relation embedding memory bank. This module keeps that interface explicit while
falling back to the PENet predictor when no memory bank is available.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

from .penet import PENetContext


class RASGGModel(nn.Module):
    """OpenSGG-compatible RA-SGG relation head."""

    def __init__(
        self,
        num_classes: int = 151,
        num_predicates: int = 51,
        visual_dim: int = 2048,
        hidden_dim: int = 512,
        embed_dim: int = 200,
        use_freq_bias: bool = True,
        freq_bias_eps: float = 1e-12,
        dropout: float = 0.1,
        memory_bank_path: Optional[str] = None,
        retrieval_logit_coef: float = 0.0,
    ):
        super().__init__()
        self.num_predicates = num_predicates
        self.retrieval_logit_coef = retrieval_logit_coef
        self.memory_bank_path = memory_bank_path
        self.base = PENetContext(
            num_classes=num_classes,
            num_predicates=num_predicates,
            visual_dim=visual_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            use_freq_bias=use_freq_bias,
            freq_bias_eps=freq_bias_eps,
            dropout=dropout,
        )
        self.register_buffer("memory_distribution", torch.empty(0), persistent=False)
        self._load_memory_distribution(memory_bank_path)

    def _load_memory_distribution(self, path: Optional[str]) -> None:
        if not path or not os.path.isfile(path):
            return
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return
        dist = payload.get("predicate_distribution") if isinstance(payload, dict) else payload
        if torch.is_tensor(dist) and dist.numel() == self.num_predicates:
            dist = dist.float().clamp(min=0)
            if float(dist.sum()) > 0:
                self.memory_distribution = dist / dist.sum()

    def forward(self, visual_feats, boxes, labels, return_obj_preds=False, **kwargs):
        out = self.base(visual_feats, boxes, labels, return_obj_preds=return_obj_preds)
        if (
            self.retrieval_logit_coef > 0
            and self.memory_distribution.numel() == self.num_predicates
            and out.get("rel_logits") is not None
            and out["rel_logits"].numel() > 0
        ):
            prior = torch.log(
                self.memory_distribution.to(out["rel_logits"].device).clamp(min=1e-12)
            )
            out["rel_logits"] = out["rel_logits"] + self.retrieval_logit_coef * prior
            out["retrieval_fallback"] = "memory_distribution"
        else:
            out["retrieval_fallback"] = "penet_no_memory"
        return out


def build_ra_sgg(args) -> RASGGModel:
    return RASGGModel(
        num_classes=getattr(args, "entity_nums", 151),
        num_predicates=getattr(args, "rel_nums", 51),
        visual_dim=getattr(args, "visual_dim", 2048),
        hidden_dim=getattr(args, "hidden_dim", 512),
        embed_dim=getattr(args, "ra_sgg_embed_dim", 200),
        use_freq_bias=getattr(args, "use_freq_bias", True),
        freq_bias_eps=getattr(args, "freq_bias_eps", 1e-12),
        dropout=getattr(args, "dropout", 0.1),
        memory_bank_path=getattr(args, "ra_sgg_memory_bank_path", None),
        retrieval_logit_coef=getattr(args, "ra_sgg_retrieval_logit_coef", 0.0),
    )
