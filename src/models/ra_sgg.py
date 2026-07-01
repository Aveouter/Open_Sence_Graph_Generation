"""OpenSGG RA-SGG/ReTAG adapter.

This module is intentionally conservative: it exposes an OpenSGG-compatible
RA-SGG method entry point while documenting that the default path is a
PENet-style no-memory adapter, not official ReTAG parity. Official ReTAG
requires a pretrained PE-Net checkpoint plus feature-bank retrieval artifacts.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .penet import PENetContext


class RASGGModel(nn.Module):
    """Minimal RA-SGG-compatible relation head for audit/smoke workflows."""

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
        self.memory_bank_path = memory_bank_path
        self.retrieval_logit_coef = retrieval_logit_coef
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
        """Load an optional predicate prior without claiming ReTAG retrieval."""
        if not path:
            return
        memory_path = Path(path)
        if not memory_path.is_file():
            return

        try:
            try:
                payload = torch.load(
                    memory_path, map_location="cpu", weights_only=False
                )
            except TypeError:
                payload = torch.load(memory_path, map_location="cpu")
        except (
            EOFError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ) as exc:
            warnings.warn(
                f"RA-SGG memory distribution could not be loaded from "
                f"{memory_path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        distribution = (
            payload.get("predicate_distribution")
            if isinstance(payload, dict)
            else payload
        )
        if not torch.is_tensor(distribution):
            warnings.warn(
                f"RA-SGG memory distribution file {memory_path} does not contain "
                "a tensor or 'predicate_distribution' tensor.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        if distribution.numel() != self.num_predicates:
            warnings.warn(
                f"RA-SGG memory distribution in {memory_path} has "
                f"{distribution.numel()} values, expected {self.num_predicates}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        distribution = distribution.float().reshape(-1).clamp(min=0)
        total = float(distribution.sum())
        if total > 0:
            self.memory_distribution = distribution / total

    def forward(self, visual_feats, boxes, labels, return_obj_preds=False, **kwargs):
        out = self.base(
            visual_feats,
            boxes,
            labels,
            return_obj_preds=return_obj_preds,
        )
        rel_logits = out.get("rel_logits")
        if (
            self.retrieval_logit_coef > 0
            and self.memory_distribution.numel() == self.num_predicates
            and rel_logits is not None
            and rel_logits.numel() > 0
        ):
            prior = torch.log(
                self.memory_distribution.to(rel_logits.device).clamp(min=1e-12)
            )
            out["rel_logits"] = rel_logits + self.retrieval_logit_coef * prior
            out["ra_sgg_adapter_status"] = "predicate_distribution_prior"
        else:
            out["ra_sgg_adapter_status"] = "penet_no_memory"
        return out


def build_ra_sgg(args) -> RASGGModel:
    """Build the conservative RA-SGG adapter from OpenSGG config args."""
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
