# src/modules/flowsg/__init__.py
"""
FlowSG: Progressive Image-Conditioned Scene Graph Generation with Flow Matching.

CVPR 2026 — Hu, Qin, Yin, Li, Li, He

This module contains the FlowSG model components:
  - vqvae.py: Slotwise VQ-VAE for scene graph tokenization (§4.1)
  - flow_matching.py: Continuous + Discrete Flow Matching (§3, §4.2)
  - graph_transformer.py: DiT-style denoiser with AdaLN, ReSA, FMA (§4.3)
"""

from .vqvae import VectorQuantizer, SlotwiseVQVAE
from .flow_matching import (
    ContinuousFlowMatching,
    DiscreteFlowMatching,
    ODESolver,
)
from .graph_transformer import (
    FlowSGDenoiser,
    FlowSGTransformerBlock,
    RelationModulatedSelfAttention,
    FlowConditionedMessageAggregation,
    GeometryHead,
    SemanticHead,
    AdaLN,
)

# Legacy aliases for backward compatibility
ConditionalFlowMatchingLoss = ContinuousFlowMatching
FlowMatchingODESolver = ODESolver

__all__ = [
    "VectorQuantizer",
    "SlotwiseVQVAE",
    "ContinuousFlowMatching",
    "DiscreteFlowMatching",
    "ODESolver",
    "FlowSGDenoiser",
    "FlowSGTransformerBlock",
    "RelationModulatedSelfAttention",
    "FlowConditionedMessageAggregation",
    "GeometryHead",
    "SemanticHead",
    "AdaLN",
    "ConditionalFlowMatchingLoss",
    "FlowMatchingODESolver",
]
