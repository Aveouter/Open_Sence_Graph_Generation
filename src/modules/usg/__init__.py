# src/modules/usg/__init__.py
"""
USG-Par: Universal Scene Graph Parser (CVPR 2025).

Wu, Fei, Chua — "Universal Scene Graph Generation"

Modules for VG-only (image) scene graph generation:
  - mask_decoder.py: Shared Mask Decoder with object queries and cross-attention
  - rpc.py: Relation Proposal Constructor (bidirectional cross-attention + top-K)
  - relation_decoder.py: Transformer relation decoder
"""

from .mask_decoder import USGMaskDecoder, USGObjectHead
from .rpc import RelationProposalConstructor
from .relation_decoder import USGRelationDecoder
