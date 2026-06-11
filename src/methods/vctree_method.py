"""
VCTree method wrapper for PyTorch Lightning.

Integrates the VCTree model (Tang et al., CVPR 2019) into OpenSGG.

Supports PredCLS and SGCLS modes. Uses dynamic tree construction
for hierarchical context encoding instead of BiLSTM.
"""
import torch
from .motifs_method import Motifs_Method, MotifsCriterion
from src.models.vctree import build_vctree


class VCTree_Method(Motifs_Method):
    """VCTree Lightning method.

    Inherits most logic from Motifs_Method; only overrides model construction
    to use the TreeLSTM-based VCTree architecture.
    """

    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **args):
        return build_vctree(self.hparams)
