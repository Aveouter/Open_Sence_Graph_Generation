"""
IMP method wrapper for PyTorch Lightning.

Integrates the Iterative Message Passing model into the OpenSGG training framework.
Inherits from Motifs_Method since the data flow is identical.
"""
import torch
from .motifs_method import Motifs_Method, MotifsCriterion
from src.models.imp import build_imp


class IMP_Method(Motifs_Method):
    """IMP Lightning method — identical data flow to Motifs, different model."""

    def _build_model(self, **args):
        return build_imp(self.hparams)

    def _build_criterion(self, **args):
        return MotifsCriterion(
            num_predicates=args.get('rel_nums', 51),
        )
