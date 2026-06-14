"""
PE-NET method wrapper for PyTorch Lightning.
"""
from .motifs_method import Motifs_Method, MotifsCriterion
from src.models.penet import build_penet


class PENet_Method(Motifs_Method):
    """PE-NET Lightning method — same data flow as Motifs."""

    def _build_model(self, **args):
        return build_penet(self.hparams)

    def _build_criterion(self, **args):
        return MotifsCriterion(
            num_predicates=args.get('rel_nums', 51),
        )
