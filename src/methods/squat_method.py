"""
SQUAT method wrapper for PyTorch Lightning.
"""
from .motifs_method import Motifs_Method, MotifsCriterion
from src.models.squat import build_squat


class Squat_Method(Motifs_Method):
    """SQUAT Lightning method — same data flow as Motifs."""

    def _build_model(self, **args):
        return build_squat(self.hparams)

    def _build_criterion(self, **args):
        return MotifsCriterion(
            num_predicates=args.get('rel_nums', 51),
        )
