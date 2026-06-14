"""
Transformer SGG method wrapper for PyTorch Lightning.
"""
from .motifs_method import Motifs_Method, MotifsCriterion
from src.models.transformer_sgg import build_transformer_sgg


class TransformerSGG_Method(Motifs_Method):
    """Transformer SGG Lightning method — same data flow as Motifs."""

    def _build_model(self, **args):
        return build_transformer_sgg(self.hparams)

    def _build_criterion(self, **args):
        return MotifsCriterion(
            num_predicates=args.get('rel_nums', 51),
        )
