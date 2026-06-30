"""RA-SGG method wrapper for PyTorch Lightning."""

from .motifs_method import Motifs_Method, MotifsCriterion
from src.models.ra_sgg import build_ra_sgg


class RA_SGG_Method(Motifs_Method):
    """Minimal RA-SGG/ReTAG adapter with Motifs-compatible data flow."""

    def _build_model(self, **args):
        return build_ra_sgg(self.hparams)

    def _build_criterion(self, **args):
        return MotifsCriterion(num_predicates=args.get("rel_nums", 51))
